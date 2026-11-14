"""Daily prompt (issue #22, STATUS.md B23): the time is inside the window and outside quiet hours,
the on-time window, pause/off, the demo trigger, and the columns stored on each check-in."""

import random
import statistics
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlmodel import Session, SQLModel

from app import baseline as baseline_mod
from app import db, prompt
from app.main import app
from app.models import CheckIn
from app.models_prompt import PromptSettings
from app.routers import checkins as checkins_router
from app.seed import seed

from .test_baseline import features

HEL = "Europe/Helsinki"


def minutes_of(st: PromptSettings, n: int = 3000, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [prompt.draw_minute(prompt.allowed_intervals(st), rng) for _ in range(n)]


# ---------------------------------------------------------------- the draw (pure)

def test_draw_is_uniform_between_09_and_21():
    ms = minutes_of(PromptSettings(patient_id=1))
    assert min(ms) >= 9 * 60 and max(ms) < 21 * 60
    assert statistics.fmean(ms) == pytest.approx(15 * 60, abs=10)  # uniform: mean at 15:00
    assert sum(m < 12 * 60 for m in ms) / len(ms) == pytest.approx(0.25, abs=0.03)  # 3 of 12 h


def test_draw_skips_quiet_hours():
    st = PromptSettings(patient_id=1, quiet_start="12:00", quiet_end="14:00")
    ms = minutes_of(st)
    assert not any(12 * 60 <= m < 14 * 60 for m in ms)
    assert min(ms) >= 9 * 60 and max(ms) < 21 * 60
    assert {m // 60 for m in ms} == set(range(9, 21)) - {12, 13}


def test_quiet_hours_over_midnight():
    st = PromptSettings(patient_id=1, quiet_start="20:00", quiet_end="10:00")
    assert prompt.allowed_intervals(st) == [(10 * 60, 20 * 60)]
    ms = minutes_of(st)
    assert min(ms) >= 10 * 60 and max(ms) < 20 * 60


def test_all_quiet_gives_no_prompt():
    st = PromptSettings(patient_id=1, quiet_start="08:00", quiet_end="22:00")
    assert prompt.draw_minute(prompt.allowed_intervals(st), random.Random(0)) is None


# ---------------------------------------------------------------- via the API

@pytest.fixture()
def pc(monkeypatch):
    """Client with a settable clock. Default: Mon 16 Nov 2026, 06:00 UTC = 08:00 in Helsinki."""
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    queue = []
    monkeypatch.setattr(checkins_router, "extract_features", lambda path: queue.pop(0))
    monkeypatch.setattr(checkins_router, "extract_text_features", lambda text: queue.pop(0))
    c = TestClient(app)
    c.queue = queue
    c.now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(baseline_mod, "clock", lambda: c.now)
    monkeypatch.setattr(prompt, "RNG", random.Random(3))
    return c


def checkin(c, kind="audio") -> dict:
    c.queue.append(features(0, random.Random(0)))
    if kind == "text":
        r = c.post("/api/checkins/text", json={"patient_id": 1, "slot": "morning", "text": "x"})
    else:
        r = c.post("/api/checkins", data={"patient_id": 1, "slot": "morning"}, files={"audio": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def test_todays_prompt_is_in_the_local_window_and_stable(pc):
    p = pc.get("/api/patients/1/prompt").json()
    assert p["date"] == "2026-11-16"
    assert "09:00" <= p["at_local"] < "21:00"
    assert p["state"] == "scheduled" and not p["open"]  # 08:00 local: before the window
    assert pc.get("/api/patients/1/prompt").json()["at"] == p["at"]  # drawn once per day


def test_prompt_respects_quiet_hours_set_by_the_person(pc):
    r = pc.put("/api/patients/1/prompt/settings", json={"quiet_start": "09:00", "quiet_end": "20:00"})
    assert r.status_code == 200
    assert "20:00" <= r.json()["at_local"] < "21:00"


def test_opens_at_its_time_with_two_hours_on_time(pc):
    at = datetime.fromisoformat(pc.get("/api/patients/1/prompt").json()["at"])
    pc.now = at + timedelta(minutes=30)
    p = pc.get("/api/patients/1/prompt").json()
    assert p["state"] == "open" and p["open"]
    assert p["on_time_left_s"] == 90 * 60


def test_on_time_answer_and_stored_columns(pc):
    pc.post("/api/patients/1/prompt/now")
    pc.now += timedelta(minutes=5)
    out = checkin(pc)
    assert out["prompt"]["answered"] and out["prompt"]["on_time"]
    with Session(db.engine) as s:
        ci = s.get(CheckIn, out["checkin_id"])
        assert ci.prompt_at is not None
        assert ci.local_hour == pytest.approx(8 + 5 / 60, abs=1e-3)  # Helsinki = UTC+2 in November
    p = pc.get("/api/patients/1/prompt").json()
    assert p["state"] == "answered" and p["on_time"] is True and not p["open"]
    pc.now += timedelta(minutes=5)
    again = checkin(pc)  # a second check-in that day answers nothing: the bonus is once a day
    assert again["prompt"] is None
    with Session(db.engine) as s:
        ci = s.get(CheckIn, again["checkin_id"])
        assert ci.prompt_at is None and ci.local_hour is not None


def test_late_checkin_counts_fully_without_on_time(pc):
    pc.post("/api/patients/1/prompt/now")
    pc.now += timedelta(hours=3)
    assert pc.get("/api/patients/1/prompt").json()["on_time_left_s"] == 0
    out = checkin(pc)
    assert out["prompt"]["on_time"] is False and out["prompt"]["late_by_s"] == 3600
    assert out["checkin_id"] and out["baseline"]["checkins"]["morning"] == 1  # stored like any other


def test_checkin_before_the_prompt_does_not_answer_it(pc):
    p = pc.get("/api/patients/1/prompt").json()
    out = checkin(pc)  # 08:00 local, prompt later today
    assert out["prompt"] is None
    pc.now = datetime.fromisoformat(p["at"]) + timedelta(minutes=1)
    assert pc.get("/api/patients/1/prompt").json()["open"]


def test_written_diary_answers_the_prompt_too(pc):
    pc.post("/api/patients/1/prompt/now")
    assert checkin(pc, kind="text")["prompt"]["on_time"]


def test_pause_today_and_turn_off(pc):
    r = pc.put("/api/patients/1/prompt/settings", json={"pause_today": True}).json()
    assert r["state"] == "paused" and r["settings"]["paused_today"]
    assert pc.post("/api/patients/1/prompt/now").status_code == 409
    assert checkin(pc)["prompt"] is None
    pc.now += timedelta(days=1)  # the pause covers one day only
    assert pc.get("/api/patients/1/prompt").json()["state"] in ("scheduled", "open")
    off = pc.put("/api/patients/1/prompt/settings", json={"enabled": False}).json()
    assert off["state"] == "off" and off["at"] is None
    with Session(db.engine) as s:
        assert prompt.is_quiet(s, 1, pc.now)


def test_demo_resend_opens_again(pc):
    pc.post("/api/patients/1/prompt/now")
    checkin(pc)
    p = pc.post("/api/patients/1/prompt/now").json()
    assert p["state"] == "open" and p["source"] == "demo" and p["on_time_left_s"] == 7200


def test_settings_validation(pc):
    assert pc.put("/api/patients/1/prompt/settings", json={"quiet_start": "25:00", "quiet_end": "07:00"}).status_code == 400
    assert pc.put("/api/patients/1/prompt/settings", json={"quiet_start": "22:00"}).status_code == 400
    assert pc.put("/api/patients/1/prompt/settings", json={"tz": "Mars/Olympus"}).status_code == 400
    assert pc.get("/api/patients/99/prompt").status_code == 404


def test_is_quiet_outside_the_window_and_in_quiet_hours(pc):
    with Session(db.engine) as s:
        assert prompt.is_quiet(s, 1, datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc))  # 07:00 local
        assert not prompt.is_quiet(s, 1, datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc))  # 12:00 local
        st = prompt.settings_for(s, 1)
        st.quiet_start, st.quiet_end = "11:30", "13:00"
        assert prompt.is_quiet(s, 1, datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------- old databases

def test_old_database_gets_the_new_nullable_columns(tmp_path, monkeypatch):
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as conn:  # CheckIn as it was before #22
        conn.execute(text("CREATE TABLE checkin (id INTEGER PRIMARY KEY, patient_id INTEGER, ts DATETIME, "
                          "slot VARCHAR, transcript VARCHAR, llm_json VARCHAR, risk_flag BOOLEAN)"))
        conn.execute(text("INSERT INTO checkin (patient_id, ts, slot, risk_flag) VALUES (1, '2026-11-01', 'morning', 0)"))
    monkeypatch.setattr(db, "engine", eng)
    db.add_missing_columns()
    cols = {c["name"] for c in inspect(eng).get_columns("checkin")}
    assert {"prompt_at", "local_hour", "speech_s"} <= cols
    with eng.connect() as conn:
        assert conn.execute(text("SELECT count(*), max(prompt_at) FROM checkin")).one() == (1, None)
