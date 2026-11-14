"""Healthy-baseline lifecycle and scoring (STATUS.md T1). Audio analysis is replaced by controlled
feature values, so these tests check the engine, not the signal processing."""

import json
import random
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import baseline as baseline_mod
from app import db
from app.engine import state
from app.main import app
from app.models import FeatureSource
from app.pipeline import PipelineResult
from app.routers import checkins as checkins_router
from app.seed import seed

STATS = state.load_branch("acoustic").meta["feature_stats"]
SOURCE = {"arousal": FeatureSource.emotion, "valence": FeatureSource.emotion}


def text_features(bad: bool, rng: random.Random) -> PipelineResult:
    """Mocked written diary (text branch) – Mahdan's real model is tested in test_text.py."""
    res = PipelineResult(transcript="(mock diary)")
    vals = {"text_logit": 4.0, "absolutist": 3.0, "first_person": 10.0, "sentiment": -0.8} if bad else {
        "text_logit": rng.gauss(-2, 0.5), "absolutist": abs(rng.gauss(0.5, 0.3)),
        "first_person": rng.gauss(5, 1), "sentiment": rng.gauss(0.5, 0.2)}
    res.features = [(f, v, FeatureSource.text) for f, v in vals.items()]
    return res


def features(depression: float, rng: random.Random) -> PipelineResult:
    """depression 0 = like corpus controls, 1 = like corpus patients; plus within-person noise."""
    res = PipelineResult()
    for f, st in STATS.items():
        v = st["control_mean"] + depression * (st["patient_mean"] - st["control_mean"]) + rng.gauss(0, st["within_sd"] * 0.5)
        res.features.append((f, v, SOURCE.get(f, FeatureSource.prosody)))
    return res


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    queue: list[PipelineResult] = []
    monkeypatch.setattr(checkins_router, "extract_features", lambda path: queue.pop(0))
    monkeypatch.setattr(checkins_router, "extract_text_features", lambda text: queue.pop(0))
    c = TestClient(app)
    c.queue = queue
    # Controllable clock: each check-in is a new day unless same_day=True
    c.day = 0
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    monkeypatch.setattr(baseline_mod, "clock", lambda: start + timedelta(days=c.day, seconds=c.tick))
    c.tick = 0
    return c


def checkin(c, res: PipelineResult, slot="morning", same_day=False, kind="audio") -> dict:
    if not same_day and c.tick:
        c.day += 1
    c.tick += 1
    c.queue.append(res)
    if kind == "text":
        r = c.post("/api/checkins/text", json={"patient_id": 1, "slot": slot, "text": "x"})
    else:
        r = c.post("/api/checkins", data={"patient_id": 1, "slot": slot}, files={"audio": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- pure statistics

def test_robust_stats_ignores_one_outlier():
    vals = [1.0, 1.1, 0.9, 1.05, 0.95, 50.0]
    center, scale = state.robust_stats(vals, prior_sd=None, k=5)
    assert 0.9 < center < 1.1
    assert scale < 1.0  # mean/SD would give ~18


def test_robust_stats_shrinks_toward_corpus_sd():
    vals = [1.0] * 4  # MAD = 0 → personal SD unknowable from 4 identical values
    _, scale = state.robust_stats(vals, prior_sd=2.0, k=5)
    assert scale == pytest.approx((5 * 4 / 9) ** 0.5)  # (n·0 + k·4) / (n + k)
    _, big_n = state.robust_stats([1.0, 3.0] * 200, prior_sd=2.0, k=5)
    assert big_n == pytest.approx(1.4826, rel=0.05)  # personal data dominates with many check-ins


# ---------------------------------------------------------------- lifecycle via the API

def test_first_checkin_starts_building_and_gives_no_score(client):
    out = checkin(client, features(0, random.Random(0)))
    assert out["baseline"]["status"] == "building"
    assert out["baseline"]["period_days"] == 14
    assert out["state"]["S"] is None


def test_locks_after_period_then_scores_and_catches_change(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    rng = random.Random(1)
    for _ in range(5):
        out = checkin(client, features(0, rng))
        assert out["baseline"]["status"] == "building" and out["state"]["S"] is None
    out = checkin(client, features(0, rng))  # 6th check-in → enough → locks
    assert out["baseline"]["status"] == "active"
    assert out["state"]["S_anchored"] is not None and abs(out["state"]["S_anchored"]) < 2
    assert out["state"]["S_rolling"] is None  # needs rolling_min_n same-slot check-ins first

    healthy_S = out["state"]["S_anchored"]
    for _ in range(4):  # four days of a full control→patient shift
        out = checkin(client, features(1.0, rng))
    # ~1.5 SD per day for a full shift (ml/reports/acoustic/summary.md); the EWMA needs a few days
    assert out["state"]["S_anchored"] > 1.0
    assert out["state"]["S_anchored"] - healthy_S > 0.8

    for _ in range(4):  # far beyond the corpus difference → S > 2 → clinician
        out = checkin(client, features(2.5, rng))
    assert out["state"]["S_anchored"] > 2
    assert out["state"]["level"] >= 2

    detail = client.get("/api/patients/1/baseline").json()
    assert detail["active"]["version"] == 1
    # B30: one pooled baseline plus the time-of-day slope; no morning/evening slots
    assert {"all", "hour"} <= set(detail["stats"])
    assert "morning" not in detail["stats"] and "evening" not in detail["stats"]


def test_new_baseline_keeps_old_one_active_until_it_locks(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    rng = random.Random(2)
    for _ in range(3):
        checkin(client, features(0, rng))
    client.post("/api/patients/1/baseline/start", json={"period_days": 14, "min_checkins": 3, "note": "new phone"})
    d = client.get("/api/patients/1/baseline").json()
    assert d["active"]["version"] == 1 and d["building"]["version"] == 2
    out = checkin(client, features(0, rng))
    assert out["state"]["S_anchored"] is not None  # still scored against v1


def test_clinician_reject_starts_over_and_confirm_is_recorded(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    rng = random.Random(3)
    for _ in range(3):
        checkin(client, features(0, rng))
    d = client.post("/api/patients/1/baseline/confirm", json={"clinician": "Dr. Virtanen"}).json()
    assert d["active"]["confirmed_by"] == "Dr. Virtanen"
    d = client.post("/api/patients/1/baseline/reject", json={"note": "had flu"}).json()
    assert d["active"] is None and d["building"]["version"] == 2
    out = checkin(client, features(0, rng))
    assert out["state"]["S"] is None  # no active baseline → no score until the new one locks


# ---------------------------------------------------------------- day combination and baseline edge cases (B16, B17)

def _lock_mixed_baseline(client, rng, days=6):
    """A spoken and a written check-in every day; locks at the 6th check-in."""
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 2 * days})
    for _ in range(days):
        checkin(client, features(0, rng))
        out = checkin(client, text_features(False, rng), same_day=True, kind="text")
    assert out["baseline"]["status"] == "active"


def test_same_day_checkins_are_combined_not_overwritten(client):
    rng = random.Random(4)
    _lock_mixed_baseline(client, rng)
    bad = checkin(client, text_features(True, rng), kind="text")  # new day: a very negative diary
    D_text = bad["state"]["branches"]["text"]
    assert D_text > 3
    later = checkin(client, features(0, rng), same_day=True)  # neutral spoken check-in, same day
    assert later["state"]["branches"]["text"] == pytest.approx(D_text)  # the diary still counts
    assert "acoustic" in later["state"]["branches"]
    assert later["state"]["S"] > 1  # not replaced by the neutral voice-only value


def test_same_branch_twice_a_day_is_combined_stouffer(client):
    """B16, with B31: the second, unprompted check-in of the day counts at half weight."""
    rng = random.Random(5)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    for _ in range(6):
        checkin(client, features(0, rng))
    a = checkin(client, features(0.8, rng))["state"]["branches"]["acoustic"]  # new day, morning
    first = state.combine_checkins([a])
    out = checkin(client, features(0.8, rng), slot="evening", same_day=True)
    row = client.get("/api/patients/1").json()["scores"][-1]
    day = json.loads(row["branch_scores_json"])
    d1, d2 = day["anchored"]["acoustic"]
    assert first == pytest.approx(d1, abs=1e-4)
    assert day["w"]["anchored"]["acoustic"] == [1.0, 0.5]
    assert row["D_anchored"] == pytest.approx((d1 + 0.5 * d2) / 1.25**0.5, abs=1e-3)
    # B20 still exact with weights: the explanation sums to S
    why = out["state"]["explanation"]
    assert sum(x["contribution"] for x in why["by_day"]) + why["older"] == pytest.approx(why["S"], abs=1e-3)
    day_parts = client.get("/api/patients/1").json()["explanation"]["by_feature"]
    assert sum(day_parts.values()) + why["older"] == pytest.approx(why["S"], abs=2e-3)


def test_old_slot_baseline_still_scores(client, monkeypatch):
    """Baselines locked before B30 have morning/evening rows and no slope: they fall back to 'all'."""
    from app.config import settings

    rng = random.Random(9)
    monkeypatch.setattr(settings, "time_of_day", "slots")
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    for _ in range(6):
        checkin(client, features(0, rng))
    assert "morning" in client.get("/api/patients/1/baseline").json()["stats"]
    monkeypatch.setattr(settings, "time_of_day", "hour")
    out = checkin(client, features(0, rng), slot="evening")
    assert out["state"]["S_anchored"] is not None and abs(out["state"]["S_anchored"]) < 3


def test_hour_slope_is_shrunk_and_removed():
    vals = [10 + 0.5 * (h - 15) for h in range(9, 22)]  # rises 0.5 per hour
    slope, h0, n = state.hour_slope(vals, list(range(9, 22)), k=5)
    assert h0 == 15 and n == 13 and 0 < slope < 0.5  # shrunk toward 0 by k pseudo-values
    assert state.hour_slope([1.0, 2.0, 3.0], [12.0, 12.0, 12.0], k=5)[0] == 0  # one hour only → no slope
    assert state.hour_slope([1.0, 2.0], [None, 9.0], k=5) is None
    adj = state.adjust_for_hour({"energy": 1.0, "jitter": 2.0}, 18.0, {"energy": (0.1, 15.0)})
    assert adj == {"energy": pytest.approx(0.7), "jitter": 2.0}
    assert state.adjust_for_hour({"energy": 1.0}, None, {"energy": (0.1, 15.0)}) == {"energy": 1.0}


def test_checkin_weights_b31():
    assert state.checkin_weight(prompted=False, earlier_today_in_branch=0) == 1.0
    assert state.checkin_weight(prompted=False, earlier_today_in_branch=2) == 0.5
    assert state.checkin_weight(prompted=True, earlier_today_in_branch=1) == 1.0
    assert state.combine_checkins([1.0, 1.0]) == pytest.approx(2**0.5)  # B16 unchanged with weights 1
    assert state.combine_checkins([1.0, 1.0], [1.0, 0.5]) == pytest.approx(1.5 / 1.25**0.5)


def test_feature_with_too_few_baseline_values_gets_no_stats(client):
    rng = random.Random(6)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    checkin(client, text_features(False, rng), kind="text")  # only ONE written entry in the baseline
    for _ in range(5):
        out = checkin(client, features(0, rng))
    assert out["baseline"]["status"] == "active"
    stats = client.get("/api/patients/1/baseline").json()["stats"]["all"]
    assert "speech_rate" in stats and "text_logit" not in stats
    out = checkin(client, text_features(True, rng), kind="text")
    assert out["state"]["S"] is None and "text" not in out["state"]["branches"]  # no guessed scale


def test_rebaseline_shows_building_progress_and_freezes_old_count(client):
    rng = random.Random(7)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    for _ in range(5):  # locks at the 3rd, two more afterwards
        out = checkin(client, features(0, rng))
    assert sum(out["baseline"]["checkins"].values()) == 3  # baseline period only, frozen at lock
    assert out["baseline_building"] is None
    client.post("/api/patients/1/baseline/start", json={"period_days": 14, "min_checkins": 3, "note": "new phone"})
    out = checkin(client, features(0, rng))
    assert out["baseline"]["status"] == "active" and out["baseline"]["version"] == 1
    assert out["baseline_building"]["status"] == "building" and out["baseline_building"]["version"] == 2


def test_mixed_baseline_without_enough_values_per_feature_keeps_building(client):
    rng = random.Random(8)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    for _ in range(3):  # 6 check-ins in total, but only 3 values per feature (< min_feature_n = 5)
        checkin(client, features(0, rng))
        out = checkin(client, text_features(False, rng), same_day=True, kind="text")
    assert out["baseline"]["status"] == "building"  # not locked into a baseline that cannot score
    for _ in range(2):  # two more spoken days → acoustic features reach 5 → locks
        out = checkin(client, features(0, rng))
    assert out["baseline"]["status"] == "active"
    stats = client.get("/api/patients/1/baseline").json()["stats"]["all"]
    assert "speech_rate" in stats and "text_logit" not in stats


def test_escalation_uses_the_shared_clock(client):
    """Escalations are dated by baseline.clock() like check-ins, not by the wall clock (seed_demo, tests)."""
    from sqlmodel import Session, select

    from app.models import Escalation

    rng = random.Random(9)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    for _ in range(6):
        checkin(client, features(0, rng))
    for _ in range(4):
        out = checkin(client, features(2.5, rng))
    assert out["state"]["level"] >= 2
    with Session(db.engine) as s:
        esc = s.exec(select(Escalation)).all()
    assert esc and all(e.ts.date() == baseline_mod.clock().date() or e.ts.date() < baseline_mod.clock().date() for e in esc)
    assert max(e.ts for e in esc).date() > datetime.now(timezone.utc).date()  # clock was advanced beyond today
