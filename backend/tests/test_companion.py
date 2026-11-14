"""Kind avatar check-ins (#25): rules, reviewed lines, one a day, quiet hours, crisis path, clinician context."""

import re
from datetime import date, timedelta

import pytest
from sqlmodel import Session, select

from app import db
from app.engine import companion
from app.models import Escalation

from .engagement_utils import ENTRIES, checkin_on_days, client, diary  # noqa: F401

T = date(2026, 11, 20)


def ago(*offsets):
    return {T - timedelta(days=o) for o in offsets}


# ---------------------------------------------------------------- reviewed lines

BANNED = [
    r"\byou (haven'?t|have not|didn'?t|did not|forgot|failed|missed|should)\b",
    r"\bbr(oke|oken|eak)\b.*\bstreak\b", r"\bstreak\b", r"\bscores?\b", r"\brelapse", r"\bclinic", r"\bdoctor",
    r"\bnurse", r"\bdiagnos", r"\bdepress", r"\btherap", r"\bsymptom", r"\bmood\b", r"\bdays? (ago|in a row)\b",
    r"\blost\b", r"\blose\b", r"\bdisappoint", r"\bworried\b", r"\bsad\b", r"\bsick\b", r"\blonely\b",
]


def all_lines():
    return [(k, line) for k, lines in companion.LINES.items() for line in lines]


@pytest.mark.parametrize("kind, line", all_lines(), ids=lambda x: x["id"] if isinstance(x, dict) else x)
def test_lines_are_warm_and_never_guilt(kind, line):
    text = line["text"].lower().replace("’", "'")
    for pat in BANNED:
        assert not re.search(pat, text), f"{line['id']}: {pat}"
    assert len(line["text"]) <= 120


def test_every_trigger_has_lines_and_unique_ids():
    assert set(companion.KINDS) <= set(companion.LINES)
    ids = [line["id"] for _, line in all_lines()]
    assert len(ids) == len(set(ids))
    assert all(line.get("offer_written") for line in companion.LINES["missed_few"])


# ---------------------------------------------------------------- rules (pure)

def test_streak_then_one_missed_day():
    assert companion.decide(ago(2, 3, 4, 5, 6), {}, T) == "missed_one_after_streak"
    assert companion.decide(ago(2, 3, 4), {}, T) is None  # no streak of 5: let the day be


@pytest.mark.parametrize("missed, kind", [(2, "missed_few"), (3, "missed_few"), (4, None), (9, None)])
def test_two_to_three_missed_days(missed, kind):
    assert companion.decide(ago(missed + 1, missed + 2), {}, T) == kind


def test_back_after_a_break():
    assert companion.decide(ago(0, 3, 4), {}, T) == "back_after_break"
    assert companion.decide(ago(0, 2, 3), {}, T) is None  # one missed day is no break
    assert companion.decide(ago(0), {}, T) is None  # a first check-in ever is not a comeback


def test_memos_much_shorter_for_three_days():
    hist = ago(*range(1, 12))
    lengths = {d: 40.0 for d in hist}
    for o in (1, 2, 3):
        lengths[T - timedelta(days=o)] = 12.0
    assert companion.decide(hist, lengths, T) == "short_memos"
    lengths[T - timedelta(days=2)] = 35.0  # only two short days
    assert companion.decide(hist, lengths, T) is None
    assert companion.decide(hist, {T - timedelta(days=o): 5.0 for o in (1, 2, 3)}, T) is None  # no usual yet


def test_nothing_on_an_ordinary_day():
    assert companion.decide(ago(1, 2, 3), {}, T) is None
    assert companion.decide(set(), {}, T) is None


# ---------------------------------------------------------------- through the API

@pytest.fixture(autouse=True)
def prompt_time_has_come(monkeypatch):
    """The daily prompt (#22) fires at a random time; these tests pin it to "already fired".
    The timing itself is tested below with the real prompt module when it is merged."""
    monkeypatch.setattr(companion, "prompt_time_reached", lambda s, pid, now: True)


def get(c, pid=1):
    r = c.get(f"/api/patients/{pid}/companion")
    assert r.status_code == 200, r.text
    return r.json()


def test_contract_and_one_message_a_day(client):
    checkin_on_days(client, [0, 1])
    client.day = 4  # 2 missed days
    out = get(client)
    assert out["kind"] == "missed_few" and out["offer_written"] is True
    assert {"message", "kind", "at"} <= set(out)
    first = out["message"]
    client.minutes = 120
    assert get(client)["message"] == first  # the same one, not a second
    client.day, client.minutes = 5, 0
    assert get(client)["kind"] == "missed_few"  # a new day may bring one


def test_respects_quiet_hours_and_pause(client, monkeypatch):
    checkin_on_days(client, [0, 1])
    client.day = 4
    monkeypatch.setattr(companion, "is_quiet", lambda s, pid, now: True)
    assert get(client)["message"] is None


def test_waits_for_the_prompt_time(client, monkeypatch):
    checkin_on_days(client, [0, 1])
    client.day = 4
    monkeypatch.setattr(companion, "prompt_time_reached", lambda s, pid, now: False)
    assert get(client)["message"] is None
    assert client.post("/api/patients/1/companion/demo").json()["kind"] == "missed_few"  # demo skips the wait


def test_glad_to_see_you_after_a_break(client):
    checkin_on_days(client, [0, 1, 4])
    out = get(client)
    assert out["kind"] == "back_after_break" and out["glad_to_see_you"] is True


def test_reply_with_crisis_words_runs_the_crisis_path(client):
    checkin_on_days(client, [0, 1])
    client.day = 4
    msg = get(client)
    r = client.post("/api/patients/1/companion/reply",
                    json={"message_id": msg["id"], "text": "honestly I don't want to be alive anymore"}).json()
    assert r["crisis"] is True
    with Session(db.engine) as s:
        esc = s.exec(select(Escalation).where(Escalation.patient_id == 1)).all()
        assert esc[-1].level == 3 and esc[-1].recipient.value == "clinician"


def test_plain_reply_is_not_crisis(client):
    checkin_on_days(client, [0, 1])
    client.day = 4
    msg = get(client)
    r = client.post("/api/patients/1/companion/reply", json={"message_id": msg["id"], "text": "busy week, all good"})
    assert r.json()["crisis"] is False


def test_clinician_context_after_three_days_without_answer(client):
    checkin_on_days(client, [0, 1])
    for d in (4, 5):  # missed 2, 3 -> two check-ins from the avatar
        client.day = d
        get(client)
    client.day = 6
    get(client)  # missed 4: the avatar is quiet now
    ctx = client.get("/api/patients/1/companion/context").json()
    assert ctx["avatar_checked_in"] is True and ctx["unanswered_messages"] == 2
    assert ctx["note"] is None and ctx["alert"] is False  # only 2 days so far
    client.day = 7
    ctx = client.get("/api/patients/1/companion/context").json()
    assert ctx["note"] == "Avatar checked in, no answer for 3 days" and ctx["alert"] is False
    with Session(db.engine) as s:  # context, never an escalation
        assert s.exec(select(Escalation).where(Escalation.patient_id == 1)).all() == []
    diary(client)  # an answer clears it
    assert client.get("/api/patients/1/companion/context").json()["avatar_checked_in"] is False


# ---------------------------------------------------------------- with the daily prompt (#22), once merged

def _real_timing(monkeypatch):
    pytest.importorskip("app.prompt")
    monkeypatch.setattr(companion, "prompt_time_reached", _REAL_PROMPT_TIME_REACHED)


_REAL_PROMPT_TIME_REACHED = companion.prompt_time_reached


def test_speaks_once_the_days_prompt_has_fired(client, monkeypatch):
    _real_timing(monkeypatch)
    checkin_on_days(client, [0, 1])
    client.day = 4
    assert client.post("/api/patients/1/prompt/now").status_code == 200
    assert get(client)["kind"] == "missed_few"


def test_quiet_when_paused_today(client, monkeypatch):
    _real_timing(monkeypatch)
    checkin_on_days(client, [0, 1])
    client.day = 4
    r = client.put("/api/patients/1/prompt/settings", json={"pause_today": True})
    assert r.status_code == 200, r.text
    assert get(client)["message"] is None
