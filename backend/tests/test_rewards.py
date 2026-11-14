"""Rewards and habits (#24): reward doing, never the score; gentle streaks; nothing earned is lost."""

from datetime import date, timedelta

import pytest
from sqlmodel import Session, select

from app import db, rewards
from app.models import Arm, InterventionLog

from .engagement_utils import ENTRIES, checkin_on_days, client, diary  # noqa: F401

D0 = date(2026, 11, 1)


def days(*offsets):
    return {D0 + timedelta(days=o) for o in offsets}


# ---------------------------------------------------------------- streak rules (pure)

def test_consecutive_days_make_a_streak():
    assert rewards.streak_at(days(0, 1, 2, 3, 4), D0 + timedelta(days=4)) == 5


def test_one_rest_day_keeps_the_streak():
    assert rewards.streak_at(days(0, 1, 2, 4, 5), D0 + timedelta(days=5)) == 5


def test_today_not_yet_checked_in_does_not_break_it():
    assert rewards.current_streak(days(0, 1, 2), D0 + timedelta(days=3)) == 3


def test_two_missed_days_in_a_row_end_the_streak():
    assert rewards.streak_at(days(0, 1, 2, 5, 6), D0 + timedelta(days=6)) == 2


def test_only_one_rest_day_a_week():
    # rest days on day 3 and day 6 are 3 days apart -> the second one ends the streak
    assert rewards.streak_at(days(0, 1, 2, 4, 5, 7), D0 + timedelta(days=7)) == 3
    # a week apart is fine
    assert rewards.streak_at(days(0, 1, 2, 4, 5, 6, 7, 8, 9, 11), D0 + timedelta(days=11)) == 10


def test_longest_streak_is_never_lost():
    hist = days(0, 1, 2, 3, 4, 5, 10)
    assert rewards.current_streak(hist, D0 + timedelta(days=10)) == 1
    assert rewards.longest_streak(hist) == 6


@pytest.mark.parametrize("spoken, speech_s, n_words, ok", [
    (True, 12.0, 0, True),
    (True, 6.0, 40, False),  # too little speech, however many words
    (True, None, 0, True),  # not measured (before #21/#22) -> benefit of the doubt
    (False, None, 25, True),
    (False, None, 8, False),
])
def test_memo_needs_enough_speech_or_words(spoken, speech_s, n_words, ok):
    assert rewards.memo_counts(spoken, speech_s, n_words) is ok


def test_level_hint():
    assert rewards.level_hint(0)["level"] == 1
    assert rewards.level_hint(49)["level"] == 1
    assert rewards.level_hint(50) == {"level": 2, "xp": 50, "next_level_xp": 150}
    assert rewards.level_hint(10_000)["next_level_xp"] is None


# ---------------------------------------------------------------- earning through the API

def get(c, pid=1):
    r = c.get(f"/api/patients/{pid}/rewards")
    assert r.status_code == 200, r.text
    return r.json()


def test_contract_shape(client):
    out = get(client)
    assert set(out) == {"balance", "xp", "level_hint", "streak", "longest_streak", "checked_in_today",
                        "days_since_last_checkin", "ledger"}
    assert out["balance"] == 0 and out["checked_in_today"] is False and out["days_since_last_checkin"] is None


def test_daily_memo_pays_once_a_day(client):
    diary(client)
    diary(client, ENTRIES["good_2"])  # second check-in the same day
    out = get(client)
    assert out["balance"] == 10 and [x["reason"] for x in out["ledger"]] == ["memo"]
    assert out["checked_in_today"] is True and out["streak"] == 1


def test_a_bad_day_earns_the_same_as_a_good_day(client):
    diary(client, ENTRIES["good_1"])
    good = get(client)["balance"]
    client.day = 1
    diary(client, ENTRIES["low_1"])
    assert get(client)["balance"] - good == good == 10


def test_too_short_written_checkin_earns_nothing(client):
    diary(client, "Long day, tired but okay.")
    assert get(client)["balance"] == 0


def test_seven_day_streak_bonus_and_sync_is_idempotent(client):
    checkin_on_days(client, range(7))
    out = get(client)
    assert out["streak"] == 7 and out["balance"] == 7 * 10 + 20
    assert sum(1 for x in out["ledger"] if x["reason"] == "streak_7") == 1
    assert get(client)["balance"] == 90  # reading again pays nothing twice


def test_streak_survives_a_rest_day_and_longest_is_kept(client):
    checkin_on_days(client, [0, 1, 2, 4, 5])
    assert get(client)["streak"] == 5
    checkin_on_days(client, [9])
    out = get(client)
    assert out["streak"] == 1 and out["longest_streak"] == 5
    assert out["balance"] == 60  # everything earned is still there


def test_action_done_feeds_the_bandit_and_earns(client):
    diary(client)  # the check-in picks today's action
    get(client)
    with Session(db.engine) as s:
        log = s.exec(select(InterventionLog).where(InterventionLog.patient_id == 1)).one()
        arm_before = s.exec(select(Arm).where(Arm.patient_id == 1, Arm.key == log.arm_key)).one().a
    r = client.post("/api/patients/1/habits/action", json={"done": True, "rating": 5})
    assert r.status_code == 200, r.text
    assert {x["reason"] for x in r.json()["earned"]} == {"action_done", "followup"}
    with Session(db.engine) as s:
        log = s.exec(select(InterventionLog).where(InterventionLog.patient_id == 1)).one()
        arm = s.exec(select(Arm).where(Arm.patient_id == 1, Arm.key == log.arm_key)).one()
    assert log.completed is True and arm.a > arm_before  # the bandit learned from it
    assert get(client)["balance"] == 10 + 10 + 3


def test_not_today_is_honest_and_still_earns_the_followup(client):
    diary(client)
    get(client)
    r = client.post("/api/patients/1/habits/action", json={"done": False})
    assert [x["reason"] for x in r.json()["earned"]] == ["followup"]
    # changing the answer to "done" later the same day adds the action reward, nothing is paid twice
    r = client.post("/api/patients/1/habits/action", json={"done": True, "rating": 3})
    assert [x["reason"] for x in r.json()["earned"]] == ["action_done"]
    assert get(client)["balance"] == 10 + 3 + 10


def test_spend_never_lowers_xp_and_refuses_overdraft(client):
    diary(client)
    with pytest.raises(rewards.InsufficientBalance):
        rewards.spend(1, 50, "hat")
    rewards.spend(1, 6, "scarf")
    out = get(client)
    assert out["balance"] == 4 and out["xp"] == 10
    assert out["ledger"][0]["reason"] == "spend:scarf" and out["ledger"][0]["amount"] == -6


def test_week_view(client):
    checkin_on_days(client, [0, 2])
    client.post("/api/patients/1/habits/action", json={"done": True})
    out = client.get("/api/patients/1/habits/week").json()
    assert len(out["days"]) == 7 and out["days"][-1] == client.now().date().isoformat()
    memo = next(h for h in out["habits"] if h["key"] == "memo")["done"]
    action = next(h for h in out["habits"] if h["key"] == "action")["done"]
    assert memo[-3:] == [True, False, True] and action[-1] is True and action[-3] is False
    assert out["earned"][-1] == 10 + 10 + 3
    assert out["today_action"]["answered"] is True and out["today_action"]["done"] is True
    assert out["today_action"]["label"]


def test_week_view_before_any_checkin_has_no_action(client):
    assert client.get("/api/patients/1/habits/week").json()["today_action"] is None


def test_unknown_patient(client):
    assert client.get("/api/patients/99/rewards").status_code == 404


def test_on_time_bonus_with_the_daily_prompt(client):
    """+5 for a memo within 2 h of the day's prompt (#22). A demo resend later never takes it back."""
    pytest.importorskip("app.prompt")
    assert client.post("/api/patients/1/prompt/now").status_code == 200
    client.minutes = 30
    diary(client)
    out = get(client)
    assert {x["reason"] for x in out["ledger"]} == {"memo", "memo_on_time"} and out["balance"] == 15
    client.post("/api/patients/1/prompt/now")  # resets today's prompt, so it no longer records the answer
    assert get(client)["balance"] == 15
