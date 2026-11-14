"""The circle (#26): own consent, opt-in sharing, preset reactions, crisis path, and NO health data."""

import json

from sqlmodel import Session, select

from app import db
from app.engine import state
from app.models import Escalation, Patient
from app.models_appie import AppiePet
from app.models_rewards import RewardLedger

from .engagement_utils import ENTRIES, checkin_on_days, client, diary  # noqa: F401


def test_appie_name_and_unlocked_look_respect_individual_sharing(client):
    now = client.now()
    with Session(db.engine) as session:
        session.add(AppiePet(patient_id=1, name="Captain Appie"))
        session.add(RewardLedger(patient_id=1, ts=now, date=now.date(), reason="test", amount=550, key="test:avatar"))
        session.commit()
    assert client.put("/api/patients/1/profile", json={"social_consent": True, "share_avatar": True, "share_level": True}).status_code == 200
    circle = client.post("/api/patients/1/circle", json={"name": "Test"}).json()
    feed = client.get("/api/circles/feed", params={"token": circle["token"]}).json()
    card = next(member for member in feed["members"] if member["is_you"])
    assert card["avatar"]["name"] == "Captain Appie" and card["avatar"]["tier"] == 2
    assert card["level"] == 4
    assert "xp" not in card["avatar"] and "food" not in card["avatar"]
    assert client.put("/api/patients/1/profile", json={"share_avatar": False, "share_level": False}).status_code == 200
    hidden = client.get("/api/circles/feed", params={"token": circle["token"]}).json()["members"][0]
    assert hidden["avatar"] is None and hidden["level"] is None

# Anything health-related that must never appear in a circle response (as a key or as a value)
HEALTH_KEYS = {
    "S", "D", "S_anchored", "S_rolling", "D_anchored", "D_rolling", "score", "scores", "state", "drift",
    "risk_level", "elevated", "driver", "explanation", "top_features", "branches", "branch_scores_json",
    "evidence_json", "features", "feature", "transcript", "diary", "text", "llm_json", "risk_flag", "crisis",
    "speech_s", "p_depressive", "baseline", "intervention", "arm_key", "rating", "reward", "escalation_id",
    "level_reason", "reason", "notes",
} | {f for dirs in state.DIRECTION.values() for f in dirs}


def walk(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def assert_no_health_data(resp: dict):
    blob = json.dumps(resp)
    for path, k, v in walk(resp):
        assert k not in HEALTH_KEYS, f"health field {k!r} at {path}"
    for snippet in (ENTRIES["low_1"][:40], ENTRIES["good_1"][:40]):
        assert snippet not in blob, "diary text leaked into a circle response"


def consent(c, pid=1, **share):
    r = c.put(f"/api/patients/{pid}/profile", json={"social_consent": True, **share})
    assert r.status_code == 200, r.text
    return r.json()


def circle_with_friend(c):
    consent(c)
    mine = c.post("/api/patients/1/circle", json={"name": "Close ones"}).json()
    friend = c.post("/api/circles/join", json={"code": mine["invite_code"], "name": "Jonas"}).json()
    return mine, friend


def feed(c, token):
    r = c.get("/api/circles/feed", params={"token": token})
    assert r.status_code == 200, r.text
    return r.json()


def test_circle_needs_its_own_consent(client):
    assert client.post("/api/patients/1/circle", json={}).status_code == 403
    consent(client)
    assert client.post("/api/patients/1/circle", json={}).status_code == 200


def test_invite_by_code_and_feed(client):
    mine, friend = circle_with_friend(client)
    out = feed(client, friend["token"])
    names = {m["name"] for m in out["members"]}
    assert names == {"Anonymous", "Jonas"}  # anonymous is the default; the real name is never shown
    assert "Mia" not in json.dumps(out)


def test_nothing_is_shared_until_opted_in(client):
    _, friend = circle_with_friend(client)
    checkin_on_days(client, [0])
    mia = next(m for m in feed(client, friend["token"])["members"] if m["role"] == "patient")
    assert mia["streak"] is None and mia["level"] is None and mia["checked_in_today"] is None and mia["avatar"] is None
    consent(client, share_streak=True, share_checked_in=True, share_level=True, share_avatar=True,
            mode="profile", display_name="Mia R.", emoji="🌿")
    mia = next(m for m in feed(client, friend["token"])["members"] if m["role"] == "patient")
    assert mia["streak"] == 1 and mia["checked_in_today"] is True and mia["level"] == 1
    assert mia["name"] == "Mia R." and mia["emoji"] == "🌿" and mia["avatar"]["patient_id"] == 1


def test_no_health_data_in_any_circle_response(client):
    """Fails if a score, diary, transcript or feature field ever appears in a circle response."""
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    checkin_on_days(client, [0, 1, 2], ENTRIES["good_1"])
    checkin_on_days(client, [3, 4, 5], ENTRIES["low_1"])  # now there are scores and maybe an escalation
    mine, friend = circle_with_friend(client)
    consent(client, share_streak=True, share_checked_in=True, share_level=True, share_avatar=True, emoji="🙂")
    client.post("/api/circles/react", json={"token": friend["token"], "to_member_id": mine["member_id"],
                                            "kind": "thinking_of_you"})
    client.minutes = 60 * 24  # past any nudge hold window
    for resp in (
        feed(client, friend["token"]),
        feed(client, mine["token"]),
        client.get("/api/patients/1/circle").json(),
        client.get("/api/patients/1/profile").json(),
        client.get("/api/social/presets").json(),
    ):
        assert_no_health_data(resp)


def test_preset_reactions_only(client):
    mine, friend = circle_with_friend(client)
    ok = client.post("/api/circles/react", json={"token": friend["token"], "to_member_id": mine["member_id"],
                                                 "kind": "walk_together"})
    assert ok.status_code == 200
    bad = client.post("/api/circles/react", json={"token": friend["token"], "to_member_id": mine["member_id"],
                                                  "kind": "you should try harder"})
    assert bad.status_code == 400
    r = feed(client, mine["token"])["reactions"][0]
    assert r["label"] == "Walk together?" and r["from_name"] == "Jonas"


def test_crisis_words_in_profile_go_to_the_clinician_and_are_not_saved(client):
    consent(client)
    r = client.put("/api/patients/1/profile", json={"pseudonym": "want to end my life"}).json()
    assert r["crisis"] is True and r["pseudonym"] is None
    with Session(db.engine) as s:
        esc = s.exec(select(Escalation).where(Escalation.patient_id == 1)).all()
        assert esc[-1].level == 3 and esc[-1].recipient.value == "clinician"
        assert s.get(Patient, 1).risk_level == 3


def test_friends_card_is_held_so_the_patient_can_cancel(client):
    mine, friend = circle_with_friend(client)
    with Session(db.engine) as s:
        p = s.get(Patient, 1)
        p.risk_level = 1
        s.add(p)
        s.add(Escalation(patient_id=1, level=1, recipient="friend", reason="test", ts=client.now()))
        s.commit()
    assert feed(client, friend["token"])["reach_out_cards"] == []  # held
    pending = client.get("/api/patients/1/circle").json()["nudge"]
    assert pending["pending"] is True and pending["can_cancel"] is True
    client.minutes = 31
    cards = feed(client, friend["token"])["reach_out_cards"]
    assert cards == [{"to_member_id": mine["member_id"], "message": "Maybe a good day to reach out to Anonymous?"}]
    assert client.post("/api/patients/1/circle/nudge/cancel").status_code == 200
    assert feed(client, friend["token"])["reach_out_cards"] == []


def test_leave_and_bad_tokens(client):
    _, friend = circle_with_friend(client)
    assert client.get("/api/circles/feed", params={"token": "nope"}).status_code == 404
    client.post("/api/circles/leave", json={"token": friend["token"]})
    assert client.get("/api/circles/feed", params={"token": friend["token"]}).status_code == 404
