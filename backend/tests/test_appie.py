"""Appie facade shares the canonical wallet, prompts and memo history, including legacy upgrades."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

from app import appie, baseline, db, rewards
from app.main import app
from app.models import CheckIn, Feature, FeatureSource, Patient, Slot
from app.models_appie import AppieHabitLog, AppiePet
from app.models_rewards import RewardLedger
from app.pipeline.phrase import PHRASE
from app.seed import seed

NOW = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    monkeypatch.setattr(baseline, "clock", lambda: NOW)
    appie.migrate_legacy(db.engine)
    return TestClient(app)


def fund(amount):
    with Session(db.engine) as session:
        session.add(RewardLedger(patient_id=1, ts=NOW, date=NOW.date(), reason="test:grant", amount=amount, key="test:grant"))
        session.commit()


def get(client):
    response = client.get("/api/patients/1/appie")
    assert response.status_code == 200, response.text
    return response.json()


def add(client, label):
    response = client.post("/api/patients/1/appie/habits", json={"label": label})
    assert response.status_code == 200, response.text
    return response.json()["key"]


def test_facade_uses_the_canonical_wallet_and_phrase(client):
    fund(55)
    avatar = get(client)
    canonical = client.get("/api/patients/1/rewards").json()
    assert avatar["food"] == canonical["balance"] == 55
    assert avatar["xp"] == canonical["xp"] == 55
    assert avatar["level"] == 2 and avatar["xp_level_start"] == 50
    assert avatar["daily_phrase"] == PHRASE
    assert len(avatar["week"]) == 7
    assert {h["key"] for h in avatar["habits"]} == {"memo", "action", "followup", "sleep", "steps", "active"}
    assert client.get("/api/patients/99999/appie").status_code == 404


def test_feed_spends_food_once_and_never_decreases_lifetime_xp(client):
    fund(55)
    body = {"amount": 10, "request_id": "same-click"}
    first = client.post("/api/patients/1/appie/feed", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["food"] == 45 and first.json()["xp"] == 55 and first.json()["fed_xp"] == 10
    assert client.post("/api/patients/1/appie/feed", json=body).json() == first.json()
    second = client.post("/api/patients/1/appie/feed", json={"amount": 500, "request_id": "second-click"})
    assert second.json()["fed"] == 45 and second.json()["food"] == 0 and second.json()["xp"] == 55
    assert client.post("/api/patients/1/appie/feed", json={"amount": 1}).status_code == 400
    assert client.get("/api/patients/1/rewards").json()["balance"] == 0


def test_concurrent_feeds_cannot_spend_the_same_food_twice(client):
    fund(5)
    def feed(index):
        return client.post("/api/patients/1/appie/feed", json={"amount": 5, "request_id": f"click-{index}"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(feed, range(4)))
    assert sorted(response.status_code for response in responses) == [200, 400, 400, 400]
    assert get(client)["food"] == 0 and get(client)["xp"] == 5


def test_concurrent_retries_of_one_feed_return_one_spend(client):
    fund(20)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post("/api/patients/1/appie/feed", json={"amount": 5, "request_id": "retry"}), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json() == responses[0].json() for response in responses)
    assert get(client)["food"] == 15


def test_custom_habits_are_capped_and_duplicates_rename_delete_preserve_food(client):
    keys = [add(client, label) for label in ("Reading", "Guitar", "Gardening", "Sketching")]
    assert add(client, "  reading  ") == keys[0]
    results = [client.post(f"/api/patients/1/appie/habits/{key}/tick").json() for key in keys]
    assert [result["reward"]["amount"] if result["reward"] else 0 for result in results] == [2, 2, 2, 0]
    assert client.post(f"/api/patients/1/appie/habits/{keys[0]}/tick").json()["reward"] is None
    assert client.put(f"/api/patients/1/appie/habits/{keys[0]}", json={"label": "Guitar"}).status_code == 409
    assert client.put(f"/api/patients/1/appie/habits/{keys[0]}", json={"label": "A few pages"}).status_code == 200
    assert client.delete(f"/api/patients/1/appie/habits/{keys[0]}").status_code == 200
    assert get(client)["food"] == 6
    assert client.post(f"/api/patients/1/appie/habits/{keys[0]}/tick").status_code == 404
    assert client.post("/api/patients/1/appie/habits", json={"label": "  "}).status_code == 422


def test_concurrent_custom_ticks_are_idempotent(client):
    key = add(client, "Play piano")
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post(f"/api/patients/1/appie/habits/{key}/tick"), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert sum(response.json()["reward"] is not None for response in responses) == 1
    assert get(client)["food"] == 2


def test_wearable_hook_uses_canonical_keys_and_does_not_create_memo_presence(client):
    with Session(db.engine) as session:
        first = appie.on_wearable(session, 1, NOW, {"sleep_hours": 7.5, "steps": 8000, "active_minutes": 25})
        session.commit()
        assert {reward["reason"] for reward in first} == {"habit_sleep", "habit_steps", "habit_active"}
        assert appie.on_wearable(session, 1, NOW, {"sleep_hours": 8, "steps": 10000, "active_minutes": 30}) == []
        session.commit()
    avatar = get(client)
    assert avatar["food"] == avatar["xp"] == 15
    assert not avatar["streak"]["today"] and avatar["streak"]["current"] == 0
    assert all(h["done"] for h in avatar["habits"] if h["key"] in ("sleep", "steps", "active"))


def test_wearable_and_phrase_only_history_do_not_count_but_featureless_legacy_memo_does(client):
    with Session(db.engine) as session:
        for index, feature in enumerate((("steps", FeatureSource.wearable), ("phrase_energy", FeatureSource.prosody))):
            checkin = CheckIn(patient_id=1, ts=NOW - timedelta(days=index), slot=Slot.morning)
            session.add(checkin)
            session.flush()
            session.add(Feature(checkin_id=checkin.id, name=feature[0], value=1, source=feature[1]))
        session.commit()
        assert rewards.checkin_history(session, 1, timezone.utc) == []
        legacy = CheckIn(patient_id=1, ts=NOW, slot=Slot.morning, transcript="A short legacy diary entry is still an actual check in, with enough words here to earn its ordinary reward today.")
        session.add(legacy)
        session.commit()
        assert len(rewards.checkin_history(session, 1, timezone.utc)) == 1
    assert get(client)["streak"]["today"]


def test_prompt_off_is_nullable_and_avatar_name_does_not_change_social_consent(client):
    from app.models_prompt import PromptSettings
    from app.models_social import Profile
    with Session(db.engine) as session:
        session.add(PromptSettings(patient_id=1, enabled=False))
        session.commit()
    out = get(client)
    assert out["prompt"]["state"] == "off" and out["prompt"]["at"] is None
    assert not out["prompt"]["bonus_available"]
    assert client.put("/api/patients/1/appie/profile", json={"avatar_name": "Little Appie"}).json()["avatar_name"] == "Little Appie"
    with Session(db.engine) as session:
        assert session.get(Profile, 1) is None


def test_legacy_wallet_habits_and_multiple_feeds_survive_without_destroying_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "clock", lambda: NOW)
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE companion (patient_id INTEGER, name TEXT, food INTEGER, xp INTEGER)")
        connection.exec_driver_sql("CREATE TABLE habit (patient_id INTEGER, key TEXT, label TEXT, active BOOLEAN)")
        connection.exec_driver_sql("CREATE TABLE habitlog (patient_id INTEGER, date TEXT, habit_key TEXT, source TEXT)")
        connection.exec_driver_sql("CREATE UNIQUE INDEX legacy_reason ON rewardledger(patient_id,date,reason)")
        connection.exec_driver_sql("INSERT INTO patient (id,name,created_at,risk_level) VALUES (1,'Mia','2026-11-01 12:00:00',0)")
        connection.exec_driver_sql("INSERT INTO companion VALUES (1,'Appie Jr',12,8)")
        connection.exec_driver_sql("INSERT INTO habit VALUES (1,'c_read','Read a book',1)")
        connection.exec_driver_sql("INSERT INTO habitlog VALUES (1,'2026-11-14','c_read','manual')")
    with Session(engine) as session:
        session.add_all([RewardLedger(patient_id=1, ts=NOW, date=NOW.date(), reason="memo_text", amount=6),
                         RewardLedger(patient_id=1, ts=NOW, date=NOW.date(), reason="memo_voice", amount=10),
                         RewardLedger(patient_id=1, ts=NOW, date=NOW.date(), reason="custom:c_read", amount=4)])
        session.commit()
    appie.migrate_legacy(engine)
    appie.migrate_legacy(engine)
    with Session(engine) as session:
        assert rewards.balance(session, 1) == 12 and rewards.xp(session, 1) == 20
        assert session.get(AppiePet, 1).fed_xp == 8
        assert session.get(AppiePet, 1).name == "Appie Jr"
        assert len(session.exec(select(AppieHabitLog)).all()) == 1
        assert session.exec(select(RewardLedger).where(RewardLedger.key == f"memo:{NOW.date()}")).one()
        assert session.execute(text("SELECT COUNT(*) FROM companion")).scalar() == 1
        first = appie.feed(session, 1, 3, "one")
        second = appie.feed(session, 1, 3, "two")
        assert first["food"] == 9 and second["food"] == 6 and second["xp"] == 20
        assert appie.tick_habit(session, 1, "c_read")["reward"] is None
    engine.dispose()
