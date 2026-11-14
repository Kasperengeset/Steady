"""Patient-app and clinician-portal endpoints (routers/views.py, app/insight.py) and action targeting."""

import random
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text as sql
from sqlmodel import SQLModel

from app import baseline as baseline_mod
from app import db
from app.engine import signals
from app.main import app
from app.seed import seed
from .test_health import healthy_day, low_day


def test_fitting_arms_follow_the_main_signal():
    keys = ["walk_outside", "call_friend", "morning_light", "music", "breathing", "small_task"]
    steps_down = [{"branch": "health", "feature": "steps", "contribution": 1.2}]
    assert set(signals.fitting_arms(steps_down, keys)) == {"walk_outside", "small_task"}
    assert signals.reason_for(steps_down) == "Your steps have been lower than usual."
    voice = [{"branch": "acoustic", "feature": "arousal", "contribution": 0.9}]
    assert "call_friend" in signals.fitting_arms(voice, keys)
    assert signals.fitting_arms([], keys) == keys  # nothing off → any action
    assert signals.reason_for(None) == signals.NEUTRAL_REASON


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.day = 0
    start = datetime.now(timezone.utc) - timedelta(days=12)
    monkeypatch.setattr(baseline_mod, "clock", lambda: start + timedelta(days=c.day))
    return c


def day(c, wear, text=None):
    assert c.post("/api/checkins/wearable", json={"patient_id": 1, **wear}).status_code == 200
    if text:
        assert c.post("/api/checkins/text", json={"patient_id": 1, "slot": "evening", "text": text}).status_code == 200
    c.day += 1


GOOD = "Slept well and went for a walk by the water. Had coffee with Sam and we laughed a lot. Cooked dinner and read a few chapters before bed."
LOW = "Did not get out of bed until late. I cancelled plans again, I just could not face it. Everything feels heavy and nothing seems worth the effort."


def test_patient_and_clinician_views_follow_a_decline(client):
    # 14 check-ins = 7 wearable + 7 diaries, so every text feature has enough baseline values (B17)
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 14})
    rng = random.Random(3)
    today = client.get("/api/patients/1/today").json()
    assert today["status"] == "building" and today["action"] is None
    for _ in range(7):
        day(client, healthy_day(rng), GOOD)
    for _ in range(4):
        day(client, low_day(rng), LOW)

    t = client.get("/api/patients/1/today").json()
    assert t["status"] in ("different", "care") and t["series"]
    assert t["action"]["reason"] != signals.NEUTRAL_REASON
    assert t["action"]["arm_key"] in signals.fitting_arms(
        [{"feature": f, "contribution": 1} for f in signals.REASON if signals.REASON[f] == t["action"]["reason"]][:1],
        [a["key"] for a in client.get("/api/patients/1/insights").json()])
    assert t["last_sync"]

    fb = client.post("/api/interventions/feedback", json={"patient_id": 1, "date": t["action"]["date"], "completed": True, "rating": 5})
    assert fb.status_code == 200
    arms = client.get("/api/patients/1/insights").json()
    tried = next(a for a in arms if a["key"] == t["action"]["arm_key"])
    assert sum(a["done"] for a in arms) == 1 and tried["helped"] == 1 and tried["mean_rating"] == 5
    assert tried["history"][-1]["rating"] == 5

    tl = client.get("/api/patients/1/timeline?days=14").json()
    assert len(tl["days"]) == 14 and tl["feelings"] and tl["feelings"][0]["tone"] == "Quieter"

    e = client.get("/api/patients/1/explain?days=7").json()
    assert e["available"] and {s["source"] for s in e["sources"]} == {"health", "acoustic", "text"}
    health = next(s for s in e["sources"] if s["source"] == "health")
    assert health["direction"] == "worse" and any(r["status"] == "worse" for r in health["signals"])
    assert e["words"]["quote"] and e["habits"]["per_week_now"] > 0

    n = client.get("/api/clinician/notifications?status=open").json()
    esc = [x for x in n if x["kind"] in ("friends", "clinic")]
    assert esc and esc[0]["patient"] == "Mia"
    assert client.post(f"/api/clinician/notifications/{esc[0]['id']}/handled").status_code == 200
    assert esc[0]["id"] in {x["id"] for x in client.get("/api/clinician/notifications?status=handled").json()}
    assert client.post("/api/clinician/notifications/baseline-1/handled").status_code == 404


def test_missing_columns_are_added_to_an_old_database():
    with db.engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS escalation")
        conn.exec_driver_sql("CREATE TABLE escalation (id INTEGER PRIMARY KEY, patient_id INTEGER, ts DATETIME, "
                             "level INTEGER, recipient VARCHAR, reason VARCHAR)")
    db.init_db()
    with db.engine.begin() as conn:
        cols = {r[1] for r in conn.execute(sql("PRAGMA table_info(escalation)"))}
    assert {"handled", "handled_at"} <= cols
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
