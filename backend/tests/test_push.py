"""Web Push for the daily prompt (#22, P2): sent once per prompt, only inside the on-time window,
never in quiet hours/pause/off, no health data in the payload, dead devices removed. No network:
push._webpush is replaced."""

import random
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from app import baseline as baseline_mod
from app import db, prompt, push
from app.main import app
from app.models_push import PushSubscription
from app.seed import seed

SUB = {"endpoint": "https://push.example.test/abc", "keys": {"p256dh": "pk", "auth": "au"}}


class Gone(Exception):
    def __init__(self, status):
        super().__init__(f"push service said {status}")
        self.response = type("R", (), {"status_code": status})()


@pytest.fixture()
def pc(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.now = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)  # 10:00 in Helsinki
    monkeypatch.setattr(baseline_mod, "clock", lambda: c.now)
    monkeypatch.setattr(prompt, "RNG", random.Random(3))
    for k, v in (("vapid_public_key", "PUB"), ("vapid_private_key", "PRIV"), ("vapid_subject", "mailto:team@example.test")):
        monkeypatch.setattr(push.settings, k, v)
    c.sent = []
    c.fail = None
    def fake(sub, payload):
        if c.fail:
            raise c.fail
        c.sent.append((sub.endpoint, payload))
    monkeypatch.setattr(push, "_webpush", fake)
    return c


def deliver(c) -> int:
    with Session(db.engine) as s:
        n = push.deliver_due(s, c.now)
        s.commit()
    return n


def devices() -> int:
    with Session(db.engine) as s:
        return len(s.exec(select(PushSubscription)).all())


def test_off_without_vapid_keys(pc, monkeypatch):
    monkeypatch.setattr(push.settings, "vapid_private_key", "")
    st = pc.get("/api/patients/1/push").json()
    assert st["enabled"] is False
    assert pc.post("/api/patients/1/push/subscribe", json=SUB).status_code == 409
    assert pc.post("/api/patients/1/prompt/now").status_code == 200  # the demo still works in-app
    assert pc.sent == []


def test_subscribe_is_idempotent_and_needs_https(pc):
    assert pc.post("/api/patients/1/push/subscribe", json=SUB).json()["devices"] == 1
    assert pc.post("/api/patients/1/push/subscribe", json=SUB).json()["devices"] == 1
    bad = {**SUB, "endpoint": "http://insecure.test/x"}
    assert pc.post("/api/patients/1/push/subscribe", json=bad).status_code == 400
    assert pc.post("/api/patients/1/push/unsubscribe", json={"endpoint": SUB["endpoint"]}).json()["devices"] == 0


def test_demo_prompt_pushes_once_with_no_health_data(pc):
    pc.post("/api/patients/1/push/subscribe", json=SUB)
    pc.post("/api/patients/1/prompt/now")
    assert len(pc.sent) == 1
    payload = pc.sent[0][1]
    assert payload == push.PROMPT_PAYLOAD  # fixed text only: it shows on a lock screen
    assert deliver(pc) == 0 and len(pc.sent) == 1  # the loop does not send it again


def test_random_prompt_is_pushed_when_it_fires(pc):
    pc.post("/api/patients/1/push/subscribe", json=SUB)
    at = datetime.fromisoformat(pc.get("/api/patients/1/prompt").json()["at"])
    pc.now = at - timedelta(minutes=1)
    assert deliver(pc) == 0  # not yet
    pc.now = at + timedelta(seconds=30)
    assert deliver(pc) == 1
    assert deliver(pc) == 0  # once per prompt


def test_no_push_when_answered_late_paused_or_off(pc):
    pc.post("/api/patients/1/push/subscribe", json=SUB)
    with Session(db.engine) as s:
        prompt.send_now(s, 1, pc.now - timedelta(hours=3))  # fired 3 h ago: outside the on-time window
        s.commit()
    assert deliver(pc) == 0
    pc.put("/api/patients/1/prompt/settings", json={"pause_today": True})
    with Session(db.engine) as s:
        prompt.send_now(s, 1, pc.now)
        s.commit()
    assert deliver(pc) == 0
    pc.put("/api/patients/1/prompt/settings", json={"pause_today": False, "enabled": False})
    assert deliver(pc) == 0
    assert pc.sent == []


def test_no_push_in_quiet_hours(pc):
    pc.post("/api/patients/1/push/subscribe", json=SUB)
    with Session(db.engine) as s:
        prompt.send_now(s, 1, pc.now)  # demo prompt at 10:00 local...
        st = prompt.settings_for(s, 1)
        st.quiet_start, st.quiet_end = "09:30", "11:00"  # ...inside quiet hours set afterwards
        s.commit()
    assert deliver(pc) == 0


def test_gone_device_is_removed_other_errors_kept(pc):
    pc.post("/api/patients/1/push/subscribe", json=SUB)
    pc.fail = RuntimeError("network down")
    assert pc.post("/api/patients/1/push/test").json()["sent"] == 0
    assert devices() == 1
    pc.fail = Gone(410)
    pc.post("/api/patients/1/push/test")
    assert devices() == 0


def test_loop_stops(pc):
    stop = threading.Event()
    stop.set()
    push.run_loop(stop, db.engine, lambda: pc.now)  # returns at once when stopped
