"""Wearable rewards, safe OAuth state handling, and repeat-sync protection."""

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from app import baseline, db, health_sync, rewards
from app.config import settings
from app.main import app
from app.models import CheckIn
from app.models_rewards import RewardLedger
from app.routers import health
from app.seed import seed


@pytest.fixture()
def client(monkeypatch, tmp_path):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    monkeypatch.setattr(health_sync, "TOKEN_DIR", tmp_path / "tokens")
    monkeypatch.setattr(baseline, "clock", lambda: datetime(2026, 11, 14, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(settings, "google_client_id", "")
    monkeypatch.setattr(settings, "google_client_secret", "")
    health._pending.clear()
    return TestClient(app)


def test_disconnected_status_and_configuration(client):
    out = client.get("/api/patients/1/health").json()
    assert not out["connected"] and not out["configured"]
    assert out["last_sync_at"] is None and out["source"] is None
    assert client.get("/api/patients/9999/health").status_code == 404
    assert client.post("/api/patients/1/health/connect").status_code == 409
    assert client.post("/api/patients/1/health/sync").status_code == 409


def test_demo_health_has_honest_source_and_cannot_duplicate_food_or_scores(client):
    one = client.post("/api/patients/1/health/demo")
    assert one.status_code == 200, one.text
    one = one.json()
    assert one["days_synced"] == 1
    assert one["status"]["source"] == "demo" and not one["status"]["connected"]
    assert {r["reason"] for r in one["rewards"]} == {"habit_sleep", "habit_steps", "habit_active"}
    two = client.post("/api/patients/1/health/demo").json()
    assert two["days_synced"] == 0 and two["skipped"] == 1 and two["rewards"] == []
    with Session(db.engine) as s:
        rows = s.exec(select(CheckIn)).all()
        assert len(rows) == 1
        assert rows[0].data_origin == "synthetic"
        assert json.loads(rows[0].llm_json)["wearable"]["measurement_date"] == "2026-11-13"
        assert len(s.exec(select(RewardLedger)).all()) == 3
        assert rewards.balance(s, 1) == 15
    companion = client.get("/api/patients/1/appie").json()
    assert all(h["done"] for h in companion["habits"] if h["key"] in ("sleep", "steps", "active"))


def test_sync_uses_completed_day_after_memo_without_rewinding_existing_checkins(client, monkeypatch):
    text = client.post("/api/checkins/text", json={"patient_id": 1, "slot": "morning", "text": "Went for a nice walk today."})
    assert text.status_code == 200
    class FakeHealth:
        errors = {}
        def identity(self):
            return {"linked": True}
        def fetch_days(self, start, end):
            today = baseline.clock().date()
            assert end == today
            return {today - timedelta(days=2): {"steps": 7000},
                    today - timedelta(days=1): {"steps": 8000, "sleep_hours": 7.5},
                    today: {"steps": 100}}
    monkeypatch.setattr(health_sync, "authorized_client", lambda _: FakeHealth())
    out = client.post("/api/patients/1/health/sync")
    assert out.status_code == 200, out.text
    assert out.json()["days_synced"] == 1 and out.json()["skipped"] == 2
    assert out.json()["status"]["source"] == "google_health"
    assert out.json()["status"]["last_day"] == "2026-11-13"
    with Session(db.engine) as s:
        rows = s.exec(select(CheckIn)).all()
        assert all(r.ts.date().isoformat() == "2026-11-14" for r in rows)
    repeat = client.post("/api/patients/1/health/sync").json()
    assert repeat["days_synced"] == 0 and repeat["rewards"] == []


def test_oauth_uses_pkce_single_use_state_and_validates_identity(client, monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "test-client")
    monkeypatch.setattr(settings, "google_client_secret", "test-secret")
    out = client.post("/api/patients/1/health/connect").json()
    params = parse_qs(urlparse(out["authorization_url"]).query)
    assert params["code_challenge_method"] == ["S256"]
    assert "test-secret" not in out["authorization_url"]
    state = params["state"][0]
    assert client.get("/api/google-health/callback?state=wrong&code=x").status_code == 400
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"access_token": "test-access-token", "refresh_token": "test-refresh-token"}
    calls = []
    def post(url, data, timeout):
        assert data["code_verifier"] and data["grant_type"] == "authorization_code"
        calls.append(url)
        return FakeResponse()
    class FakeHealth:
        def __init__(self, session):
            pass
        def identity(self):
            calls.append("identity")
            return {"linked": True}
    monkeypatch.setattr(health.requests, "post", post)
    monkeypatch.setattr(health, "GoogleHealth", FakeHealth)
    response = client.get(f"/api/google-health/callback?state={state}&code=test-code", follow_redirects=False)
    assert response.status_code == 303 and "health=connected" in response.headers["location"]
    assert calls[-1] == "identity"
    assert client.get("/api/patients/1/health").json()["connected"]
    assert "test-access-token" not in client.get("/api/patients/1/health").text
    assert client.get(f"/api/google-health/callback?state={state}&code=test-code").status_code == 400
    assert client.post("/api/patients/1/health/demo").status_code == 409
    assert client.delete("/api/patients/1/health/connection").status_code == 200
    assert not client.get("/api/patients/1/health").json()["connected"]


def test_oauth_cancel_never_connects(client, monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "test-client")
    monkeypatch.setattr(settings, "google_client_secret", "test-secret")
    url = client.post("/api/patients/1/health/connect").json()["authorization_url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    out = client.get(f"/api/google-health/callback?state={state}&error=access_denied", follow_redirects=False)
    assert out.status_code == 303 and "health=cancelled" in out.headers["location"]
    assert not client.get("/api/patients/1/health").json()["connected"]


def test_same_measurement_day_cannot_earn_again_after_midnight(client, monkeypatch):
    assert client.post("/api/patients/1/health/demo").json()["days_synced"] == 1
    monkeypatch.setattr(baseline, "clock", lambda: datetime(2026, 11, 15, 8, tzinfo=timezone.utc))
    class FakeHealth:
        errors = {}
        def identity(self):
            return {}
        def fetch_days(self, start, end):
            return {datetime(2026, 11, 13).date(): {"steps": 8000}}
    monkeypatch.setattr(health_sync, "authorized_client", lambda _: FakeHealth())
    result = client.post("/api/patients/1/health/sync").json()
    assert result["days_synced"] == 0 and result["rewards"] == []
    with Session(db.engine) as s:
        assert len(s.exec(select(CheckIn)).all()) == 1
