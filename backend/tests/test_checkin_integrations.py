"""Integration regressions for the new UI without changing Kasper's scoring contracts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from app import baseline, db
from app.main import app
from app.models import Arm, CheckIn, Feature, FeatureSource, InterventionLog, StateScore
from app.pipeline import PipelineResult
from app.routers import checkins
from app.seed import seed

DIARY = ("I went for a gentle walk this morning and made breakfast. Later I called a friend "
         "and we talked about the books we have been reading. It was a quiet ordinary day.")


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.now = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
    monkeypatch.setattr(baseline, "clock", lambda: c.now)
    monkeypatch.setattr(checkins, "extract_features", lambda _: PipelineResult(
        features=[("speech_rate", 2.0, FeatureSource.prosody)], transcript=DIARY, speech_s=12))
    return c


def memo(c, key=None):
    return c.post("/api/checkins/text", json={"patient_id": 1, "slot": "morning", "text": DIARY},
                  headers={"Idempotency-Key": key} if key else {})


@pytest.mark.parametrize("spoken", [False, True])
def test_network_retry_replays_the_original_response_without_new_evidence(client, spoken):
    def submit():
        if spoken:
            return client.post("/api/checkins", data={"patient_id": 1, "slot": "morning"},
                               files={"audio": ("memo.wav", b"RIFF audio", "audio/wav")},
                               headers={"Idempotency-Key": "request-one"})
        return memo(client, "request-one")
    first, second = submit(), submit()
    assert first.status_code == second.status_code == 200, second.text
    assert first.json() == second.json()
    with Session(db.engine) as s:
        assert len(s.exec(select(CheckIn)).all()) == 1
        assert len(s.exec(select(Feature)).all()) == len(first.json()["features"])
    rewards = client.get("/api/patients/1/rewards").json()
    assert rewards["balance"] == 10 and rewards["xp"] == 10
    detail = client.get("/api/patients/1").json()["checkins"][0]
    assert "request_key" not in detail and "response_json" not in detail


def test_retry_does_not_append_a_half_weight_checkin_or_change_exact_explanation(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 5})
    for _ in range(6):
        assert memo(client).status_code == 200
        client.now += timedelta(days=1)
    first = memo(client, "with-baseline").json()
    retry = memo(client, "with-baseline").json()
    assert first == retry and first["state"]["explanation"] is not None
    with Session(db.engine) as s:
        score = s.exec(select(StateScore).where(StateScore.date == client.now.date())).one()
        day = json.loads(score.branch_scores_json)
        assert day["w"]["anchored"]["text"] == [1.0]
    # A distinct extra entry still uses Kasper's B31 half-weight rule.
    assert memo(client, "a-distinct-entry").status_code == 200
    with Session(db.engine) as s:
        score = s.exec(select(StateScore).where(StateScore.date == client.now.date())).one()
        assert json.loads(score.branch_scores_json)["w"]["anchored"]["text"] == [1.0, 0.5]


def test_followup_retry_and_edit_replace_one_bandit_trial(client):
    memo(client)
    with Session(db.engine) as s:
        arm_key = s.exec(select(InterventionLog)).one().arm_key
    body = {"patient_id": 1, "date": client.now.date().isoformat(), "completed": True, "rating": 5}
    assert client.post("/api/interventions/feedback", json=body).status_code == 200
    assert client.post("/api/interventions/feedback", json=body).json()["rewards"] == []
    with Session(db.engine) as s:
        arm = s.exec(select(Arm).where(Arm.key == arm_key)).one()
        assert arm.a == 1.5 and arm.b == 1.5
    body["completed"] = False
    client.post("/api/interventions/feedback", json=body)
    with Session(db.engine) as s:
        arm = s.exec(select(Arm).where(Arm.key == arm_key)).one()
        assert arm.a == 1.0 and arm.b == 2.0


def test_optional_phrase_transcription_failure_preserves_the_main_memo(client, monkeypatch):
    paths = []
    def fail(path):
        paths.append(path)
        raise RuntimeError("network unavailable")
    monkeypatch.setattr(checkins, "transcribe_phrase", fail)
    monkeypatch.setattr(checkins, "add_phrase", lambda result, path, transcript: None)
    out = client.post("/api/checkins", data={"patient_id": 1, "slot": "morning"}, files={
        "audio": ("memo.wav", b"RIFF memo", "audio/wav"),
        "phrase": ("phrase.wav", b"RIFF phrase", "audio/wav"),
    })
    assert out.status_code == 200, out.text
    assert paths and all(not Path(path).exists() for path in paths)


def test_wearable_sync_cannot_answer_a_daily_prompt(client):
    assert client.post("/api/patients/1/prompt/now").status_code == 200
    response = client.post("/api/checkins/wearable", json={"patient_id": 1, "active_minutes": 25})
    assert response.status_code == 200, response.text
    assert response.json()["prompt"] is None and response.json()["state"]["S"] is None
    status = client.get("/api/patients/1/prompt").json()
    assert status["state"] == "open" and status["answered_at"] is None
    rewards = client.get("/api/patients/1/rewards").json()
    assert rewards["checked_in_today"] is False
    assert any(r["reason"] == "habit_active" for r in response.json()["rewards"])


@pytest.mark.parametrize("field,value", [
    ("steps", -1), ("steps", float("nan")), ("steps", float("inf")),
    ("sleep_hours", 25), ("sleep_midpoint", -25), ("resting_hr", 0),
    ("hrv_rmssd", -1), ("active_minutes", 1441),
    ("minute_steps", [-1] * 1440), ("minute_steps", [float("inf")] * 1440),
])
def test_wearable_validation_is_finite_and_within_bounds(field, value):
    from pydantic import ValidationError
    from app.schemas import WearableDayIn
    with pytest.raises(ValidationError):
        WearableDayIn.model_validate({"patient_id": 1, field: value})


def test_midnight_and_exercise_inputs_preserve_health_contract(client):
    response = client.post("/api/checkins/wearable", json={
        "patient_id": 1, "sleep_hours": 8, "sleep_midpoint": -0.5,
        "minute_steps": [300.0] * 20 + [0.0] * 1420, "active_minutes": 5,
    })
    assert response.status_code == 200, response.text
    assert all(r["reason"] != "habit_active" for r in response.json()["rewards"])
