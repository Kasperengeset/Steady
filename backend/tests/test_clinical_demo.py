"""A synthetic reference exercises real scoring; new live entries remain independently auditable."""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, col, select

from app import baseline, clinical_demo, db
from app.main import app
from app.models import Baseline, BaselineStat, BaselineStatus, CheckIn, Feature, FeatureSource, Patient, StateScore
from app.pipeline import llm, text as text_model
from app.seed import seed

NOW = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
LIVE_DIARY = (
    "I went outside for a walk this morning and made myself breakfast. I felt a little tired, "
    "but I enjoyed the fresh air and a conversation with a friend. Later I read a chapter of "
    "my book and cooked dinner. I was glad that I made time for a few ordinary things today."
)


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    monkeypatch.setattr(baseline, "clock", lambda: NOW)
    return TestClient(app)


def create(client, scenario="drift"):
    response = client.post("/api/clinician/demo", json={"scenario": scenario})
    assert response.status_code == 200, response.text
    return response.json()


def test_append_only_demo_is_idempotent_and_preserves_existing_people(client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A clinical seed must not call an external model.")
    monkeypatch.setattr(llm, "analyze", forbidden)
    monkeypatch.setattr(llm, "rate_text", forbidden)
    clock = baseline.clock
    with Session(db.engine) as session:
        original = session.get(Patient, 1).model_dump()
    first = create(client)
    second = create(client)
    assert first["created"] and not second["created"]
    assert first["patient_id"] == second["patient_id"] != 1
    assert first["entry_count"] == second["entry_count"] == 72
    assert first["synthetic_entry_count"] == 72 and first["live_entry_count"] == 0
    assert "Synthetic demonstration" in first["disclosure"]
    assert baseline.clock is clock
    with Session(db.engine) as session:
        assert session.get(Patient, 1).model_dump() == original
        assert session.exec(select(CheckIn).where(CheckIn.patient_id == 1)).all() == []
        assert len(session.exec(select(Patient)).all()) == 2


def test_healthy_reference_covers_all_three_branches_and_drift_uses_real_engine(client):
    result = create(client)
    pid = result["patient_id"]
    with Session(db.engine) as session:
        reference = session.exec(select(Baseline).where(Baseline.patient_id == pid, Baseline.status == BaselineStatus.active)).one()
        assert reference.period_days == 14
        assert "SYNTHETIC" in reference.note
        assert "not clinical confirmation" in reference.confirmed_by
        stats = session.exec(select(BaselineStat).where(BaselineStat.baseline_id == reference.id, BaselineStat.slot == "all")).all()
        assert {"speech_rate", "arousal", "text_logit", "sentiment", "steps", "sleep_hours"} <= {row.feature for row in stats}
        assert all(row.n >= 14 for row in stats)
        baseline_entries = session.exec(select(CheckIn).where(CheckIn.patient_id == pid, CheckIn.ts <= reference.locked_at)).all()
        assert len(baseline_entries) == 42
        assert all(row.data_origin == "synthetic" for row in baseline_entries)
        scores = session.exec(select(StateScore).where(StateScore.patient_id == pid).order_by(col(StateScore.date))).all()
        assert len(scores) == 10
        assert scores[-1].S > 1.0
        assert {"acoustic", "text", "health"} <= set(json.loads(scores[-1].branch_scores_json)["anchored"])
        assert json.loads(scores[-1].evidence_json)["anchored"]["text"]


def test_synthetic_diaries_use_actual_local_text_model_and_keep_feature_provenance(client):
    result = create(client)
    with Session(db.engine) as session:
        entries = session.exec(select(CheckIn).where(CheckIn.patient_id == result["patient_id"]).order_by(col(CheckIn.ts))).all()
        written = next(entry for entry in entries if entry.transcript)
        assert written.transcript in clinical_demo.GOOD
        actual = text_model.analyze_text(written.transcript)
        stored = {feature.name: feature.value for feature in session.exec(select(Feature).where(Feature.checkin_id == written.id)).all()}
        assert stored == pytest.approx(actual["features"])
        assert written.analysis_json is not None
        assert json.loads(written.llm_json)["demo"]["text_analysis"] == "actual local TF-IDF and lexicon model"
        voice = next(entry for entry in entries if entry.speech_s is not None)
        assert "no recording analysed" in json.loads(voice.llm_json)["demo"]["measurement"]


def test_new_actual_diary_is_live_persists_real_features_and_idempotent_analysis(client):
    result = create(client)
    pid = result["patient_id"]
    before_workspace = client.get(f"/api/clinician/patients/{pid}/workspace?days=30")
    assert before_workspace.status_code == 200, before_workspace.text
    before_workspace = before_workspace.json()
    assert before_workspace["coverage"]["total_entries"] == 72
    assert before_workspace["coverage"]["synthetic_entries"] == 72
    assert before_workspace["baseline"]["active"]["origin"] == "synthetic"
    assert all(source["entries"] == 24 and source["live_entries"] == 0 for source in before_workspace["sources"])
    body = {"patient_id": pid, "slot": "evening", "text": LIVE_DIARY}
    headers = {"Idempotency-Key": "actual-diary-after-synthetic-baseline"}
    first = client.post("/api/checkins/text", json=body, headers=headers)
    retry = client.post("/api/checkins/text", json=body, headers=headers)
    assert first.status_code == retry.status_code == 200, first.text
    assert first.json() == retry.json()
    response = first.json()
    assert response["state"]["S"] is not None and response["analysis"]
    audit = client.get(f"/api/clinician/entries/{response['checkin_id']}")
    assert audit.status_code == 200, audit.text
    audit = audit.json()
    assert audit["entry"]["data_origin"] == "live" and audit["entry"]["kind"] == "text"
    assert audit["analysis"] == response["analysis"]
    assert audit["analysis"]["provenance"] == "snapshot"
    assert audit["analysis"]["baseline_origin"] == "synthetic"
    assert audit["analysis"]["before"]["S"] == before_workspace["current"]["S"]
    assert audit["word_evidence"]["available"]
    after_workspace = client.get(f"/api/clinician/patients/{pid}/workspace?days=30").json()
    assert after_workspace["coverage"]["total_entries"] == 73
    assert after_workspace["coverage"]["live_entries"] == 1
    source_live_counts = {source["key"]: source["live_entries"] for source in after_workspace["sources"]}
    assert source_live_counts == {"acoustic": 0, "text": 1, "health": 0}
    page = client.get(f"/api/clinician/patients/{pid}/entries?limit=200").json()
    assert page["total"] == 73 and len(page["entries"]) == 73
    assert sum(entry["kind"] == "wearable" for entry in page["entries"]) == 24
    assert sum(entry["kind"] == "voice" for entry in page["entries"]) == 24
    assert sum(entry["kind"] == "text" for entry in page["entries"]) == 25
    with Session(db.engine) as session:
        entries = session.exec(select(CheckIn).where(CheckIn.patient_id == pid)).all()
        assert len(entries) == 73
        live = session.get(CheckIn, response["checkin_id"])
        assert live.data_origin == "live" and live.transcript == LIVE_DIARY
        assert json.loads(live.analysis_json) == response["analysis"]
        features = session.exec(select(Feature).where(Feature.checkin_id == live.id)).all()
        assert features and all(feature.source == FeatureSource.text for feature in features)
        assert {feature.name: feature.value for feature in features} == pytest.approx(text_model.analyze_text(LIVE_DIARY)["features"])
        score = session.exec(select(StateScore).where(StateScore.patient_id == pid, StateScore.date == NOW.date())).one()
        assert json.loads(score.evidence_json)["anchored"]["text"]
        assert json.loads(score.branch_scores_json)["w"]["anchored"]["text"] == [1.0]
    again = create(client)
    assert again["entry_count"] == 73 and again["live_entry_count"] == 1


@pytest.mark.parametrize("scenario,expected,status", [("stable", 72, "active"), ("building", 21, "building")])
def test_alternate_narratives_are_separate_profiles(client, scenario, expected, status):
    out = create(client, scenario)
    assert out["scenario"] == scenario and out["entry_count"] == expected
    assert out["baseline_status"] == status
    with Session(db.engine) as session:
        assert session.get(Patient, out["patient_id"]).demo_key.endswith(scenario)
        if scenario == "building":
            assert session.exec(select(StateScore).where(StateScore.patient_id == out["patient_id"])).all() == []
