"""Real local text inference, exact audit snapshots, and honest provenance in clinician APIs."""

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, select

from app import baseline, db
from app.config import settings
from app.main import app
from app.models import BaselineStat, CheckIn, Feature, FeatureSource, Patient, Slot
from app.pipeline import extract_text_features
from app.routers.checkins import _record_checkin
from app.seed import seed

GOOD = ("I made breakfast and walked by the water this morning. My friend called and we laughed about "
        "our plans for the weekend. I cooked dinner and read a book before going to bed.")
LOW = ("I stayed in bed all morning and could not find the energy to do anything. I cancelled plans "
       "with my friends again. Everything feels heavy and I am struggling to care about things.")


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.now = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=1)
    monkeypatch.setattr(baseline, "clock", lambda: c.now)
    monkeypatch.setattr(settings, "rate_text_enabled", False)
    return c


def diary(client, text=GOOD, key=None):
    response = client.post("/api/checkins/text", json={"patient_id": 1, "slot": "morning", "text": text},
                           headers={"Idempotency-Key": key} if key else {})
    assert response.status_code == 200, response.text
    return response.json()


def reference(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 5})
    for _ in range(5):
        diary(client)
        client.now += timedelta(days=1)


def test_real_entry_is_logged_with_actual_local_model_outputs_and_word_evidence(client):
    result = diary(client, key="clinical-live-one")
    detail = client.get(f"/api/clinician/entries/{result['checkin_id']}")
    assert detail.status_code == 200, detail.text
    out = detail.json()
    assert out["entry"]["kind"] == "text" and out["entry"]["data_origin"] == "live"
    assert out["transcript"] == GOOD and out["analysis"]["provenance"] == "snapshot"
    actual_logit = next(f["value"] for f in out["features"] if f["name"] == "text_logit")
    assert out["raw_outputs"]["text"]["logit"] == actual_logit
    assert out["raw_outputs"]["text"]["p_depressive"] == pytest.approx(1 / (1 + math.exp(-actual_logit)))
    assert out["word_evidence"]["available"] and out["word_evidence"]["provenance"] == "at_ingestion"
    assert out["analysis"]["scoring_status"] == "collecting_baseline"
    assert out["analysis"]["after"] is None
    assert client.get("/api/patients/1").json()["checkins"][0].get("analysis_json") is None
    retry = diary(client, key="clinical-live-one")
    assert retry == result
    assert client.get("/api/clinician/patients/1/entries").json()["total"] == 1


def test_snapshot_freezes_references_and_exact_branch_contributions(client):
    reference(client)
    result = diary(client, LOW, key="clinical-scored")
    analysis = result["analysis"]
    assert analysis["score_updated"] and analysis["baseline_version"] == 1
    assert analysis["baseline_origin"] == "live"
    tracks = analysis["tracks"]["anchored"]
    assert tracks["D"]["text"] == pytest.approx(sum(tracks["shares"]["text"].values()), abs=1e-4)
    assert analysis["entry_weights"]["text"] == 1
    assert analysis["day_weights"]["anchored"]["text"] == [1]
    assert analysis["after"]["S"] == result["state"]["S"]
    assert "text_logit" in tracks["stats"]["text"]
    with Session(db.engine) as s:
        row = s.get(CheckIn, result["checkin_id"])
        assert json.loads(row.analysis_json) == analysis
        for stat in s.exec(select(BaselineStat)).all():
            stat.center += 100
            s.add(stat)
        s.commit()
    detail = client.get(f"/api/clinician/entries/{result['checkin_id']}").json()
    assert detail["analysis"] == analysis  # reads never substitute today's changed reference
    assert any(f["used"] and f["z"] is not None and f["center"] is not None for f in detail["features"])


def test_synthetic_reference_is_explicit_and_new_entries_still_live(client):
    with Session(db.engine) as s:
        patient = s.get(Patient, 1)
        patient.demo_key = "clinical-unit-demo"
        s.add(patient)
        baseline.start(s, 1, period_days=0, min_checkins=5, now=client.now - timedelta(days=6), note="Synthetic reference for software demonstration.")
        s.commit()
        for day in range(5):
            _record_checkin(s, patient, Slot.morning, extract_text_features(GOOD),
                            recorded_at=client.now - timedelta(days=6 - day), data_origin="synthetic")
    live = diary(client, LOW)
    assert live["analysis"]["data_origin"] == "live"
    assert live["analysis"]["baseline_origin"] == "synthetic"
    workspace = client.get("/api/clinician/patients/1/workspace?days=7").json()
    assert workspace["patient"]["demo_key"] == "clinical-unit-demo"
    assert workspace["baseline"]["origin"] == "synthetic"
    assert workspace["coverage"]["synthetic_entries"] == 5 and workspace["coverage"]["live_entries"] == 1
    overview = client.get("/api/clinician/overview").json()
    assert overview["summary"]["live_entries_today"] == 1
    assert {m["key"] for m in overview["models"]} == {"acoustic", "text", "health"}
    assert all(len(m["artifact_sha256"]) == 64 for m in overview["models"])
    assert any("not validated" in line for line in overview["limitations"])


def test_legacy_reconstruction_never_claims_original_entry_weights_or_after_score(client):
    reference(client)
    with Session(db.engine) as s:
        row = CheckIn(patient_id=1, slot=Slot.morning, ts=client.now, transcript=LOW)
        s.add(row)
        s.flush()
        for name, value, src in extract_text_features(LOW).features:
            s.add(Feature(checkin_id=row.id, name=name, value=value, source=src))
        s.commit()
        row_id = row.id
    out = client.get(f"/api/clinician/entries/{row_id}").json()
    assert out["entry"]["data_origin"] == "unknown"
    assert out["analysis"]["provenance"] == "reconstructed"
    assert out["analysis"]["before"] is None and out["analysis"]["after"] is None
    assert out["analysis"]["entry_weights"] == {} and not out["analysis"]["score_updated"]
    assert all(not f["usage_known"] for f in out["features"])
    assert out["analysis"]["limitations"]
    with Session(db.engine) as s:
        assert s.get(CheckIn, row_id).analysis_json is None  # inspecting old entries does not rewrite history


def test_all_sources_and_pagination_are_visible_without_fabricated_voice_probability(client):
    diary(client)
    wear = client.post("/api/checkins/wearable", json={"patient_id": 1, "steps": 6500, "sleep_hours": 7.5})
    assert wear.status_code == 200, wear.text
    page = client.get("/api/clinician/patients/1/entries?limit=1&offset=0").json()
    assert page["total"] == 2 and len(page["entries"]) == 1 and page["entries"][0]["kind"] == "wearable"
    detail = client.get(f"/api/clinician/entries/{wear.json()['checkin_id']}").json()
    assert detail["raw_outputs"]["text"] is None and detail["raw_outputs"]["acoustic"] == {}
    assert detail["transcript"] is None and detail["transcription_status"] == "not-applicable"
    assert not detail["word_evidence"]["available"]
    sources = client.get("/api/clinician/patients/1/workspace").json()["sources"]
    assert next(x for x in sources if x["key"] == "health")["entries"] == 1
    assert next(x for x in sources if x["key"] == "acoustic")["entries"] == 0


def test_missing_records_and_query_bounds(client):
    assert client.get("/api/clinician/patients/999/workspace").status_code == 404
    assert client.get("/api/clinician/patients/999/entries").status_code == 404
    assert client.get("/api/clinician/entries/999").status_code == 404
    assert client.get("/api/clinician/patients/1/entries?limit=0").status_code == 422
    assert client.get("/api/clinician/patients/1/workspace?days=1").status_code == 422


def test_clinical_risk_stays_visible_before_baseline_scoring(client):
    with Session(db.engine) as s:
        patient = s.get(Patient, 1)
        patient.risk_level = 3
        s.add(patient)
        s.commit()
    cohort = client.get("/api/clinician/overview").json()["patients"][0]
    workspace = client.get("/api/clinician/patients/1/workspace").json()
    assert cohort["needs_attention"] and cohort["current"]["level"] == 3
    assert workspace["current"]["level"] == 3
    assert workspace["current"]["S"] is None and not workspace["current"]["has_score"]


def test_audit_failure_rolls_back_entry_features_and_rewards(client, monkeypatch):
    from app import clinical
    from app.models_rewards import RewardLedger

    def unavailable(*args, **kwargs):
        raise ValueError("Audit snapshot unavailable")

    monkeypatch.setattr(clinical, "capture_analysis", unavailable)
    with pytest.raises(ValueError, match="Audit snapshot unavailable"):
        diary(client, key="audit-must-be-atomic")
    with Session(db.engine) as s:
        assert s.exec(select(CheckIn)).all() == []
        assert s.exec(select(Feature)).all() == []
        assert s.exec(select(RewardLedger)).all() == []
