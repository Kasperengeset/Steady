"""Text branch (pipeline/text.py) and the written-diary endpoint, on the real exported model."""

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import db
from app import pipeline
from app.main import app
from app.models import FeatureSource
from app.pipeline import text as tp
from app.seed import seed

DIARY = Path(__file__).resolve().parents[2] / "ml" / "text" / "diary_sanity.csv"
ENTRIES = {r["id"]: r["text"] for r in csv.DictReader(open(DIARY, encoding="utf-8")) if r["variant"] == "as written"}


# ---------------------------------------------------------------- features and safety

def test_features_and_label():
    low = tp.analyze_text(ENTRIES["low_1"])
    good = tp.analyze_text(ENTRIES["good_1"])
    assert set(low["features"]) == {"text_logit", "absolutist", "first_person", "sentiment"}
    assert 0 <= good["p_depressive"] < 0.5 <= low["p_depressive"] <= 1
    assert low["features"]["text_logit"] > good["features"]["text_logit"]
    assert low["label"] == "depressive language" and good["label"] == "everyday language"


def test_masking_leaves_no_bridging_bigram():
    m = tp.mask_text(tp.clean_text("I feel like my depression is back, my therapist upped my meds"))
    assert "depress" not in m.lower() and "therapist" not in m.lower()
    toks = tp.analyze(m)
    assert "my is" not in toks and "is back" in toks


def test_absolutist_skips_negation():
    f = tp.lexicon_features(tp.prepare("not always sad but nothing ever helps"))
    words = 7
    assert f["absolutist"] == pytest.approx(100 * 2 / words)  # nothing, ever; "not always" skipped


@pytest.mark.parametrize("text, expected", [
    ("Sometimes I think about ending my life.", True),
    ("I don't want to be alive anymore", True),
    ("Jeg tenker på selvmord", True),
    ("I killed it at work today and my mum is proud", False),
    ("Long day, tired but okay.", False),
])
def test_crisis_phrases(text, expected):
    assert tp.analyze_text(text)["crisis"] is expected


def test_stub_transcript_never_reaches_the_score(monkeypatch):
    """STATUS.md T2: while llm.analyze is a stub, a spoken check-in adds no llm or text features."""
    monkeypatch.setattr(pipeline.audio, "load", lambda path: None)
    monkeypatch.setattr(pipeline.prosody, "extract_full_signal", lambda x: None)
    monkeypatch.setattr(pipeline.prosody, "from_full", lambda full: {"speech_rate": 2.0})
    monkeypatch.setattr(pipeline.emotion, "run_signal", lambda x: (None, {"arousal": 0.5, "valence": 0.5}))
    res = pipeline.extract_features("unused.wav")
    sources = {src for _, _, src in res.features}
    assert FeatureSource.llm not in sources and FeatureSource.text not in sources
    assert res.crisis is False


# ---------------------------------------------------------------- written diary through the engine

@pytest.fixture()
def client():
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    return TestClient(app)


def post_text(c, text, slot="evening"):
    return c.post("/api/checkins/text", json={"patient_id": 1, "slot": slot, "text": text})


def test_written_diary_builds_baseline_then_flags_a_low_day(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 5})
    for i in range(1, 6):  # five good days = the healthy baseline
        r = post_text(client, ENTRIES[f"good_{i}"])
        assert r.status_code == 200, r.text
    out = r.json()
    assert out["baseline"]["status"] == "active"
    assert {f["source"] for f in out["features"]} == {"text"}

    out = post_text(client, ENTRIES["low_1"]).json()
    assert out["state"]["branches"].get("text") is not None
    assert out["state"]["S_anchored"] > 1  # clearly worse than this person's good days


def test_written_diary_crisis_goes_to_clinician(client):
    out = post_text(client, "I can't do this anymore. Sometimes I think about ending my life.").json()
    assert out["crisis"] is True
    assert out["state"]["level"] == 3


def test_written_diary_rejects_bad_input(client):
    assert post_text(client, "   ").status_code == 400
    r = client.post("/api/checkins/text", json={"patient_id": 99, "slot": "evening", "text": "hello"})
    assert r.status_code == 404
