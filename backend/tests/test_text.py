"""Text branch (pipeline/text.py) and the written-diary endpoint, on the real exported model."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import baseline as baseline_mod
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


@pytest.fixture()
def no_audio_models(monkeypatch):
    """Spoken check-in without openSMILE / wav2vec2 – these tests are about the transcript path."""
    monkeypatch.setattr(pipeline.audio, "load", lambda path: None)
    monkeypatch.setattr(pipeline.audio, "active_seconds", lambda x: 20.0)
    monkeypatch.setattr(pipeline.prosody, "extract_full_signal", lambda x: None)
    monkeypatch.setattr(pipeline.prosody, "from_full", lambda full: {"speech_rate": 2.0})
    monkeypatch.setattr(pipeline.emotion, "run_signal", lambda x: (None, {"arousal": 0.5, "valence": 0.5}))


class FakeGemini:
    """Stands in for google-genai; records the request, never touches the network."""

    def __init__(self, transcript=None, error=None):
        self.transcript, self.error, self.calls = transcript, error, []
        self.interactions = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return type("R", (), {"output_text": self.transcript})()


def spoken(monkeypatch, tmp_path, fake, key="test-key"):
    monkeypatch.setattr(pipeline.llm.settings, "gemini_api_key", key)
    monkeypatch.setattr(pipeline.llm, "_client", lambda: fake)
    wav = tmp_path / "checkin.wav"
    wav.write_bytes(b"RIFF....WAVE")
    return pipeline.extract_features(str(wav))


def test_no_key_uses_stub_and_nothing_reaches_the_score(monkeypatch, tmp_path, no_audio_models):
    """STATUS.md T2: without a key the stub is used, and a stub adds no llm or text features."""
    fake = FakeGemini("should not be called")
    res = spoken(monkeypatch, tmp_path, fake, key="")
    sources = {src for _, _, src in res.features}
    assert FeatureSource.llm not in sources and FeatureSource.text not in sources
    assert fake.calls == [] and res.crisis is False


def test_transcript_feeds_the_text_branch(monkeypatch, tmp_path, no_audio_models):
    fake = FakeGemini(ENTRIES["low_1"])
    res = spoken(monkeypatch, tmp_path, fake)
    assert res.transcript == ENTRIES["low_1"]
    assert {n for n, _, src in res.features if src == FeatureSource.text} == {
        "text_logit", "absolutist", "first_person", "sentiment"}
    call = fake.calls[0]  # privacy and mode settings actually sent
    assert call["store"] is False
    assert call["model"] == "gemini-3.5-transcribe"
    assert call["generation_config"]["transcription_config"] == {
        "language_codes": ["en-US"], "mode": {"type": "verbatim"}}
    assert call["input"][0]["type"] == "audio" and call["input"][0]["mime_type"] == "audio/wav"


def test_crisis_phrase_in_transcript_flags(monkeypatch, tmp_path, no_audio_models):
    res = spoken(monkeypatch, tmp_path, FakeGemini("um I just, I keep thinking about ending my life"))
    assert res.crisis is True


def test_api_failure_falls_back_to_stub(monkeypatch, tmp_path, no_audio_models):
    res = spoken(monkeypatch, tmp_path, FakeGemini(error=RuntimeError("503 unavailable")))
    assert FeatureSource.text not in {src for _, _, src in res.features}


def test_retries_when_model_overloaded(monkeypatch, tmp_path, no_audio_models):
    class FlakyGemini(FakeGemini):
        def create(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                raise RuntimeError("503 UNAVAILABLE: the model is overloaded")
            return type("R", (), {"output_text": ENTRIES["good_1"]})()

    monkeypatch.setattr(pipeline.llm.time, "sleep", lambda s: None)
    fake = FlakyGemini()
    res = spoken(monkeypatch, tmp_path, fake)
    transcribe_calls = [c for c in fake.calls if c["model"] == pipeline.llm.settings.transcribe_model]
    assert len(transcribe_calls) == 2 and res.transcript == ENTRIES["good_1"]  # (+1 content-rating call, #28)


def test_does_not_retry_spend_cap(monkeypatch, tmp_path, no_audio_models):
    monkeypatch.setattr(pipeline.llm.time, "sleep", lambda s: None)
    fake = FakeGemini(error=RuntimeError("429 RESOURCE_EXHAUSTED: spend cap reached"))
    res = spoken(monkeypatch, tmp_path, fake)
    assert len(fake.calls) == 1  # no hammering the API; check-in continues on voice alone
    assert FeatureSource.text not in {src for _, _, src in res.features}


def test_refuses_androids_audio(monkeypatch, tmp_path):
    fake = FakeGemini("x")
    monkeypatch.setattr(pipeline.llm, "_client", lambda: fake)
    clip = tmp_path / "data" / "androids" / "01_CF56_1.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="Androids"):
        pipeline.llm.transcribe(str(clip))
    assert fake.calls == []


# ---------------------------------------------------------------- written diary through the engine

@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    # One diary per day, as in real use (check-ins on the same day are combined, STATUS.md B16)
    c.day = 0
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    monkeypatch.setattr(baseline_mod, "clock", lambda: start + timedelta(days=c.day))
    return c


def post_text(c, text, slot="evening"):
    r = c.post("/api/checkins/text", json={"patient_id": 1, "slot": slot, "text": text})
    c.day += 1
    return r


def test_written_diary_builds_baseline_then_flags_a_low_day(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 5})
    for i in range(1, 6):  # five good days = the healthy baseline
        r = post_text(client, ENTRIES[f"good_{i}"])
        assert r.status_code == 200, r.text
    out = r.json()
    assert out["baseline"]["status"] == "active"
    assert {f["source"] for f in out["features"]} == {"text"}

    first = post_text(client, ENTRIES["low_1"]).json()
    assert first["state"]["branches"]["text"] > 1  # one low diary is clearly worse than this person's good days
    for k in (2, 3, 4):  # … and a run of low days lifts S over 1 (one diary alone moves S at most α·3, B34)
        out = post_text(client, ENTRIES[f"low_{k}"]).json()
    assert out["state"]["S_anchored"] > 1


def test_written_diary_crisis_goes_to_clinician(client):
    out = post_text(client, "I can't do this anymore. Sometimes I think about ending my life.").json()
    assert out["crisis"] is True
    assert out["state"]["level"] == 3


def test_written_diary_rejects_bad_input(client):
    assert post_text(client, "   ").status_code == 400
    r = client.post("/api/checkins/text", json={"patient_id": 99, "slot": "evening", "text": "hello"})
    assert r.status_code == 404
