"""Fixed reading phrase (issue #21, STATUS.md B22): reading check, optional upload, own baseline,
and the phrase never moves D. Signal processing is mocked; these tests check the plumbing."""

import random

import numpy as np
import pytest
from sqlmodel import Session, SQLModel, select

from app import db, pipeline
from app.engine import state
from app.models import CheckIn, Feature
from app.pipeline import audio, phrase
from app.routers import checkins as checkins_router
from app.seed import seed

from .test_baseline import client, features  # noqa: F401 – fixture reused

SR = audio.SR


def tone(seconds: float, lead: float = 0.5, tail: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    voice = 0.3 * np.sin(2 * np.pi * 180 * np.arange(n) / SR) + 0.02 * rng.standard_normal(n)
    return np.concatenate([np.zeros(int(lead * SR)), voice, np.zeros(int(tail * SR))]).astype(np.float32)


# ---------------------------------------------------------------- reading check (pure)

def test_wer_zero_for_the_phrase_with_fillers_spelling_and_split_compounds():
    assert phrase.wer(phrase.PHRASE) == 0
    said = ("Um, when the sun light strikes rain drops in the air, uh, they act as a prism and form a rainbow. "
            "The rainbow is a division of white light into many beautiful colours.")
    assert phrase.wer(said) == 0


def test_wer_counts_substitutions_and_deletions():
    ref = phrase.REFERENCE
    one_off = phrase.PHRASE.replace("prism", "prison")
    assert phrase.wer(one_off) == pytest.approx(1 / len(ref))
    first_sentence = phrase.PHRASE.split(". ")[0]
    assert 0.4 < phrase.wer(first_sentence) < 0.6  # half the words missing


def test_wer_gate_rejects_free_talk_instead_of_the_phrase():
    check = phrase.reading_check(9.0, "I slept badly and I don't really want to read anything today, sorry.")
    assert check["wer"] > phrase.MAX_WER
    assert check["ok"] is False and "WER" in check["reason"]


def test_reading_check_passes_a_normal_read_and_gives_the_syllable_rate():
    check = phrase.reading_check(9.0, phrase.PHRASE)
    assert check["ok"] and check["wer"] == 0 and check["checked_by"] == "transcript"
    assert check["syllables_per_s"] == pytest.approx(41 / 9, abs=1e-3)


def test_without_transcript_only_timing_is_checked():
    assert phrase.reading_check(9.0, None)["ok"]
    assert phrase.reading_check(9.0, None)["checked_by"] == "timing"
    assert not phrase.reading_check(0.8, None)["ok"]  # a tap, not a read
    assert not phrase.reading_check(40.0, None)["ok"]  # ~1 syllable/s: not the phrase


def test_speech_span_trims_silence_and_active_seconds_skip_pauses():
    x = tone(6.0, lead=1.0, tail=2.0)
    assert audio.speech_span(x) == pytest.approx(6.0, abs=0.05)
    with_pause = np.concatenate([tone(3.0, 0.5, 0), np.zeros(2 * SR, np.float32), tone(3.0, 0, 0.5)])
    assert audio.speech_span(with_pause) == pytest.approx(8.0, abs=0.05)
    assert audio.active_seconds(with_pause) == pytest.approx(6.0, abs=0.05)
    assert audio.speech_span(np.zeros(SR, np.float32)) == 0.0


# ---------------------------------------------------------------- upload via the API

@pytest.fixture()
def phrase_models(monkeypatch):
    """The phrase recording: 9 s of 'speech', fake openSMILE/wav2vec2 values set per test."""
    vals = {"x": tone(9.0), "transcript": phrase.PHRASE, "scale": 1.0}
    monkeypatch.setattr(pipeline.audio, "load", lambda path: vals["x"])
    monkeypatch.setattr(pipeline.prosody, "extract_full_signal", lambda x: None)
    monkeypatch.setattr(pipeline.prosody, "from_full", lambda full: {
        "speech_rate": 4.0 * vals["scale"], "f0_std": 0.2 * vals["scale"], "pause_ratio": 0.3,
        "energy": 0.5 * vals["scale"], "jitter": 0.02, "shimmer": 1.1})
    monkeypatch.setattr(pipeline.emotion, "run_signal", lambda x: (None, {"arousal": 0.9, "valence": 0.1}))
    monkeypatch.setattr(checkins_router, "transcribe_phrase", lambda path: vals["transcript"])
    return vals


def post(c, res, with_phrase: bool, slot="morning") -> dict:
    if c.tick:
        c.day += 1
    c.tick += 1
    c.queue.append(res)
    files = {"audio": ("memo.wav", b"x", "audio/wav")}
    if with_phrase:
        files["phrase"] = ("phrase.wav", b"x", "audio/wav")
    r = c.post("/api/checkins", data={"patient_id": 1, "slot": slot}, files=files)
    assert r.status_code == 200, r.text
    return r.json()


def stored(checkin_id: int) -> dict[str, float]:
    with Session(db.engine) as s:
        return {f.name: f.value for f in s.exec(select(Feature).where(Feature.checkin_id == checkin_id))}


def test_old_clients_without_phrase_keep_working(client):
    out = post(client, features(0, random.Random(0)), with_phrase=False)
    assert out["phrase"] is None
    assert not any(n.startswith("phrase_") for n in stored(out["checkin_id"]))


def test_phrase_features_are_stored_with_prefix(client, phrase_models):
    out = post(client, features(0, random.Random(0)), with_phrase=True)
    assert out["phrase"]["ok"] and out["phrase"]["wer"] == 0
    names = stored(out["checkin_id"])
    assert {"phrase_energy", "phrase_arousal", "phrase_valence", "phrase_syllable_rate"} <= set(names)
    assert names["phrase_arousal"] == 0.9
    assert "energy" in names and names["energy"] != names["phrase_energy"]  # free talk kept separate
    with Session(db.engine) as s:
        assert '"phrase"' in s.get(CheckIn, out["checkin_id"]).llm_json  # the check is kept with the check-in


def test_failed_read_skips_phrase_features_but_keeps_the_checkin(client, phrase_models):
    phrase_models["transcript"] = "I don't want to read this today."
    out = post(client, features(0, random.Random(0)), with_phrase=True)
    assert out["phrase"]["ok"] is False
    names = stored(out["checkin_id"])
    assert not any(n.startswith("phrase_") for n in names)
    assert "energy" in names  # the memo still counts


def test_phrase_transcript_never_reaches_the_text_branch(client, phrase_models):
    out = post(client, features(0, random.Random(0)), with_phrase=True)
    assert not any(f["source"] == "text" for f in out["features"])  # memo mock has no transcript
    with Session(db.engine) as s:
        assert s.get(CheckIn, out["checkin_id"]).transcript is None


def test_crisis_words_in_the_phrase_recording_still_flag(client, phrase_models):
    phrase_models["transcript"] = "I want to kill myself"
    out = post(client, features(0, random.Random(0)), with_phrase=True)
    assert out["crisis"] is True


def test_phrase_is_not_a_scored_feature():
    for branch in state.DIRECTION.values():
        assert not any(f.startswith("phrase_") for f in branch)


def _run(c, with_phrase: bool, vals: dict | None) -> list[dict]:
    """Baseline (6 check-ins) then a relapse; returns the state after each check-in."""
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c.day, c.tick = 0, 0
    c.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    rng, wild = random.Random(7), random.Random(99)
    out = []
    for i in range(14):
        if vals is not None:
            vals["scale"] = wild.uniform(0.2, 5.0)  # phrase values swing wildly from day to day
        out.append(post(c, features(0 if i < 8 else 1.5, rng), with_phrase=with_phrase)["state"])
    return out


def test_phrase_features_never_change_D(client, phrase_models):
    without = _run(client, False, None)
    with_ = _run(client, True, phrase_models)
    assert any(s["S"] is not None for s in without)
    for a, b in zip(without, with_):
        assert a["S"] == b["S"]
        assert a["S_anchored"] == b["S_anchored"] and a["S_rolling"] == b["S_rolling"]
        assert a["branches"] == b["branches"]
        assert (a["explanation"] or {}).get("by_feature") == (b["explanation"] or {}).get("by_feature")
    # ...while the phrase still gets its own baseline values for later (#27)
    stats = client.get("/api/patients/1/baseline").json()["stats"]["all"]
    assert "phrase_energy" in stats
