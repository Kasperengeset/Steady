"""ml/voice/phrase_check.py: file naming, quality filter and the pre-registered gate (B29)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "voice"))
import phrase_check as pc  # noqa: E402


def _table(sd_phrase: float, sd_free: float, people=("kasper", "mahdan"), sessions=5, seed=0) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    rows = []
    for p in people:
        for i in range(sessions):
            hour = 9 + 3 * i
            for kind, sd, dur in (("phrase", sd_phrase, 9.0), ("free", sd_free, 30.0)):
                rows.append({"person": p, "session": f"20261114-{hour:02d}00", "kind": kind, "hour": hour,
                             "duration_s": dur, **{f: 1.0 + sd * rng.standard_normal() for f in pc.FEATS}})
    return pd.DataFrame(rows)


def test_file_name_convention():
    assert pc.parse_name(Path("kasper_20261114-0930_phrase.m4a")) == {
        "person": "kasper", "session": "20261114-0930", "kind": "phrase", "hour": 9.5}
    assert pc.parse_name(Path("mahdan_20261114-2100_free.wav"))["kind"] == "free"
    assert pc.parse_name(Path("Kasper_bad1.wav")) is None
    assert pc.parse_name(Path("kasper_20261114-0930_memo.wav")) is None


def test_refuses_corpus_paths():
    with pytest.raises(SystemExit):
        pc._refuse_corpus(Path("/tmp/data/androids/Reading-Task"))


def test_clearly_steadier_phrase_passes():
    res = pc.analyse(_table(0.1, 1.0, sessions=6))
    assert res["enough_data"]
    assert {f for f, r in res["features"].items() if r["passes"]} == pc.CANDIDATES
    assert all(r["sd_ratio"] < 0.3 for r in res["features"].values())


def test_equal_noise_does_not_pass():
    res = pc.analyse(_table(1.0, 1.0, seed=3))
    assert not any(r["passes"] for r in res["features"].values())


def test_one_person_is_not_enough():
    res = pc.analyse(_table(0.1, 1.0, people=("kasper",), sessions=8))
    assert not res["enough_data"]
    assert not any(r["passes"] for r in res["features"].values())


def test_quality_drops_bad_lengths_and_incomplete_sessions(tmp_path):
    df = _table(0.5, 1.0)
    df.loc[0, "duration_s"] = 1.0  # a phrase cut off after 1 s
    df = df.drop(index=3)  # a session without its free-talk file
    kept, notes = pc.quality(df)
    assert len(notes) == 3  # the short phrase, then its session and the one without free talk are incomplete
    assert kept.groupby(["person", "session"])["kind"].nunique().eq(2).all()
    text = pc.write_report(pc.analyse(df), tmp_path / "r.md")
    assert "Gate (B29" in text and "per-session values stay" in text
