"""Feature extraction pipeline. Each extractor returns {name: value}.

Contract: extract_features(audio_path) -> PipelineResult. Raw audio is deleted by the
caller right after this returns – extractors must not keep references to it.
extract_text_features(text) -> PipelineResult for a written diary entry.
"""

from dataclasses import dataclass, field

from ..models import FeatureSource
from . import audio, emotion, llm, prosody, text


@dataclass
class PipelineResult:
    features: list[tuple[str, float, FeatureSource]] = field(default_factory=list)
    transcript: str | None = None
    llm_json: str | None = None
    crisis: bool = False


def _add_text(res: PipelineResult, txt: str) -> None:
    """Text branch: TF-IDF model + lexicon features + deterministic crisis phrases."""
    out = text.analyze_text(txt)
    for name, v in out["features"].items():
        res.features.append((name, v, FeatureSource.text))
    res.crisis = res.crisis or out["crisis"]


def extract_features(audio_path: str) -> PipelineResult:
    res = PipelineResult()
    x = audio.load(audio_path)  # decode once, share between extractors
    for name, v in prosody.from_full(prosody.extract_full_signal(x)).items():
        res.features.append((name, v, FeatureSource.prosody))
    _, emo = emotion.run_signal(x)
    for name in ("arousal", "valence"):
        res.features.append((name, emo[name], FeatureSource.emotion))
    content = llm.analyze(audio_path)
    res.transcript = content.transcript
    res.llm_json = content.raw_json
    res.crisis = content.risk
    # STATUS.md T2: the stub's random numbers must not reach the score. Only a real
    # transcript feeds the content scores and the text branch.
    if not content.stub:
        for name, v in content.scores.items():
            res.features.append((name, v, FeatureSource.llm))
        if content.transcript:
            _add_text(res, content.transcript)
    return res


def extract_text_features(diary: str) -> PipelineResult:
    res = PipelineResult(transcript=diary)
    _add_text(res, diary)
    return res
