"""Feature extraction pipeline. Each extractor returns {name: value}.

Contract: extract_features(audio_path) -> PipelineResult. Raw audio is deleted by the
caller right after this returns – extractors must not keep references to it.
"""

from dataclasses import dataclass, field

from ..models import FeatureSource
from . import emotion, llm, prosody


@dataclass
class PipelineResult:
    features: list[tuple[str, float, FeatureSource]] = field(default_factory=list)
    transcript: str | None = None
    llm_json: str | None = None
    crisis: bool = False


def extract_features(audio_path: str) -> PipelineResult:
    res = PipelineResult()
    for name, v in prosody.extract(audio_path).items():
        res.features.append((name, v, FeatureSource.prosody))
    for name, v in emotion.extract(audio_path).items():
        res.features.append((name, v, FeatureSource.emotion))
    content = llm.analyze(audio_path)
    for name, v in content.scores.items():
        res.features.append((name, v, FeatureSource.llm))
    res.transcript = content.transcript
    res.llm_json = content.raw_json
    res.crisis = content.risk
    return res
