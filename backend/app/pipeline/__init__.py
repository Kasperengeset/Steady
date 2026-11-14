"""Feature extraction pipeline. Each extractor returns {name: value}.

Contract: extract_features(audio_path) -> PipelineResult. Raw audio is deleted by the
caller right after this returns – extractors must not keep references to it.
extract_text_features(text) -> PipelineResult for a written diary entry.
add_phrase(res, phrase_path, transcript) adds the fixed reading phrase (issue #21) to a spoken check-in.
"""

import json
import logging
from dataclasses import dataclass, field

from ..config import settings
from ..models import FeatureSource
from . import audio, emotion, health, llm, phrase, prosody, text

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    features: list[tuple[str, float, FeatureSource]] = field(default_factory=list)
    transcript: str | None = None
    llm_json: str | None = None
    crisis: bool = False
    speech_s: float | None = None  # seconds with voice energy in the free-talk memo (None for text)
    phrase_check: dict | None = None  # reading check of the fixed phrase (pipeline/phrase.py)


def _add_text(res: PipelineResult, txt: str) -> None:
    """Text branch: TF-IDF model + lexicon features + deterministic crisis phrases, plus Gemini content
    scores of the writer's own current state (#28; weight 0 in text.json, STATUS.md B33). Only the free
    talk or diary reaches here, never the fixed phrase. No key or an invalid reply → no content scores."""
    out = text.analyze_text(txt)
    for name, v in out["features"].items():
        res.features.append((name, v, FeatureSource.text))
    res.crisis = res.crisis or out["crisis"]
    rating = llm.rate_text(txt) if settings.rate_text_enabled else None
    if rating:
        for name, v in rating["scores"].items():
            res.features.append((name, v, FeatureSource.llm))
        payload = json.loads(res.llm_json) if res.llm_json else {}
        payload["content"] = {"evidence": rating["evidence"], "language": rating["language"]}  # clinician view
        res.llm_json = json.dumps(payload)


def extract_features(audio_path: str) -> PipelineResult:
    res = PipelineResult()
    x = audio.load(audio_path)  # decode once, share between extractors
    res.speech_s = round(audio.active_seconds(x), 2)
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


def transcribe_phrase(phrase_path: str) -> str | None:
    """Transcript of the phrase recording for the reading check; None without a key or on failure."""
    content = llm.analyze(phrase_path)
    return None if content.stub else content.transcript


def add_phrase(res: PipelineResult, phrase_path: str, transcript: str | None) -> None:
    """Fixed phrase (issue #21): reading check, then `phrase_` acoustic features if the read passed.

    Never raises – a bad phrase recording must not cost the person their check-in. The phrase transcript
    never reaches the text branch; only the deterministic crisis phrase list runs on it.
    """
    try:
        x = audio.load(phrase_path)
    except ValueError as e:
        check = {"phrase_id": phrase.PHRASE_ID, "ok": False, "reason": str(e)}
    else:
        span = audio.speech_span(x)
        check = phrase.reading_check(span, transcript)
        if check["ok"]:
            try:
                feats = prosody.from_full(prosody.extract_full_signal(x))
                _, emo = emotion.run_signal(x)
            except Exception as e:  # noqa: BLE001 – keep the memo even if the phrase fails to analyse
                log.warning("phrase features failed (%s): %s", type(e).__name__, str(e)[:200])
                check.update(ok=False, reason="analysis failed")
            else:
                for name, v in feats.items():
                    res.features.append((f"phrase_{name}", v, FeatureSource.prosody))
                for name in ("arousal", "valence"):
                    res.features.append((f"phrase_{name}", emo[name], FeatureSource.emotion))
                res.features.append(("phrase_syllable_rate", check["syllables_per_s"], FeatureSource.prosody))
    if transcript and text.crisis_hits(transcript):
        res.crisis = True
    res.phrase_check = check
    payload = json.loads(res.llm_json) if res.llm_json else {}
    payload["phrase"] = check
    res.llm_json = json.dumps(payload)


def extract_text_features(diary: str) -> PipelineResult:
    res = PipelineResult(transcript=diary)
    _add_text(res, diary)
    return res


def extract_wearable_features(**day) -> PipelineResult:
    """Health measurements, using the same extractor as training and Google Health replay."""
    res = PipelineResult()
    for name, value in health.wearable_features(**day).items():
        res.features.append((name, value, FeatureSource.wearable))
    return res
