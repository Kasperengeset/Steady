"""Gemini: verbatim transcript of a spoken check-in (STATUS.md T3b).

The transcript goes to the text branch (pipeline/text.py) and the crisis phrase check, via
pipeline/__init__.py. Content scores (mood, energy, anhedonia, social contact) are not
implemented yet, so analyze() returns no scores.

- Model `gemini-3.5-transcribe`, verbatim mode (the default): keeps "um", repetitions and false
  starts, which smart mode would remove. Language pinned to English (settings.transcribe_language).
- Privacy: audio is sent inline (a Files API upload would keep a copy for 48 hours), store=False,
  and the caller deletes the raw audio right after. Never call this on Androids Corpus audio
  (licence, STATUS.md B2) – _refuse_corpus() blocks paths under data/androids.
- Without GEMINI_API_KEY in .env, or if the call fails, it falls back to the stub (stub=True), so
  the app keeps working and nothing invented reaches the score.
"""

import base64
import json
import logging
import mimetypes
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

AUDIO_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/m4a", ".aac": "audio/aac",
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".flac": "audio/flac", ".webm": "audio/webm",
    ".aiff": "audio/aiff",
}


@dataclass
class ContentAnalysis:
    transcript: str
    scores: dict[str, float]
    risk: bool
    raw_json: str
    stub: bool = False  # True for the placeholder – its numbers must not reach the score


@lru_cache(maxsize=1)
def _client():
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key, http_options={"api_version": "v1beta"})


def _refuse_corpus(audio_path: str) -> None:
    if "androids" in (p.lower() for p in Path(audio_path).resolve().parts):
        raise ValueError("refusing to send Androids Corpus audio to an external API (licence, STATUS.md B2)")


RETRY_DELAYS_S = (1.0, 3.0)  # 503 "model overloaded" is common on new models; 429 (spend cap) is not retried


def _overloaded(e: Exception) -> bool:
    msg = str(e).lower()
    return "503" in msg or "unavailable" in msg or "overloaded" in msg


def transcribe(audio_path: str) -> str:
    """Verbatim English transcript of a short recording (inline audio, store=False)."""
    _refuse_corpus(audio_path)
    path = Path(audio_path)
    mime = AUDIO_MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "audio/wav"
    audio = {"type": "audio", "data": base64.b64encode(path.read_bytes()).decode("ascii"), "mime_type": mime}
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        try:
            r = _client().interactions.create(
                model=settings.transcribe_model,
                input=[audio],
                store=False,
                generation_config={"transcription_config": {
                    "language_codes": [settings.transcribe_language],
                    "mode": {"type": "verbatim"},
                }},
            )
            return (r.output_text or "").strip()
        except Exception as e:
            if attempt == len(RETRY_DELAYS_S) or not _overloaded(e):
                raise
            log.info("transcription: model overloaded, retry in %.0f s", RETRY_DELAYS_S[attempt])
            time.sleep(RETRY_DELAYS_S[attempt])
    raise RuntimeError("unreachable")


def _stub() -> ContentAnalysis:
    payload = {"transcript": "(stub) I slept okay last night.", "risk": False}
    return ContentAnalysis(payload["transcript"], {}, False, json.dumps(payload), stub=True)


def analyze(audio_path: str) -> ContentAnalysis:
    if not settings.gemini_api_key:
        return _stub()
    try:
        transcript = transcribe(audio_path)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 – network or API error: keep the check-in, skip the text branch
        log.warning("transcription failed (%s): %s", type(e).__name__, str(e)[:200])
        return _stub()
    payload = {"transcript": transcript, "model": settings.transcribe_model}
    return ContentAnalysis(transcript, {}, False, json.dumps(payload))
