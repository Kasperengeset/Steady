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


# --------------------------------------------------------------------------- content scores (#28)
RATE_FIELDS = ("mood", "energy_self", "anhedonia", "social_contact")  # names the engine already knows

RATE_SCHEMA = {
    "type": "object",
    "properties": {
        "about_self_now": {"type": "boolean"},
        "language": {"type": "string", "enum": ["en", "no", "other"]},
        **{f: {"type": "integer", "minimum": 0, "maximum": 10} for f in RATE_FIELDS},
        "evidence": {
            "type": "object",
            "properties": {f: {"type": "string"} for f in RATE_FIELDS},
            "required": list(RATE_FIELDS),
        },
    },
    "required": ["about_self_now", "language", *RATE_FIELDS, "evidence"],
}

RATE_INSTRUCTION = """You rate a short diary entry or spoken check-in from a person in remission after depression treatment.
Rate ONLY the writer's own state today (now). Do not count:
- past episodes ("a few months ago I felt hopeless") unless the writer says it is still so,
- other people's feelings (a friend, a sister, a character in a book or film),
- negated states ("I am not sad" means not sad), hypotheticals ("if I had stayed I would feel..."), quotes.
Read sarcasm for what is meant ("oh great, another fantastic day on the sofa" is a low day).
Flat or masked wording ("everything is fine, I did not feel anything") can be a low day: rate what is described.
Scales, integers 0 to 10:
- mood: 0 = very low, sad or hopeless; 10 = very good
- energy_self: 0 = exhausted, cannot get going; 10 = full of energy
- anhedonia: 0 = enjoys and wants to do things; 10 = no interest or pleasure in anything
- social_contact: 0 = avoided or had no contact with people; 10 = warm, active contact
evidence: for each score, a short quote (at most 12 words) from the text that supports it, or "" if none.
about_self_now: false if the text says almost nothing about the writer's own current state.
language: "en", "no" (Norwegian) or "other". The text may be English or Norwegian; rate both the same way."""


def validate_rating(raw: str | None) -> dict | None:
    """Strict check of Gemini's reply. Invalid JSON, a missing field or an out-of-range value → None
    (no scores, never a guessed neutral 5). about_self_now = false → None as well (#28)."""
    try:
        d = json.loads(raw or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("about_self_now") is not True:
        return None
    scores = {}
    for f in RATE_FIELDS:
        v = d.get(f)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 10:
            return None
        scores[f] = float(v)
    ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    return {"scores": scores, "evidence": {f: str(ev.get(f, ""))[:200] for f in RATE_FIELDS},
            "language": d.get("language") if d.get("language") in ("en", "no", "other") else "other"}


def rate_text(text: str) -> dict | None:
    """Context-aware content scores of the writer's own current state (#28), or None.

    gemini-4-hydrogen (settings.rate_model), temperature 0, strict JSON schema, store=False. No key, an
    error or an invalid reply → None. Never pass Reddit or Androids text here (B2); the text and the
    reply are never logged.
    """
    if not settings.gemini_api_key or not text or not text.strip():
        return None
    for attempt in range(len(RETRY_DELAYS_S) + 1):
        try:
            r = _client().interactions.create(
                model=settings.rate_model,
                input=text,
                system_instruction=RATE_INSTRUCTION,
                response_format=RATE_SCHEMA,
                response_mime_type="application/json",
                generation_config={"temperature": 0.0},
                store=False,
            )
            return validate_rating(r.output_text)
        except Exception as e:  # noqa: BLE001 – network/API error: no scores
            if attempt < len(RETRY_DELAYS_S) and _overloaded(e):
                time.sleep(RETRY_DELAYS_S[attempt])
                continue
            log.warning("rate_text failed (%s)", type(e).__name__)
            return None
    return None
