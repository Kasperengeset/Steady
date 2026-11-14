"""Gemini: transcript + content scores + crisis flag as JSON. STUB."""

import json
import random
from dataclasses import dataclass


@dataclass
class ContentAnalysis:
    transcript: str
    scores: dict[str, float]
    risk: bool
    raw_json: str
    stub: bool = False  # True while analyze() is a placeholder – its numbers must not reach the score


def analyze(audio_path: str) -> ContentAnalysis:
    # TODO: google-genai, Vertex AI EU region, structured JSON output
    scores = {
        "mood": float(random.randint(4, 8)),  # 0–10
        "energy_self": float(random.randint(3, 8)),
        "anhedonia": float(random.randint(0, 4)),
        "social_contact": float(random.randint(0, 5)),
    }
    payload = {"transcript": "(stub) I slept okay last night.", **scores, "risk": False}
    return ContentAnalysis(payload["transcript"], scores, False, json.dumps(payload), stub=True)
