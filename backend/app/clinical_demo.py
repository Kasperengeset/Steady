"""Append-only clinical demonstrations scored by the application's actual engine.

The diaries are synthetic and run through the real local text model. Acoustic
measurements are simulated from the installed corpus statistics; wearable days
are generated and passed through the real health extractor. No recording is
created or analysed, no external model is called, and no existing patient is reset.
"""

import json
import math
import random
import threading
from datetime import datetime, time, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, col, select

from . import baseline
from .db import get_session
from .engine import state
from .models import Arm, Baseline, BaselineStatus, CheckIn, Contact, ContactKind, FeatureSource, Patient, Slot, StateScore
from .pipeline import PipelineResult, extract_wearable_features
from .pipeline import text as text_model
from .seed import ARMS

router = APIRouter(tags=["clinical demo"])
_seed_lock = threading.Lock()
VERSION = "clinical-demo-v1"
SCENARIOS = {"drift": ("Alex · drift demo", 24), "stable": ("Mia · stable demo", 24), "building": ("Jonas · baseline demo", 7)}
DISCLOSURE = (
    "Synthetic demonstration, not a real patient or a validated relapse test. "
    "Diaries are scored by the installed local text model. Voice measurements are simulated "
    "from corpus statistics; wearable measurements are generated. The ordinary state engine "
    "scores all observations. New entries you submit are labelled live and remain separate."
)
BASELINE_NOTE = (
    "SYNTHETIC DEMO REFERENCE. Fourteen generated healthy days, including simulated acoustic and "
    "wearable measurements and synthetic diaries analysed by the real local text model. "
    "This is not a clinician-confirmed remission baseline. Live entries are compared with this "
    "demonstration reference only; no inference about clinical performance is justified."
)

GOOD = [
    "I woke at my usual time and made breakfast before going outside. I enjoyed a walk and noticed the leaves changing along the path. Later I read a few pages and called a friend. I felt interested in our conversation and glad that I made time for it. I still had enough energy to cook dinner and relax in the evening.",
    "I slept comfortably and got up when my alarm rang. I took a short walk after breakfast and spent some time on my hobby. I enjoyed making something with my hands. I answered a friend's message and we made a simple plan for the weekend. I felt calm today, and ordinary tasks seemed manageable even when I needed a break.",
    "I had a quiet morning and enjoyed breakfast by the window. I went outside for some fresh air and listened to music while walking. I made lunch, tidied my room and read a chapter of my book. I felt connected when I talked to a friend later. I was a little tired in the evening, but I enjoyed how I spent the day.",
    "I got up at my usual time and felt ready for a gentle day. I ate breakfast, went for a walk and worked on a small hobby project. I enjoyed the music I played in the afternoon. I called a friend and laughed about something ordinary. I felt content with what I managed, and I am looking forward to tomorrow's small plans.",
    "I had enough sleep and made time for a slow breakfast. I went outside, bought a few groceries and cooked something I liked. I spent the afternoon reading and checking in with a friend. I enjoyed the conversation and felt comfortable in my own company afterwards. I had some worries, but I could put them aside and return to what I was doing.",
]
NEUTRAL = [
    "I woke a little later than usual and felt tired over breakfast. I still went outside for a short walk and answered a friend's message. I tried to read, although I had to go over a page twice. I cooked something simple and listened to music. I did less than I planned today, but I managed the ordinary things with a few extra pauses.",
    "I found it harder to get started this morning and spent longer sitting after breakfast. I went for a shorter walk and left my hobby project for another day. I replied to a friend but did not feel like a long conversation. I made dinner and watched something familiar. I felt flat in the evening and hoped I would have more energy tomorrow.",
]
LOW = [
    "I woke late and stayed in bed after my alarm. I felt tired before doing anything and making breakfast seemed difficult. I tried to read my book but could not enjoy it. I left a friend's message unanswered and did not go outside. I spent most of the afternoon sitting alone. I felt flat in the evening, and even small tasks seemed like too much effort.",
    "I slept badly and struggled to get up this morning. I had no energy for my usual walk and left my hobby untouched. I tried listening to music but it did not interest me. I avoided a call from a friend because conversation felt exhausting. I ate something simple and sat quietly for the rest of the evening. I felt low and disconnected from the things I usually enjoy.",
    "I stayed in bed longer than I intended and felt heavy when I finally got up. I did not feel interested in breakfast or my book. I kept putting off a short walk and never went. I saw a message from a friend but could not find the energy to answer it. I felt alone and worried that ordinary tasks were becoming harder to manage.",
    "I woke tired and found it difficult to begin the day. I left the curtains closed for a while and skipped my usual walk. I tried working on my hobby but nothing about it felt enjoyable. I did not want to talk when a friend called. I spent the evening sitting on my own, feeling flat and discouraged about how little I had managed.",
    "I slept at odd times and struggled to get out of bed. I felt exhausted and could not settle into a book or enjoy my usual music. I left messages unopened and avoided going outside. I kept thinking that everything was too difficult, even cooking a small meal. I felt low throughout the evening and wished I could find some interest in the things that normally help me.",
]


class DemoIn(BaseModel):
    scenario: Literal["drift", "stable", "building"] = "drift"


def _metadata(modality: str) -> str:
    return json.dumps({"demo": {"version": VERSION, "data_origin": "synthetic", "modality": modality,
                                "text_analysis": "actual local TF-IDF and lexicon model" if modality == "text" else None,
                                "measurement": "simulated from corpus feature statistics; no recording analysed" if modality == "voice" else "generated wearable day" if modality == "wearable" else "synthetic first-person diary"}})


def _acoustic(index: int, severity: float, rng: random.Random) -> PipelineResult:
    features = []
    for name, stats in state.load_branch("acoustic").meta["feature_stats"].items():
        mean = stats["control_mean"] + severity * (stats["patient_mean"] - stats["control_mean"])
        value = mean + rng.gauss(0, stats["within_sd"] * .35)
        value = max(.001, value)
        if name in ("pause_ratio", "arousal", "valence"):
            value = min(.995, value)
        features.append((name, value, FeatureSource.emotion if name in ("arousal", "valence") else FeatureSource.prosody))
    return PipelineResult(features=features, llm_json=_metadata("voice"), speech_s=round(25 + math.sin(index), 2))


def _wearable(index: int, severity: float, rng: random.Random) -> tuple[PipelineResult, dict]:
    # A generated step timeline, never imported from a real person's wearable.
    minutes = [0.0] * 1440
    start = 8 * 60 + round(severity * 80)
    for minute in range(start, min(start + 12 * 60, 1440)):
        minutes[minute] = max(0, (11.5 - severity * 5.0) * (1 + .25 * math.sin(minute / 65)))
    walk_start = 10 * 60 + round(severity * 90)
    for minute in range(walk_start, min(walk_start + max(5, round(26 - severity * 12)), 1440)):
        minutes[minute] += max(15, 100 - severity * 45)
    scale = 1 + .05 * math.sin(index * 1.7)
    minutes = [round(value * scale, 2) for value in minutes]
    result = extract_wearable_features(
        minute_steps=minutes,
        sleep_hours=7.6 - severity * 1.15 + rng.gauss(0, .18),
        sleep_midpoint=3.4 + severity * 1.3 + rng.gauss(0, .15),
        resting_hr=61 + severity * 5.0 + rng.gauss(0, 1.3),
        hrv_rmssd=max(12, 49 - severity * 15 + rng.gauss(0, 2.0)),
    )
    result.llm_json = _metadata("wearable")
    return result, {"active_minutes": sum(value >= 100 for value in minutes)}


def _diary(index: int, severity: float) -> PipelineResult:
    diary = GOOD[index % len(GOOD)] if severity < .2 else NEUTRAL[index % len(NEUTRAL)] if severity < .65 else LOW[index % len(LOW)]
    # Calling the local analyzer directly deliberately bypasses optional cloud content rating.
    output = text_model.analyze_text(diary)
    return PipelineResult(transcript=diary, llm_json=_metadata("text"), crisis=output["crisis"],
                          features=[(name, float(value), FeatureSource.text) for name, value in output["features"].items()])


def _summary(s: Session, patient: Patient, scenario: str, created: bool) -> dict:
    entries = s.exec(select(CheckIn).where(CheckIn.patient_id == patient.id)).all()
    scores = s.exec(select(StateScore).where(StateScore.patient_id == patient.id).order_by(col(StateScore.date))).all()
    active = baseline.latest(s, patient.id, BaselineStatus.active)
    return {"patient_id": patient.id, "patient_name": patient.name, "demo_key": patient.demo_key,
            "scenario": scenario, "created": created, "entry_count": len(entries),
            "synthetic_entry_count": sum(entry.data_origin == "synthetic" for entry in entries),
            "live_entry_count": sum(entry.data_origin == "live" for entry in entries),
            "baseline_status": "active" if active else "building", "latest_score": scores[-1].S if scores else None,
            "model_branches": ["acoustic", "text", "health"], "disclosure": DISCLOSURE}


def seed(s: Session, scenario: str = "drift") -> dict:
    """Create or resume an isolated story. Repeating it never duplicates entries or resets people."""
    from .routers.checkins import _record_checkin

    if scenario not in SCENARIOS:
        raise ValueError("Unknown clinical demonstration.")
    with _seed_lock:
        key = f"{VERSION}-{scenario}"
        patient = s.exec(select(Patient).where(Patient.demo_key == key)).first()
        created = patient is None
        if patient is None:
            patient = Patient(name=SCENARIOS[scenario][0], demo_key=key, created_at=baseline.clock())
            s.add(patient)
            s.flush()
            for arm, label in ARMS:
                s.add(Arm(patient_id=patient.id, key=arm, label=label))
            s.add(Contact(patient_id=patient.id, name="Demo care team", kind=ContactKind.clinician))
            s.commit()
            s.refresh(patient)
        existing = s.exec(select(CheckIn).where(CheckIn.patient_id == patient.id)).all()
        expected = SCENARIOS[scenario][1] * 3
        if len([entry for entry in existing if entry.data_origin == "synthetic"]) >= expected:
            return _summary(s, patient, scenario, created)
        if any(entry.data_origin == "live" for entry in existing):
            out = _summary(s, patient, scenario, created)
            out["warning"] = "This demonstration already has live entries. Earlier generated history was left unchanged."
            return out
        created_at = patient.created_at if patient.created_at.tzinfo else patient.created_at.replace(tzinfo=timezone.utc)
        start = created_at.date() - timedelta(days=SCENARIOS[scenario][1])
        healthy = baseline.latest(s, patient.id, BaselineStatus.building) or baseline.latest(s, patient.id, BaselineStatus.active)
        if healthy is None:
            healthy = baseline.start(s, patient.id, period_days=14, min_checkins=14, now=datetime.combine(start, time(8), tzinfo=timezone.utc), note=BASELINE_NOTE)
            s.commit()
        have = {entry.request_key for entry in existing}
        rng = random.Random(key)
        for index in range(SCENARIOS[scenario][1]):
            day = start + timedelta(days=index)
            severity = max(0.0, (index - 13) / 10) * 1.7 if scenario == "drift" else 0.0
            # Freeze only the 14 healthy days before adding the first drift observation.
            if index == 14 and healthy.status == BaselineStatus.building:
                lock_at = datetime.combine(day, time(8), tzinfo=timezone.utc)
                if not baseline.try_lock(s, healthy, lock_at):
                    raise RuntimeError("The generated healthy reference did not contain enough model observations.")
                healthy.confirmed_by = "Synthetic demonstration reference, not clinical confirmation"
                healthy.note = BASELINE_NOTE
                s.add(healthy)
                s.commit()
            wearable, extras = _wearable(index, severity, rng)
            observations = [
                ("wearable", 8, 5, Slot.morning, wearable, extras),
                ("voice", 11, 0, Slot.morning, _acoustic(index, severity, rng), None),
                ("text", 18, 0, Slot.evening, _diary(index, severity), None),
            ]
            for modality, hour, minute, slot, result, extra in observations:
                request_key = f"{key}:{index}:{modality}"
                if request_key in have:
                    continue
                _record_checkin(s, patient, slot, result, extra, recorded_at=datetime.combine(day, time(hour, minute), tzinfo=timezone.utc), request_key=request_key, data_origin="synthetic")
                have.add(request_key)
        return _summary(s, patient, scenario, created)


@router.post("/clinician/demo")
def create_demo(body: DemoIn = DemoIn(), s: Session = Depends(get_session)) -> dict:
    return seed(s, body.scenario)
