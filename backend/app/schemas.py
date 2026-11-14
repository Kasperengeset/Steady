"""API contract – mirrored in frontend/src/lib/api.ts. Change both together."""

from datetime import date as Date

from pydantic import BaseModel

from .models import Slot


class FeatureOut(BaseModel):
    name: str
    value: float
    source: str


class StateOut(BaseModel):
    S: float | None
    level: int
    branches: dict[str, float] = {}  # per-branch D_b today


class InterventionOut(BaseModel):
    arm_key: str
    label: str


class CheckInResponse(BaseModel):
    checkin_id: int
    features: list[FeatureOut]
    state: StateOut
    intervention: InterventionOut
    crisis: bool


class TextCheckInIn(BaseModel):
    """Written diary entry (text branch)."""

    patient_id: int
    slot: Slot
    text: str


class FeedbackIn(BaseModel):
    patient_id: int
    date: Date
    completed: bool
    rating: int | None = None


class FeedbackOut(BaseModel):
    reward: float


class PatientSummary(BaseModel):
    id: int
    name: str
    S: float | None
    level: int
    trend: list[float]


class PatientDetail(BaseModel):
    patient: dict
    scores: list[dict]
    checkins: list[dict]
    top_features: list[dict]


class FriendHint(BaseModel):
    show_hint: bool
    message: str


class SimIn(BaseModel):
    days: int = 70
    seed: int = 0
    relapse_day: int = 40
