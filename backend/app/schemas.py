"""API contract – mirrored in frontend/src/lib/api.ts. Change both together."""

from datetime import date as Date

from pydantic import BaseModel


class FeatureOut(BaseModel):
    name: str
    value: float
    source: str


class StateOut(BaseModel):
    S: float | None
    level: int


class InterventionOut(BaseModel):
    arm_key: str
    label: str


class CheckInResponse(BaseModel):
    checkin_id: int
    features: list[FeatureOut]
    state: StateOut
    intervention: InterventionOut
    crisis: bool


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
    days: int = 60
    seed: int = 0
