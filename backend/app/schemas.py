"""API contract – mirrored in frontend/src/lib/api.ts. Change both together."""

from datetime import date as Date
from typing import Annotated

from pydantic import BaseModel, Field

from .models import Slot
from .schemas_prompt import PromptAnswerOut


class FeatureOut(BaseModel):
    name: str
    value: float
    source: str


class StateOut(BaseModel):
    S: float | None  # combined = max(S_anchored, S_rolling); None while the healthy baseline is building
    level: int
    branches: dict[str, float] = {}  # per-branch D_b today (driver track)
    S_anchored: float | None = None  # vs the fixed healthy baseline – catches slow drift
    S_rolling: float | None = None  # vs the last 14 days (same slot) – catches abrupt change
    driver: str | None = None  # which track set S
    elevated: bool = False  # unusually elevated vs the healthy normal – clinician only (B21)
    explanation: dict | None = None  # exact split of S over days and features + a sentence (B20)


class BaselineOut(BaseModel):
    version: int
    status: str  # building | active | retired | rejected
    started_at: str
    locked_at: str | None
    period_days: int
    days_elapsed: float
    checkins: dict[str, int]  # per slot since the baseline started
    min_checkins: int
    confirmed_by: str | None


class BaselineStartIn(BaseModel):
    period_days: int | None = None  # default settings.healthy_period_days (demo: 0)
    min_checkins: int | None = None
    note: str | None = None


class BaselineConfirmIn(BaseModel):
    clinician: str
    note: str | None = None


class BaselineDetail(BaseModel):
    active: BaselineOut | None
    building: BaselineOut | None
    stats: dict[str, dict[str, dict[str, float]]]  # slot -> feature -> {center, scale, n}


class InterventionOut(BaseModel):
    arm_key: str
    label: str


class CheckInResponse(BaseModel):
    checkin_id: int
    features: list[FeatureOut]
    state: StateOut
    baseline: BaselineOut  # the baseline used for scoring (or the building one if none is active)
    baseline_building: BaselineOut | None = None  # a new baseline being built (while the old one still scores)
    intervention: InterventionOut
    crisis: bool
    prompt: PromptAnswerOut | None = None  # this check-in answered today's daily prompt (#22)
    phrase: dict | None = None  # reading check of the fixed phrase, if one was uploaded (#21)
    rewards: list[dict] = Field(default_factory=list)
    analysis: dict | None = None


class TextCheckInIn(BaseModel):
    """Written diary entry (text branch)."""

    patient_id: int
    slot: Slot
    text: str = Field(max_length=10000)


class WearableDayIn(BaseModel):
    """Completed wearable day. Missing measurements are omitted; exercise is engagement-only."""

    patient_id: int
    slot: Slot = Slot.morning
    minute_steps: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] | None = None
    steps: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sleep_hours: float | None = Field(default=None, ge=0, le=24, allow_inf_nan=False)
    sleep_midpoint: float | None = Field(default=None, ge=-24, le=24, allow_inf_nan=False)
    resting_hr: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    hrv_rmssd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    active_minutes: float | None = Field(default=None, ge=0, le=1440, allow_inf_nan=False)


class FeedbackIn(BaseModel):
    patient_id: int
    date: Date
    completed: bool
    rating: int | None = Field(default=None, ge=1, le=5)


class FeedbackOut(BaseModel):
    reward: float
    rewards: list[dict] = Field(default_factory=list)


class PatientSummary(BaseModel):
    id: int
    name: str
    S: float | None
    level: int
    trend: list[float]
    driver: str | None = None
    baseline_status: str | None = None  # building | active | None
    elevated: bool = False


class PatientDetail(BaseModel):
    patient: dict
    scores: list[dict]
    checkins: list[dict]
    top_features: list[dict]
    explanation: dict | None = None


class FriendHint(BaseModel):
    show_hint: bool
    message: str


class SimIn(BaseModel):
    days: int = 70
    seed: int = 0
    relapse_day: int = 40
