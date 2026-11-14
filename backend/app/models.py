"""Data model. Features are stored in long format so new signals need no migration."""

from datetime import date as Date
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def now() -> datetime:
    return datetime.now(timezone.utc)


class Slot(str, Enum):
    morning = "morning"
    evening = "evening"


class ContactKind(str, Enum):
    friend = "friend"
    clinician = "clinician"


class FeatureSource(str, Enum):
    prosody = "prosody"
    emotion = "emotion"
    llm = "llm"
    text = "text"
    wearable = "wearable"


class Patient(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=now)
    risk_level: int = 0  # current escalation level 0/1/2, 3 = crisis
    demo_key: str | None = Field(default=None, index=True, unique=True)


class Contact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    name: str
    kind: ContactKind


class CheckIn(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("patient_id", "request_key", name="uq_checkin_request"),)

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    ts: datetime = Field(default_factory=now)
    slot: Slot
    transcript: str | None = None
    llm_json: str | None = None
    risk_flag: bool = False
    prompt_at: datetime | None = None  # UTC time of the daily prompt this check-in answered (#22), else None
    local_hour: float | None = None  # actual time of the check-in as a local decimal hour, e.g. 14.5 (#22, #27)
    speech_s: float | None = None  # seconds with voice energy in the memo (None for written diaries)
    request_key: str | None = None
    response_json: str | None = None  # replay the original response after a network retry
    analysis_json: str | None = None  # immutable measurements and decisions captured at ingestion
    data_origin: str | None = None  # live | synthetic; legacy unmarked rows remain unknown


class Feature(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    checkin_id: int = Field(foreign_key="checkin.id", index=True)
    name: str
    value: float
    source: FeatureSource


class StateScore(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: Date
    D: float  # combined daily deviation = the track that drives S
    S: float  # combined EWMA state = max(S_anchored, S_rolling) – used for escalation
    level: int = 0
    # Two tracks (STATUS.md T1): vs the fixed healthy baseline (slow drift) and vs the last 14 days (abrupt change)
    D_anchored: float | None = None
    S_anchored: float | None = None
    D_rolling: float | None = None
    S_rolling: float | None = None
    driver: str | None = None  # "anchored" | "rolling" – which track set S
    baseline_version: int | None = None
    top_features_json: str = "[]"
    branch_scores_json: str = "{}"  # {"anchored": {"acoustic": [D per check-in], ...}, "rolling": {...}}
    evidence_json: str = "{}"  # {"anchored": {"acoustic": {feature: Σ share over today's check-ins}}, ...} (B20)
    elevated: bool = False  # unusually elevated vs healthy normal – clinician only (B21)


class BaselineStatus(str, Enum):
    building = "building"  # collecting check-ins during the healthy period
    active = "active"  # locked, used for scoring
    retired = "retired"  # replaced by a newer locked baseline
    rejected = "rejected"  # clinician said the period was not representative


class Baseline(SQLModel, table=True):
    """Fixed "healthy" reference, built from the first weeks after TMS. Never updated automatically."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    version: int
    status: BaselineStatus = BaselineStatus.building
    started_at: datetime = Field(default_factory=now)
    locked_at: datetime | None = None
    period_days: int
    min_checkins: int
    confirmed_by: str | None = None  # clinician who confirmed remission (optional)
    note: str | None = None


class BaselineStat(SQLModel, table=True):
    """Robust centre/scale per feature and slot ("morning" | "evening" | "all")."""

    id: int | None = Field(default=None, primary_key=True)
    baseline_id: int = Field(foreign_key="baseline.id", index=True)
    slot: str
    feature: str
    center: float  # median
    scale: float  # MAD-based SD shrunk toward the corpus within-person SD
    n: int


class Arm(SQLModel, table=True):
    """Beta posterior for one intervention (Thompson sampling)."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    key: str
    label: str
    a: float = 1.0
    b: float = 1.0


class InterventionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: Date
    arm_key: str
    completed: bool | None = None
    rating: int | None = None  # 1–5
    reward: float | None = None


class Escalation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    ts: datetime = Field(default_factory=now)
    level: int
    recipient: ContactKind
    reason: str
    handled: bool = False
    handled_at: datetime | None = None
