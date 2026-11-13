"""Data model. Features are stored in long format so new signals need no migration."""

from datetime import date as Date
from datetime import datetime, timezone
from enum import Enum

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
    wearable = "wearable"


class Patient(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=now)
    risk_level: int = 0  # current escalation level 0/1/2, 3 = crisis


class Contact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    name: str
    kind: ContactKind


class CheckIn(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    ts: datetime = Field(default_factory=now)
    slot: Slot
    transcript: str | None = None
    llm_json: str | None = None
    risk_flag: bool = False


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
    D: float  # weighted daily deviation
    S: float  # EWMA-smoothed state
    level: int = 0
    top_features_json: str = "[]"


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
