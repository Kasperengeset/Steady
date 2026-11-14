"""Appie presentation and custom habits. The canonical reward ledger owns every food balance."""

from datetime import date as Date, datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class AppiePet(SQLModel, table=True):
    patient_id: int = Field(foreign_key="patient.id", primary_key=True)
    name: str = "Appie"
    fed_xp: int = 0  # Treats consumed, for animation/history only. Lifetime XP lives in rewards.py.
    feed_count: int = 0
    last_fed_at: datetime | None = None
    last_level: int = 1
    leveled_on: Date | None = None
    legacy_migrated: bool = False


class AppieHabit(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("patient_id", "key"), UniqueConstraint("patient_id", "normalized_label"))
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    key: str
    label: str
    normalized_label: str
    active: bool = True


class AppieHabitLog(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("patient_id", "date", "habit_key"),)
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: Date = Field(index=True)
    habit_key: str
    source: str = "manual"


class AppieFeed(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("patient_id", "request_id"),)
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    request_id: str
    response_json: str
