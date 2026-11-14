"""Rewards (#24). Kept apart from models.py so the engagement work does not collide with the engine.

Rules (app/rewards.py): reward doing, never the score. Nothing earned is ever taken back. A spend (the
avatar shop, #23) is a negative row, so XP = total ever earned never goes down.
"""

from datetime import date as Date
from datetime import datetime

from sqlmodel import Field, SQLModel

from .models import now


class RewardLedger(SQLModel, table=True):
    """One row per reward or spend. Balance = Σ amount; XP = Σ positive amounts."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    ts: datetime = Field(default_factory=now)
    date: Date = Field(index=True)  # the day the reward belongs to (the patient's local day)
    reason: str  # memo | memo_on_time | action_done | followup | streak_7 | spend:<what>
    amount: int
    key: str | None = Field(default=None, index=True)  # e.g. "memo:2026-11-14": each thing pays once
