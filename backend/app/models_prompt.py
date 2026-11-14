"""Daily prompt (issue #22): once a day at a random time between 09:00 and 21:00 local time.

New tables only (created by db.init_db). The two nullable CheckIn columns it needs, `prompt_at` and
`local_hour`, live on CheckIn in models.py and are added to an old app.db by db.add_missing_columns().
"""

from datetime import date as Date
from datetime import datetime

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class PromptSettings(SQLModel, table=True):
    """Per-person settings for the daily prompt (issue #22). Times are local "HH:MM" in `tz`."""

    patient_id: int = Field(foreign_key="patient.id", primary_key=True)
    tz: str = "Europe/Helsinki"
    window_start: str = "09:00"
    window_end: str = "21:00"
    quiet_start: str | None = None  # e.g. "12:00"; no prompt between quiet_start and quiet_end
    quiet_end: str | None = None
    enabled: bool = True
    paused_on: Date | None = None  # local date the person paused; the pause covers that day only


class DailyPrompt(SQLModel, table=True):
    """Once a day at a random time (experience sampling): "Time for your check-in"."""

    __table_args__ = (UniqueConstraint("patient_id", "date"),)

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: Date  # local date in the person's tz
    at: datetime  # when it fires, UTC
    answered_at: datetime | None = None  # UTC; a late answer still counts fully, just no on-time bonus
    checkin_id: int | None = Field(default=None, foreign_key="checkin.id")
    source: str = "random"  # "random" | "demo" ("Send today's prompt now")
    pushed_at: datetime | None = None  # Web Push sent (P2, app/push.py); once per prompt
