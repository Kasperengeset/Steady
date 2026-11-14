"""Web Push subscriptions (issue #22, P2). New table only (created by db.init_db)."""

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PushSubscription(SQLModel, table=True):
    """One browser/device that asked for the daily prompt. The endpoint is a secret capability URL."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    endpoint: str = Field(unique=True)
    p256dh: str
    auth: str
    created_at: datetime = Field(default_factory=_now)
