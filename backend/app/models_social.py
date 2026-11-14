"""The circle (#26) and the avatar's check-in messages (#25).

No health data lives here or leaves through the social routers: no scores, drift, diaries, voice,
transcripts, features or clinician notes. Sharing streak, level, avatar or "checked in today" is opt-in,
item by item, and the social part has its own consent (health data is special-category data, GDPR).
"""

from datetime import date as Date
from datetime import datetime
from enum import Enum

from sqlmodel import Field, SQLModel

from .models import now


class IdentityMode(str, Enum):
    profile = "profile"  # a name, and a photo if they want
    anonymous = "anonymous"  # a pseudonym and the avatar only; the clinician still knows who it is


class Profile(SQLModel, table=True):
    patient_id: int = Field(foreign_key="patient.id", primary_key=True)
    mode: IdentityMode = IdentityMode.anonymous
    display_name: str | None = None
    pseudonym: str | None = None
    photo_url: str | None = None
    emoji: str | None = None  # optional emoji status
    share_avatar: bool = False
    share_level: bool = False
    share_streak: bool = False
    share_checked_in: bool = False
    social_consent_at: datetime | None = None  # separate consent for the social part
    updated_at: datetime = Field(default_factory=now)


class Circle(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    owner_patient_id: int = Field(foreign_key="patient.id", index=True)
    name: str = "My circle"
    invite_code: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=now)


class MemberRole(str, Enum):
    patient = "patient"  # someone using the app (has an avatar)
    friend = "friend"  # a light friend account: a name and a token, no health features


class Membership(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    circle_id: int = Field(foreign_key="circle.id", index=True)
    role: MemberRole
    patient_id: int | None = Field(default=None, foreign_key="patient.id", index=True)
    friend_name: str | None = None
    token: str = Field(index=True, unique=True)  # the member's key to the feed (no accounts in the demo)
    joined_at: datetime = Field(default_factory=now)
    left_at: datetime | None = None


class Reaction(SQLModel, table=True):
    """Preset reactions only (P1). Free-text chat is P3."""

    id: int | None = Field(default=None, primary_key=True)
    circle_id: int = Field(foreign_key="circle.id", index=True)
    from_member_id: int = Field(foreign_key="membership.id")
    to_member_id: int = Field(foreign_key="membership.id", index=True)
    kind: str  # a key from routers/social.py PRESETS
    ts: datetime = Field(default_factory=now)


class NudgeCancel(SQLModel, table=True):
    """The patient cancelled the friends card for this escalation within the hold window."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    escalation_id: int = Field(foreign_key="escalation.id")
    ts: datetime = Field(default_factory=now)


class CompanionMessage(SQLModel, table=True):
    """At most one avatar check-in a day (#25). The reply's text is never stored, only that there was one."""

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: Date = Field(index=True)  # the patient's local day
    ts: datetime = Field(default_factory=now)
    kind: str  # trigger: missed_one_after_streak | missed_few | short_memos | back_after_break
    line_id: str
    text: str
    offer_written: bool = False
    seen_at: datetime | None = None
    replied_at: datetime | None = None
