"""API contract for the daily prompt (#22) – mirrored in frontend/src/lib/api.ts. Change both together."""

from pydantic import BaseModel


class PromptSettingsOut(BaseModel):
    tz: str
    window_start: str  # "HH:MM" local
    window_end: str
    quiet_start: str | None
    quiet_end: str | None
    enabled: bool
    paused_today: bool


class PromptOut(BaseModel):
    date: str  # local date
    state: str  # off | paused | none | scheduled | open | answered
    open: bool  # fired and not answered yet: show "Your check-in is ready"
    at: str | None  # UTC ISO time it fires
    at_local: str | None  # "HH:MM" in the person's tz
    window_minutes: int  # on-time window (bonus, #24); a late check-in still counts fully
    on_time_left_s: int | None  # while open: seconds left in the on-time window (0 = late)
    answered_at: str | None
    on_time: bool | None
    source: str | None  # random | demo
    settings: PromptSettingsOut


class PromptSettingsIn(BaseModel):
    """Only the fields sent are changed. quiet_start/quiet_end: null clears them."""

    tz: str | None = None
    quiet_start: str | None = None
    quiet_end: str | None = None
    enabled: bool | None = None
    pause_today: bool | None = None  # true: no prompt today; false: resume today


class PromptAnswerOut(BaseModel):
    """On a check-in response: this check-in answered today's prompt."""

    answered: bool
    on_time: bool  # within the 2 h window – the only thing it changes is the on-time bonus (#24)
    prompt_at: str
    late_by_s: int
