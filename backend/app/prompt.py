"""Daily prompt (issue #22, STATUS.md B23): "Time for your check-in", once a day at a random time.

Experience sampling (EMA): the time is drawn uniformly between 09:00 and 21:00 local time, outside the
person's quiet hours, so check-ins catch the day as it is, not only the moments people choose to log.

- Drawn lazily, the first time today's prompt is asked for, but always over the whole day's window, so
  the time does not depend on when the app is opened. A prompt drawn for 10:00 and first seen at 15:00
  is simply open (and late) at once.
- On time = answered within WINDOW_MIN of the prompt. A late check-in still counts fully: lateness only
  means no on-time bonus (#24), never a penalty (#18 "never punish").
- Only a check-in after the prompt has fired answers it, once. Each check-in stores the prompt it
  answered (`CheckIn.prompt_at`) and its local time (`CheckIn.local_hour`) for the engine (#27).

Read helpers for other features (companion #25, rewards #24): todays, status, is_quiet, local_now.
"""

import random
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlmodel import Session, select

from .models import CheckIn
from .models_prompt import DailyPrompt, PromptSettings

WINDOW_MIN = 120  # on-time window after the prompt fires (issue #22: 2 h)
DAY_MIN = 24 * 60
RNG = random.Random()  # tests replace it to make draws reproducible


def _aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo; everything is stored in UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_hm(hm: str) -> int:
    """'HH:MM' -> minutes after midnight (00:00–24:00)."""
    h, m = hm.split(":")
    minutes = int(h) * 60 + int(m)
    if not (0 <= int(m) < 60 and 0 <= minutes <= DAY_MIN):
        raise ValueError(f"not a time of day: {hm!r}")
    return minutes


def fmt_hm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def tz_of(st: PromptSettings) -> ZoneInfo:
    try:
        return ZoneInfo(st.tz)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def settings_for(s: Session, patient_id: int) -> PromptSettings:
    st = s.get(PromptSettings, patient_id)
    if st is None:
        st = PromptSettings(patient_id=patient_id)
        s.add(st)
        s.flush()
    return st


def local_now(s: Session, patient_id: int, now: datetime) -> datetime:
    return _aware(now).astimezone(tz_of(settings_for(s, patient_id)))


def allowed_intervals(st: PromptSettings) -> list[tuple[int, int]]:
    """Minutes of the day [start, end) when a prompt may fire: the window minus quiet hours.

    Quiet hours may wrap midnight (22:00–07:00), then they cut [22:00, 24:00) and [00:00, 07:00).
    """
    lo, hi = parse_hm(st.window_start), parse_hm(st.window_end)
    spans = [(lo, hi)] if lo < hi else []
    if st.quiet_start and st.quiet_end:
        qs, qe = parse_hm(st.quiet_start), parse_hm(st.quiet_end)
        quiet = [(qs, qe)] if qs < qe else [(qs, DAY_MIN), (0, qe)] if qs > qe else []
        for q0, q1 in quiet:
            spans = [piece for a, b in spans for piece in ((a, min(b, q0)), (max(a, q1), b)) if piece[0] < piece[1]]
    return spans


def draw_minute(intervals: list[tuple[int, int]], rng: random.Random) -> int | None:
    """Uniform over all allowed minutes (each minute equally likely, wherever it falls)."""
    total = sum(b - a for a, b in intervals)
    if total <= 0:
        return None
    k = rng.randrange(total)
    for a, b in intervals:
        if k < b - a:
            return a + k
        k -= b - a
    raise RuntimeError("unreachable")


def in_quiet_hours(st: PromptSettings, minute: int) -> bool:
    return not any(a <= minute < b for a, b in allowed_intervals(st))


def _off_today(st: PromptSettings, today: date) -> str | None:
    if not st.enabled:
        return "off"
    if st.paused_on == today:
        return "paused"
    return None


def _existing(s: Session, patient_id: int, day: date) -> DailyPrompt | None:
    return s.exec(select(DailyPrompt).where(DailyPrompt.patient_id == patient_id, DailyPrompt.date == day)).first()


def todays(s: Session, patient_id: int, now: datetime, rng: random.Random | None = None) -> DailyPrompt | None:
    """Today's prompt, drawn on first use. None when prompts are off or paused today, or when quiet
    hours leave no time in the window."""
    st = settings_for(s, patient_id)
    tz = tz_of(st)
    today = _aware(now).astimezone(tz).date()
    if _off_today(st, today):
        return None
    p = _existing(s, patient_id, today)
    if p is not None:
        return p
    minute = draw_minute(allowed_intervals(st), rng or RNG)
    if minute is None:
        return None
    at_local = datetime.combine(today, time(0), tzinfo=tz) + timedelta(minutes=minute)
    # Two requests drawing at once: unique (patient, date) makes the second fail; routers retry once.
    p = DailyPrompt(patient_id=patient_id, date=today, at=at_local.astimezone(timezone.utc))
    s.add(p)
    s.flush()
    return p


def send_now(s: Session, patient_id: int, now: datetime) -> DailyPrompt:
    """Demo: make today's prompt fire now (and open again if it was already answered)."""
    st = settings_for(s, patient_id)
    today = _aware(now).astimezone(tz_of(st)).date()
    p = _existing(s, patient_id, today) or DailyPrompt(patient_id=patient_id, date=today, at=now)
    p.at, p.answered_at, p.checkin_id, p.source = now, None, None, "demo"
    s.add(p)
    s.flush()
    return p


def update_settings(s: Session, patient_id: int, now: datetime, changes: dict) -> PromptSettings:
    """Apply settings. `pause_today` pauses (True) or resumes (False) today only. If today's prompt has
    not fired yet, it is drawn again under the new settings (it may then be open at once)."""
    st = settings_for(s, patient_id)
    for k in ("tz", "window_start", "window_end", "quiet_start", "quiet_end", "enabled"):
        if k in changes:
            setattr(st, k, changes[k])
    today = _aware(now).astimezone(tz_of(st)).date()
    if "pause_today" in changes:
        st.paused_on = today if changes["pause_today"] else None
    s.add(st)
    p = _existing(s, patient_id, today)
    if p is not None and p.answered_at is None and p.source == "random" and _aware(p.at) > _aware(now):
        s.delete(p)
    s.flush()
    return st


def is_quiet(s: Session, patient_id: int, now: datetime) -> bool:
    """True when nothing should reach the person now: prompts off, paused today, quiet hours, or outside
    the prompt window (night)."""
    st = settings_for(s, patient_id)
    lnow = _aware(now).astimezone(tz_of(st))
    if _off_today(st, lnow.date()):
        return True
    return in_quiet_hours(st, lnow.hour * 60 + lnow.minute)


def _on_time(p: DailyPrompt) -> bool | None:
    if p.answered_at is None:
        return None
    return _aware(p.answered_at) - _aware(p.at) <= timedelta(minutes=WINDOW_MIN)


def status(s: Session, patient_id: int, now: datetime) -> dict:
    """What the Today screen needs: is the prompt open, and how long is left in the on-time window."""
    now = _aware(now)
    st = settings_for(s, patient_id)
    tz = tz_of(st)
    today = now.astimezone(tz).date()
    off = _off_today(st, today)
    p = None if off else todays(s, patient_id, now)
    out = {
        "date": today.isoformat(),
        "state": off or "none",  # off | paused | none | scheduled | open | answered
        "open": False,
        "at": None,
        "at_local": None,
        "window_minutes": WINDOW_MIN,
        "on_time_left_s": None,
        "answered_at": None,
        "on_time": None,
        "source": None,
        "settings": {
            "tz": st.tz, "window_start": st.window_start, "window_end": st.window_end,
            "quiet_start": st.quiet_start, "quiet_end": st.quiet_end,
            "enabled": st.enabled, "paused_today": st.paused_on == today,
        },
    }
    if p is None:
        return out
    at = _aware(p.at)
    out.update(at=at.isoformat(), at_local=at.astimezone(tz).strftime("%H:%M"), source=p.source)
    if p.answered_at is not None:
        out.update(state="answered", answered_at=_aware(p.answered_at).isoformat(), on_time=_on_time(p))
    elif now < at:
        out["state"] = "scheduled"
    else:
        left = at + timedelta(minutes=WINDOW_MIN) - now
        out.update(state="open", open=True, on_time_left_s=max(0, int(left.total_seconds())))
    return out


def answer(s: Session, patient_id: int, checkin: CheckIn, now: datetime) -> dict | None:
    """Called for every check-in. Stores its local time, and if today's prompt has fired and is still
    unanswered, marks it answered by this check-in.

    Returns None (nothing answered) or {answered, on_time, prompt_at, late_by_s}. Does not draw a new
    prompt: a check-in from a client that never showed the prompt must not count as prompted.
    """
    now = _aware(now)
    st = settings_for(s, patient_id)
    lnow = now.astimezone(tz_of(st))
    checkin.local_hour = round(lnow.hour + lnow.minute / 60 + lnow.second / 3600, 3)
    s.add(checkin)
    if _off_today(st, lnow.date()):
        return None
    p = _existing(s, patient_id, lnow.date())
    if p is None or p.answered_at is not None or now < _aware(p.at):
        return None
    p.answered_at, p.checkin_id = now, checkin.id
    checkin.prompt_at = p.at
    s.add(p)
    late = now - _aware(p.at) - timedelta(minutes=WINDOW_MIN)
    return {
        "answered": True,
        "on_time": bool(_on_time(p)),
        "prompt_at": _aware(p.at).isoformat(),
        "late_by_s": max(0, int(late.total_seconds())),
    }
