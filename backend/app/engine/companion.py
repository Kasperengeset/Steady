"""Kind avatar check-ins (#25): rules over the check-in history, reviewed lines, at most one a day.

Triggers (first matching wins):
  back_after_break         checked in today after >= 2 missed days -> "glad to see you", no word on the gap
  missed_few               2-3 missed days -> "I've missed hearing from you", offers a written check-in
  missed_one_after_streak  a (gentle) streak >= 5, then 1 missed day -> "I kept your spot warm yesterday"
  short_memos              the last 3 days' memos all < half the person's usual length -> today's small thing

The avatar is a companion, not a clinician: it reads only *when* and *how long* people checked in,
never scores, features or what they said. It comes before the escalation ladder and does not replace it.
Proactive messages wait for the day's prompt time (#22); every message respects quiet hours and pause.
After 3 missed days it stops; the clinician may see "avatar checked in, no answer" as context only.
"""

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median

from sqlmodel import Session, col, select

from .. import rewards
from ..models_social import CompanionMessage

LINES: dict[str, list[dict]] = {
    k: v for k, v in json.loads((Path(__file__).parents[1] / "companion_lines.json").read_text()).items()
    if not k.startswith("_")
}
KINDS = ("back_after_break", "missed_few", "missed_one_after_streak", "short_memos")
BREAK_MIN_MISSED = 2
FEW_MISSED = (2, 3)
STREAK_MIN = 5
SHORT_DAYS = 3
SHORT_RATIO = 0.5
USUAL_DAYS = 14
USUAL_MIN_DAYS = 5
NO_ANSWER_DAYS = 3


def decide(days: set[date], lengths: dict[date, float], today: date) -> str | None:
    """Pure rule: which check-in (if any) fits today. `lengths` = longest memo per day, in seconds."""
    earlier = [d for d in days if d < today]
    if today in days:
        prev = max(earlier, default=None)
        return "back_after_break" if prev and (today - prev).days - 1 >= BREAK_MIN_MISSED else None
    if not earlier:
        return None  # nobody to miss yet
    last = max(earlier)
    missed = (today - last).days - 1
    if missed in FEW_MISSED:
        return "missed_few"
    if missed == 1 and rewards.streak_at(days, last) >= STREAK_MIN:
        return "missed_one_after_streak"
    if missed == 0:
        recent = [today - timedelta(days=i) for i in range(1, SHORT_DAYS + 1)]
        start = recent[-1]
        usual = [v for d, v in lengths.items() if start - timedelta(days=USUAL_DAYS) <= d < start]
        if (len(usual) >= USUAL_MIN_DAYS and all(d in lengths for d in recent)
                and all(lengths[d] < SHORT_RATIO * median(usual) for d in recent)):
            return "short_memos"
    return None


def pick_line(kind: str, patient_id: int, today: date, avoid: str | None = None) -> dict:
    """Stable per person and day (no randomness in tests), and not the same line as last time."""
    lines = LINES[kind]
    i = int(hashlib.sha256(f"{patient_id}:{today}:{kind}".encode()).hexdigest(), 16) % len(lines)
    if lines[i]["id"] == avoid and len(lines) > 1:
        i = (i + 1) % len(lines)
    return lines[i]


# ---------------------------------------------------------------------------------------------
# Timing: the daily prompt (#22), quiet hours and pause. Lazy imports: works before #22 is merged.
# ---------------------------------------------------------------------------------------------


def is_quiet(s: Session, patient_id: int, now: datetime) -> bool:
    try:
        from .. import prompt
    except ImportError:
        return False
    return bool(prompt.is_quiet(s, patient_id, now))


def prompt_time_reached(s: Session, patient_id: int, now: datetime) -> bool:
    """Proactive messages are sent at the day's prompt time. Off or paused today -> no prompt -> False."""
    try:
        from .. import prompt
    except ImportError:
        return True
    p = prompt.todays(s, patient_id, now)
    return p is not None and rewards.aware(now) >= rewards.aware(p.at)


# ---------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------


def _history(s: Session, patient_id: int, now: datetime) -> tuple[set[date], dict[date, float], date]:
    tz = rewards.patient_tz(s, patient_id, now)
    hist = rewards.checkin_history(s, patient_id, tz)
    lengths: dict[date, float] = {}
    for c in hist:
        if c["length_s"] is not None:
            lengths[c["day"]] = max(lengths.get(c["day"], 0.0), c["length_s"])
    return {c["day"] for c in hist}, lengths, rewards.local_day(now, tz)


def _todays_message(s: Session, patient_id: int, today: date) -> CompanionMessage | None:
    return s.exec(select(CompanionMessage).where(
        CompanionMessage.patient_id == patient_id, CompanionMessage.date == today
    )).first()


def evaluate(s: Session, patient_id: int, now: datetime, force: bool = False) -> CompanionMessage | None:
    """Today's message: the stored one, or a new one if a rule fires and the timing allows (not committed).

    force=True (demo) skips the prompt-time wait and replaces today's message. Quiet hours and pause still apply.
    """
    days, lengths, today = _history(s, patient_id, now)
    existing = _todays_message(s, patient_id, today)
    if existing and not force:
        return existing
    if is_quiet(s, patient_id, now):
        return existing
    kind = decide(days, lengths, today)
    if kind is None:
        return existing
    if kind != "back_after_break" and not force and not prompt_time_reached(s, patient_id, now):
        return None
    last = s.exec(select(CompanionMessage).where(
        CompanionMessage.patient_id == patient_id, CompanionMessage.kind == kind
    ).order_by(col(CompanionMessage.ts).desc())).first()
    line = pick_line(kind, patient_id, today, avoid=last.line_id if last else None)
    if existing:
        s.delete(existing)
        s.flush()
    msg = CompanionMessage(patient_id=patient_id, date=today, ts=now, kind=kind, line_id=line["id"],
                           text=line["text"], offer_written=bool(line.get("offer_written")))
    s.add(msg)
    s.flush()
    return msg


def glad_to_see_you(s: Session, patient_id: int, now: datetime) -> bool:
    """For the avatar's state (#23): the first check-in today came after a break."""
    days, lengths, today = _history(s, patient_id, now)
    return decide(days, lengths, today) == "back_after_break"


def clinician_context(s: Session, patient_id: int, now: datetime) -> dict:
    """"Avatar checked in, no answer for N days" – context for the clinician, never an alert or escalation."""
    days, _, today = _history(s, patient_id, now)
    last = max(days, default=None)
    msgs = s.exec(select(CompanionMessage).where(
        CompanionMessage.patient_id == patient_id, CompanionMessage.kind != "back_after_break"
    ).order_by(col(CompanionMessage.date))).all()
    replied = [m.date for m in msgs if m.replied_at]
    answered_until = max([d for d in [last, *replied] if d], default=None)
    open_ = [m for m in msgs if not m.replied_at and (answered_until is None or m.date > answered_until)]
    n_days = (today - open_[0].date).days if open_ else 0
    note = f"Avatar checked in, no answer for {n_days} days" if open_ and n_days >= NO_ANSWER_DAYS else None
    return {
        "avatar_checked_in": bool(open_),
        "unanswered_messages": len(open_),
        "first_unanswered_on": open_[0].date.isoformat() if open_ else None,
        "days_without_answer": n_days,
        "note": note,
        "alert": False,
    }
