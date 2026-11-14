"""Habits and rewards (#24): earn, spend, balance, XP, gentle streaks.

Rewards are reconciled from what is already stored (check-ins, the bandit's InterventionLog, and the
daily prompt from #22 when it exists) by `sync()`. That function is idempotent: every reward has a key
such as "memo:2026-11-14" and is paid once. So the check-in upload needs no hook, seeded demo data earns
like live use, and nothing earned is ever taken back: sync only adds rows.

Product rules (#18):
- Reward doing, never the score. Nothing here reads StateScore, features or transcripts' content.
  A check-in on a bad day earns the same as one on a good day.
- Gentle streaks: one rest day a week keeps the streak; the longest streak is never lost.
- Anti-gaming: one memo reward a day; a spoken memo needs >= 10 s of speech, a written one >= 20 words.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone, tzinfo

from sqlmodel import Session, col, select
from sqlalchemy import update

from .db import engine
from .models import Arm, CheckIn, Feature, FeatureSource, InterventionLog, Patient
from .models_rewards import RewardLedger

AMOUNTS = {"memo": 10, "memo_on_time": 5, "action_done": 10, "followup": 3, "streak_7": 20}
LABELS = {
    "memo": "Daily check-in",
    "memo_on_time": "Checked in within the prompt window",
    "action_done": "Today's small thing done",
    "followup": "Told us how it went",
    "streak_7": "7-day streak",
}
MIN_SPEECH_S = 10.0
MIN_WORDS = 20  # a written check-in, roughly 10 s of speech
ON_TIME_MIN = 120  # the prompt's on-time window (#22)
STREAK_BONUS_EVERY = 7
REST_DAY_EVERY = 7  # one rest day per 7 days keeps the streak
# Level hint for the avatar (#23 owns the real levels): XP needed for level 1..5
LEVEL_XP = [0, 50, 150, 300, 500]

BUILTIN_HABITS = [
    ("memo", "Daily check-in"),
    ("action", "Today's small thing"),
    ("followup", "Told us how it went"),
]


class InsufficientBalance(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# Time: the patient's local day
# ---------------------------------------------------------------------------------------------


def aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo; everything is stored in UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def patient_tz(s: Session, patient_id: int, now: datetime) -> tzinfo:
    """The patient's time zone from the daily prompt settings (#22) if merged, otherwise UTC."""
    try:
        from . import prompt
    except ImportError:
        return timezone.utc
    return prompt.local_now(s, patient_id, aware(now)).tzinfo or timezone.utc


def local_day(dt: datetime, tz: tzinfo) -> date:
    return aware(dt).astimezone(tz).date()


# ---------------------------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------------------------


def streak_at(days: set[date], end: date) -> int:
    """Gentle streak of active days ending at `end` (a rest day may sit right before `end`'s activity).

    Walk back from `end`: an active day adds 1. A single missed day is a rest day and keeps the streak if
    the day before it is active and no other rest day is within REST_DAY_EVERY days. Rest days add 0.
    """
    d, n, rests = end, 0, []
    while True:
        if d in days:
            n += 1
        elif (d - timedelta(days=1)) in days and all(abs((r - d).days) >= REST_DAY_EVERY for r in rests):
            rests.append(d)
        else:
            return n
        d -= timedelta(days=1)


def current_streak(days: set[date], today: date) -> int:
    """Today not checked in yet does not break anything: count from today (it may be the rest day)."""
    return streak_at(days, today)


def longest_streak(days: set[date]) -> int:
    return max((streak_at(days, d) for d in days), default=0)


# ---------------------------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------------------------


def memo_counts(spoken: bool, speech_s: float | None, n_words: int) -> bool:
    """Anti-gaming: a spoken memo needs >= 10 s of speech (not measured yet -> benefit of the doubt),
    a written check-in >= 20 words."""
    if spoken:
        return speech_s is None or speech_s >= MIN_SPEECH_S
    return n_words >= MIN_WORDS


def _spoken_ids(s: Session, ids: list[int]) -> set[int]:
    if not ids:
        return set()
    return set(s.exec(
        select(Feature.checkin_id).where(col(Feature.checkin_id).in_(ids), Feature.source == FeatureSource.prosody,
                                        ~col(Feature.name).startswith("phrase_", autoescape=True))
    ).all())


def checkin_history(s: Session, patient_id: int, tz: tzinfo) -> list[dict]:
    """Each check-in as {id, ts, day, spoken, speech_s, n_words, length_s, qualifies}. Content is not kept."""
    rows = s.exec(select(CheckIn).where(CheckIn.patient_id == patient_id).order_by(col(CheckIn.ts))).all()
    # Wearable-only observations and the fixed reading phrase are not daily memos.
    # This history also drives social presence and companion messages, so a sync
    # must not make a person appear to have checked in or reset the missing-day gap.
    memo_ids = set(s.exec(select(Feature.checkin_id).where(
        col(Feature.checkin_id).in_([c.id for c in rows]),
        col(Feature.source).in_((FeatureSource.prosody, FeatureSource.text)),
        ~col(Feature.name).startswith("phrase_", autoescape=True),
    )).all()) if rows else set()
    feature_ids = set(s.exec(select(Feature.checkin_id).where(col(Feature.checkin_id).in_([c.id for c in rows]))).all()) if rows else set()
    spoken = _spoken_ids(s, [c.id for c in rows])
    out = []
    for c in rows:
        legacy_memo = c.id not in feature_ids and (bool((c.transcript or "").strip()) or c.speech_s is not None)
        if c.id not in memo_ids and not legacy_memo:
            continue
        speech_s = getattr(c, "speech_s", None)  # set by the check-in flow once #21/#22 is merged
        is_spoken = c.id in spoken or speech_s is not None
        n_words = len((c.transcript or "").split())
        qualifies = memo_counts(is_spoken, speech_s, n_words)
        if speech_s is not None:
            length_s = float(speech_s)
        elif n_words:
            length_s = n_words / 2.5  # ~150 words per minute
        else:
            length_s = None
        out.append({
            "id": c.id, "ts": aware(c.ts), "day": local_day(c.ts, tz), "spoken": is_spoken,
            "speech_s": speech_s, "n_words": n_words, "length_s": length_s, "qualifies": qualifies,
        })
    return out


def _on_time_checkins(s: Session, patient_id: int) -> set[int]:
    """Check-in ids that answered the daily prompt within its window. Empty until #22 is merged."""
    try:
        from .models_prompt import DailyPrompt
    except ImportError:
        return set()
    out = set()
    for p in s.exec(select(DailyPrompt).where(DailyPrompt.patient_id == patient_id)).all():
        if p.checkin_id and p.answered_at and p.at and aware(p.answered_at) - aware(p.at) <= timedelta(minutes=ON_TIME_MIN):
            out.add(p.checkin_id)
    return out


# ---------------------------------------------------------------------------------------------
# Earning
# ---------------------------------------------------------------------------------------------


def sync(s: Session, patient_id: int, now: datetime) -> list[RewardLedger]:
    """Pay every reward that is due and not yet paid. Returns the new rows (not committed)."""
    # The unique ledger key prevents duplicate payment; serialize before reading
    # keys so concurrent app/social/reward refreshes also return successfully.
    s.execute(update(Patient).where(Patient.id == patient_id).values(name=Patient.name))
    tz = patient_tz(s, patient_id, now)
    have = set(s.exec(select(RewardLedger.key).where(
        RewardLedger.patient_id == patient_id, col(RewardLedger.key).is_not(None)
    )).all())
    due: list[tuple[str, str, date, datetime]] = []  # (key, reason, day, ts)

    hist = checkin_history(s, patient_id, tz)
    on_time = _on_time_checkins(s, patient_id)
    first_by_day: dict[date, dict] = {}
    for c in hist:
        if c["qualifies"] and c["day"] not in first_by_day:
            first_by_day[c["day"]] = c
    memo_days = set(first_by_day)
    for d, c in first_by_day.items():
        due.append((f"memo:{d}", "memo", d, c["ts"]))
        if any(x["id"] in on_time and x["day"] == d and x["qualifies"] for x in hist):
            due.append((f"memo_on_time:{d}", "memo_on_time", d, c["ts"]))
        n = streak_at(memo_days, d)
        if n and n % STREAK_BONUS_EVERY == 0:
            due.append((f"streak_7:{d}", "streak_7", d, c["ts"]))

    for log in s.exec(select(InterventionLog).where(InterventionLog.patient_id == patient_id)).all():
        if log.completed is not None:  # "not today" is an honest answer and earns too
            due.append((f"followup:{log.date}", "followup", log.date, now))
        if log.completed:
            due.append((f"action_done:{log.date}", "action_done", log.date, now))

    new = []
    for key, reason, d, ts in due:
        if key in have:
            continue
        row = RewardLedger(patient_id=patient_id, ts=ts, date=d, reason=reason, amount=AMOUNTS[reason], key=key)
        s.add(row)
        have.add(key)
        new.append(row)
    if new:
        s.flush()
    return new


def balance(s: Session, patient_id: int) -> int:
    return sum(s.exec(select(RewardLedger.amount).where(RewardLedger.patient_id == patient_id)).all())


def xp(s: Session, patient_id: int) -> int:
    """Total ever earned. Spending never lowers it."""
    return sum(a for a in s.exec(select(RewardLedger.amount).where(RewardLedger.patient_id == patient_id)).all() if a > 0)


def level_hint(xp_total: int) -> dict:
    level = max(i + 1 for i, need in enumerate(LEVEL_XP) if xp_total >= need)
    nxt = LEVEL_XP[level] if level < len(LEVEL_XP) else None
    return {"level": level, "xp": xp_total, "next_level_xp": nxt}


def spend(patient_id: int, amount: int, reason: str, s: Session | None = None) -> RewardLedger:
    """For the avatar shop (#23). Refuses if the balance is too low. Commits only its own session."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    from .baseline import clock

    own = s is None
    s = s or Session(engine)
    try:
        now = clock()
        sync(s, patient_id, now)
        if balance(s, patient_id) < amount:
            raise InsufficientBalance(f"balance too low for {amount}")
        row = RewardLedger(patient_id=patient_id, ts=now, date=local_day(now, patient_tz(s, patient_id, now)),
                           reason=f"spend:{reason}", amount=-amount)
        s.add(row)
        s.flush()
        if own:
            s.commit()
            s.refresh(row)
        return row
    finally:
        if own:
            s.close()


# ---------------------------------------------------------------------------------------------
# Read models
# ---------------------------------------------------------------------------------------------


def presence(s: Session, patient_id: int, now: datetime) -> dict:
    """Streak and "checked in today" – what the avatar, the companion and the circle may show."""
    tz = patient_tz(s, patient_id, now)
    today = local_day(now, tz)
    hist = checkin_history(s, patient_id, tz)
    memo_days = {c["day"] for c in hist if c["qualifies"]}
    all_days = {c["day"] for c in hist}
    last = max(all_days, default=None)
    return {
        "today": today,
        "streak": current_streak(memo_days, today),
        "longest_streak": longest_streak(memo_days),
        "checked_in_today": today in all_days,
        "days_since_last_checkin": (today - last).days if last else None,
    }


def summary(s: Session, patient_id: int, now: datetime, limit: int = 50) -> dict:
    """GET /patients/{id}/rewards – the contract agreed with the avatar work (#23)."""
    sync(s, patient_id, now)
    p = presence(s, patient_id, now)
    total = xp(s, patient_id)
    rows = s.exec(
        select(RewardLedger).where(RewardLedger.patient_id == patient_id)
        .order_by(col(RewardLedger.ts).desc(), col(RewardLedger.id).desc()).limit(limit)
    ).all()
    return {
        "balance": balance(s, patient_id),
        "xp": total,
        "level_hint": level_hint(total),
        "streak": p["streak"],
        "longest_streak": p["longest_streak"],
        "checked_in_today": p["checked_in_today"],
        "days_since_last_checkin": p["days_since_last_checkin"],
        "ledger": [
            {"ts": aware(r.ts).isoformat(), "date": r.date.isoformat(), "reason": r.reason,
             "label": LABELS.get(r.reason, r.reason.removeprefix("spend:")), "amount": r.amount}
            for r in rows
        ],
    }


def week(s: Session, patient_id: int, now: datetime, days: int = 7) -> dict:
    """Habits week view: the last `days` local days and which built-in habits were done."""
    sync(s, patient_id, now)
    tz = patient_tz(s, patient_id, now)
    today = local_day(now, tz)
    span = [today - timedelta(days=i) for i in reversed(range(days))]
    memo_days = {c["day"] for c in checkin_history(s, patient_id, tz) if c["qualifies"]}
    logs = {log.date: log for log in s.exec(
        select(InterventionLog).where(InterventionLog.patient_id == patient_id, col(InterventionLog.date) >= span[0])
    ).all()}
    done = {
        "memo": [d in memo_days for d in span],
        "action": [bool(logs.get(d) and logs[d].completed) for d in span],
        "followup": [bool(logs.get(d) and logs[d].completed is not None) for d in span],
    }
    earned: dict[date, int] = defaultdict(int)
    for r in s.exec(select(RewardLedger).where(
        RewardLedger.patient_id == patient_id, col(RewardLedger.date) >= span[0], RewardLedger.amount > 0
    )).all():
        earned[r.date] += r.amount
    # Today's small thing (the bandit's pick, made at the first check-in). Its log date is the UTC day.
    log = s.exec(select(InterventionLog).where(
        InterventionLog.patient_id == patient_id, InterventionLog.date == aware(now).date()
    )).first()
    arm = log and s.exec(select(Arm).where(Arm.patient_id == patient_id, Arm.key == log.arm_key)).first()
    return {
        "days": [d.isoformat() for d in span],
        "habits": [{"key": k, "label": label, "done": done[k]} for k, label in BUILTIN_HABITS],
        "earned": [earned[d] for d in span],
        "today_action": None if log is None else {
            "arm_key": log.arm_key, "label": arm.label if arm else log.arm_key,
            "answered": log.completed is not None, "done": bool(log.completed),
        },
    }


def require_patient(s: Session, patient_id: int) -> Patient:
    p = s.get(Patient, patient_id)
    if p is None:
        from fastapi import HTTPException

        raise HTTPException(404, "patient not found")
    return p
