"""Appie read model over the canonical rewards, prompt and companion engines.

There is one wallet: RewardLedger. Food is its balance; XP is its positive lifetime
earnings. Feeding adds a spend, never turns XP back down. All Appie writes take a
patient row write lock before reading balances/keys, including on SQLite.
"""

import json
import math
import uuid
from datetime import date, datetime, timedelta

from sqlalchemy import inspect, text, update
from sqlmodel import Session, col, select

from . import baseline, prompt, rewards
from .engine import companion as messages
from .models import BaselineStat, BaselineStatus, InterventionLog, Patient
from .models_appie import AppieFeed, AppieHabit, AppieHabitLog, AppiePet
from .models_rewards import RewardLedger
from .models_social import IdentityMode, Profile
from .pipeline.phrase import PHRASE

LEVELS = [0, 50, 150, 350, 700, 1200, 2000]
TITLES = ["Newcomer", "Explorer", "Regular", "Steady", "Trailblazer", "Guide", "Legend"]
CUSTOM_REWARD, CUSTOM_MAX_PER_DAY = 2, 3
HEALTH = {"sleep": (5, "Slept well", "bedtime"), "steps": (5, "Reached your usual steps", "directions_walk"),
          "active": (5, "20 active minutes", "fitness_center")}


def level_of(xp: int) -> int:
    return max(i + 1 for i, threshold in enumerate(LEVELS) if xp >= threshold)


def tier_of(level: int) -> int:
    return 1 if level < 3 else 2 if level < 5 else 3


def _lock(s: Session, pid: int) -> None:
    # A write before reads serializes read/modify/write across workers as well as threads.
    result = s.execute(update(Patient).where(Patient.id == pid).values(name=Patient.name))
    if not result.rowcount:
        rewards.require_patient(s, pid)


def _day(s: Session, pid: int, now: datetime) -> date:
    return rewards.local_day(now, rewards.patient_tz(s, pid, now))


def _pet(s: Session, pid: int) -> AppiePet:
    pet = s.get(AppiePet, pid)
    if pet is None:
        pet = AppiePet(patient_id=pid)
        s.add(pet)
        s.flush()
    return pet


def _legacy_key(reason: str, day: date, row_id: int) -> tuple[str, str]:
    mapped = {"memo_voice": "memo", "memo_text": "memo", "on_time": "memo_on_time"}.get(reason, reason)
    if mapped in rewards.AMOUNTS:
        return f"{mapped}:{day}", mapped
    if mapped.startswith("habit_") and mapped[6:] in HEALTH:
        key = mapped[6:]
        return f"appie:health:{key}:{day}", f"appie:health:{key}"
    if mapped.startswith("custom:"):
        key = mapped.removeprefix("custom:")
        return f"appie:custom:{key}:{day}", f"appie:custom:{key}"
    return f"appie:legacy:row:{row_id}", reason


def migrate_legacy_patient(s: Session, pid: int, now: datetime) -> AppiePet:
    """Import Claude's old tables without importing old models, dropping rows or opening another wallet.

    Run at startup before normal rewards reconciliation. Existing unkeyed rewards get
    canonical keys first, so reconciliation cannot pay those same days again. A ledger
    reconciliation preserves the previous pet's food exactly; old fed XP is retained
    as treat history and included in the minimum lifetime earned progress.
    """
    pet = _pet(s, pid)
    if pet.legacy_migrated:
        return pet
    tables = set(inspect(s.connection()).get_table_names())
    old = None
    if "companion" in tables:
        columns = {column["name"] for column in inspect(s.connection()).get_columns("companion")}
        if {"patient_id", "food", "xp", "name"} <= columns:
            old = s.execute(text('SELECT patient_id, name, food, xp FROM companion WHERE patient_id=:pid'), {"pid": pid}).mappings().first()
    if old is not None:
        pet.name = str(old["name"] or "Appie").strip()[:16] or "Appie"
        pet.fed_xp = max(0, int(old["xp"] or 0))
        have = set(s.exec(select(RewardLedger.key).where(RewardLedger.patient_id == pid, col(RewardLedger.key).is_not(None))).all())
        rows = s.exec(select(RewardLedger).where(RewardLedger.patient_id == pid, col(RewardLedger.key).is_(None), RewardLedger.amount > 0).order_by(col(RewardLedger.id))).all()
        for row in rows:
            key, reason = _legacy_key(row.reason, row.date, row.id)
            # Old text+voice upgrades may be separate positive rows. Preserve them,
            # while only the first row reserves that day's canonical memo key.
            row.key = key if key not in have else f"appie:legacy:row:{row.id}"
            row.reason = reason if key not in have else f"appie:legacy:row:{row.id}"
            have.add(row.key)
            s.add(row)
        s.flush()
        day = _day(s, pid, now)
        target_food = max(0, int(old["food"] or 0))
        missing_earned = max(0, target_food + pet.fed_xp - rewards.xp(s, pid))
        if missing_earned:
            s.add(RewardLedger(patient_id=pid, ts=now, date=day, reason="appie:legacy:progress", amount=missing_earned, key="appie:legacy:progress"))
            s.flush()
        difference = target_food - rewards.balance(s, pid)
        if difference:
            s.add(RewardLedger(patient_id=pid, ts=now, date=day, reason="appie:legacy:balance", amount=difference, key="appie:legacy:balance"))
            s.flush()
    if "habit" in tables:
        columns = {column["name"] for column in inspect(s.connection()).get_columns("habit")}
        if {"patient_id", "key", "label", "active"} <= columns:
            for row in s.execute(text('SELECT "key", label, active FROM habit WHERE patient_id=:pid'), {"pid": pid}).mappings():
                label = " ".join(str(row["label"]).split())[:40]
                norm = label.casefold()
                if label and not s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid, AppieHabit.normalized_label == norm)).first():
                    s.add(AppieHabit(patient_id=pid, key=row["key"], label=label, normalized_label=norm, active=bool(row["active"])))
    if "habitlog" in tables:
        columns = {column["name"] for column in inspect(s.connection()).get_columns("habitlog")}
        if {"patient_id", "date", "habit_key", "source"} <= columns:
            for row in s.execute(text('SELECT date, habit_key, source FROM habitlog WHERE patient_id=:pid'), {"pid": pid}).mappings():
                d = date.fromisoformat(str(row["date"])[:10])
                _log(s, pid, d, row["habit_key"], row["source"] or "manual")
    pet.legacy_migrated = True
    pet.last_level = level_of(rewards.xp(s, pid))
    s.add(pet)
    s.flush()
    return pet


def migrate_legacy(engine) -> None:
    """Idempotent startup helper, after new tables and nullable ledger.key exist."""
    with Session(engine) as s:
        for pid in s.exec(select(Patient.id)).all():
            _lock(s, pid)
            migrate_legacy_patient(s, pid, baseline.clock())
        s.commit()
    if engine.dialect.name == "sqlite":
        with engine.begin() as connection:
            duplicates = connection.exec_driver_sql('SELECT patient_id, "key", COUNT(*) FROM rewardledger WHERE "key" IS NOT NULL GROUP BY patient_id, "key" HAVING COUNT(*) > 1 LIMIT 1').first()
            if duplicates:
                raise RuntimeError("Reward ledger contains duplicate canonical keys. Back up and reconcile before starting.")
            connection.exec_driver_sql('CREATE UNIQUE INDEX IF NOT EXISTS uq_appie_reward_key ON rewardledger (patient_id, "key") WHERE "key" IS NOT NULL')


def _prepare(s: Session, pid: int, now: datetime) -> AppiePet:
    _lock(s, pid)
    return migrate_legacy_patient(s, pid, now)


def _log(s: Session, pid: int, day: date, key: str, source: str = "manual") -> bool:
    if s.exec(select(AppieHabitLog).where(AppieHabitLog.patient_id == pid, AppieHabitLog.date == day, AppieHabitLog.habit_key == key)).first():
        return False
    s.add(AppieHabitLog(patient_id=pid, date=day, habit_key=key, source=source))
    s.flush()
    return True


def _award(s: Session, pid: int, day: date, now: datetime, reason: str, amount: int, label: str) -> dict | None:
    key = f"{reason}:{day}"
    if s.exec(select(RewardLedger.id).where(RewardLedger.patient_id == pid, RewardLedger.key == key)).first():
        return None
    s.add(RewardLedger(patient_id=pid, ts=now, date=day, reason=reason, amount=amount, key=key))
    s.flush()
    return {"reason": reason, "label": label, "amount": amount}


def on_wearable(s: Session, pid: int, now: datetime, values: dict[str, float]) -> list[dict]:
    """Only actual wearable observations call this. The caller owns the transaction."""
    _prepare(s, pid, now)
    day, usual = _day(s, pid, now), {}
    active = baseline.latest(s, pid, BaselineStatus.active)
    if active:
        usual = {stat.feature: stat.center for stat in s.exec(select(BaselineStat).where(BaselineStat.baseline_id == active.id, BaselineStat.slot == "all")).all() if stat.center > 0}
    def meets(name: str, threshold: float):
        value = values.get(name)
        return value is not None and math.isfinite(value) and value >= threshold
    done = {
        "sleep": meets("sleep_hours", min(7.0, usual.get("sleep_hours", 7.0))),
        "steps": meets("steps", .9 * usual.get("steps", 6000 / .9)),
        "active": meets("active_minutes", 20),
    }
    out = []
    for key, completed in done.items():
        if completed:
            _log(s, pid, day, key, "wearable")
            amount, label, _ = HEALTH[key]
            reward = _award(s, pid, day, now, f"appie:health:{key}", amount, label)
            if reward:
                reward["reason"] = f"habit_{key}"
                out.append(reward)
    return out


def _label(reason: str, habits: list[AppieHabit]) -> str:
    if reason in rewards.LABELS:
        return rewards.LABELS[reason]
    if reason.startswith("appie:health:"):
        return HEALTH.get(reason.rsplit(":", 1)[-1], (0, "Wearable habit", ""))[1]
    if reason.startswith("appie:custom:"):
        return next((habit.label for habit in habits if f"appie:custom:{habit.key}" == reason), "Custom habit")
    return "Previous Appie progress" if reason.startswith("appie:legacy:") else reason.removeprefix("spend:")


def view(s: Session, pid: int) -> dict:
    now = baseline.clock()
    pet = _prepare(s, pid, now)
    rewards.sync(s, pid, now)
    patient = rewards.require_patient(s, pid)
    presence = rewards.presence(s, pid, now)
    today = presence["today"]
    history = rewards.checkin_history(s, pid, rewards.patient_tz(s, pid, now))
    all_days = {entry["day"] for entry in history}
    canonical_week = rewards.week(s, pid, now)
    builtins = {habit["key"]: habit for habit in canonical_week["habits"]}
    custom = list(s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid)).all())
    logs = {(row.date, row.habit_key) for row in s.exec(select(AppieHabitLog).where(AppieHabitLog.patient_id == pid, AppieHabitLog.date >= today - timedelta(days=6))).all()}
    habits = [{"key": key, "label": label, "icon": icon, "auto": True, "food": rewards.AMOUNTS[reason], "done": builtins[key]["done"][-1], "hint": hint}
              for key, label, icon, reason, hint in [
                  ("memo", "Daily check-in", "mic", "memo", "10 seconds of speech or 20 written words"),
                  ("action", "Today's small thing", "bolt", "action_done", "Let us know how today's action went"),
                  ("followup", "Your action follow-up", "chat_bubble", "followup", "An honest answer counts, including not today"),
              ]]
    habits += [{"key": key, "label": label, "icon": icon, "auto": True, "food": amount, "done": (today, key) in logs, "hint": "From your latest completed Fitbit day"} for key, (amount, label, icon) in HEALTH.items()]
    habits += [{"key": h.key, "label": h.label, "icon": "favorite", "auto": False, "food": CUSTOM_REWARD, "done": (today, h.key) in logs, "hint": "Tap when you've done it"} for h in custom if h.active]
    keys = {h["key"] for h in habits}
    week = [{"date": day, "checked_in": date.fromisoformat(day) in all_days, "done": sum(h["done"][i] for h in builtins.values()) + sum((date.fromisoformat(day), key) in logs for key in keys - set(builtins)), "total": len(habits)} for i, day in enumerate(canonical_week["days"])]
    total = rewards.xp(s, pid)
    level, tier = level_of(total), tier_of(level_of(total))
    if level > pet.last_level:
        pet.last_level, pet.leveled_on = level, today
        s.add(pet)
    status = prompt.status(s, pid, now)
    at = datetime.fromisoformat(status["at"]) if status["at"] else None
    deadline = at + timedelta(minutes=prompt.WINDOW_MIN) if at else None
    msg = messages.evaluate(s, pid, now)
    checked = presence["checked_in_today"]
    kind = msg.kind if msg else "done" if checked else "welcome" if not history else "waiting"
    line = msg.text if msg else "Thanks for today. Every kind of day belongs here." if checked else f"Hi, I'm {pet.name}. Tell me about your day and I'll keep you company." if not history else "I'm here whenever you feel like checking in."
    local_hour = prompt.local_now(s, pid, now).hour
    mood = "wave" if kind in ("welcome", "back_after_break") else "happy" if checked else "care" if msg else "sleep" if local_hour >= 21 or local_hour < 7 else "calm"
    if pet.leveled_on == today:
        mood = "celebrate"
    rows = s.exec(select(RewardLedger).where(RewardLedger.patient_id == pid, RewardLedger.date == today, RewardLedger.amount > 0).order_by(col(RewardLedger.ts))).all()
    profile = s.get(Profile, pid)
    anonymous = profile is None or profile.mode == IdentityMode.anonymous
    display_name = (profile.pseudonym or "Anonymous") if profile and anonymous else (profile.display_name or patient.name) if profile else "Anonymous"
    out = {
        "patient": {"id": pid, "name": patient.name}, "name": pet.name, "mood": mood, "message": line, "message_kind": kind,
        "message_id": msg.id if msg else None, "daily_phrase": PHRASE,
        "level": level, "title": TITLES[level - 1], "tier": tier, "xp": total, "xp_level_start": LEVELS[level - 1], "xp_next": LEVELS[level] if level < len(LEVELS) else None,
        "next_unlock": {1: "Headphones at level 3", 2: "Backpack and star at level 5"}.get(tier),
        "food": rewards.balance(s, pid), "fed_xp": pet.fed_xp, "feed_count": pet.feed_count,
        "streak": {"current": presence["streak"], "longest": presence["longest_streak"], "today": checked, "days_since_last": presence["days_since_last_checkin"]},
        "prompt": {**status, "answered": status["state"] == "answered", "on_time_until": deadline.isoformat() if deadline else None, "bonus_available": status["open"] and bool(status["on_time_left_s"]), "minutes_left": max(0, (status["on_time_left_s"] or 0) // 60), "done_before_prompt": checked and status["state"] == "scheduled"},
        "habits": habits, "week": week, "earned_today": [{"reason": row.reason, "label": _label(row.reason, custom), "amount": row.amount} for row in rows],
        "earned_today_total": sum(row.amount for row in rows), "profile": {"display_name": display_name, "anonymous": anonymous},
    }
    s.commit()
    return out


def feed(s: Session, pid: int, amount: int, request_id: str | None = None) -> dict:
    now = baseline.clock()
    pet = _prepare(s, pid, now)
    if request_id:
        prior = s.exec(select(AppieFeed).where(AppieFeed.patient_id == pid, AppieFeed.request_id == request_id)).first()
        if prior:
            return json.loads(prior.response_json)
    rewards.sync(s, pid, now)
    food = rewards.balance(s, pid)
    amount = min(amount, food)
    if amount <= 0:
        raise rewards.InsufficientBalance("No food left. A check-in or a small habit can earn more.")
    # Legacy ledgers may also retain UNIQUE(patient,date,reason). Each spend gets
    # its own reason so repeated legitimate feeds work without rebuilding tables.
    rewards.spend(pid, amount, f"appie_food:{request_id or uuid.uuid4().hex}", s=s)
    pet.fed_xp += amount
    pet.feed_count += 1
    pet.last_fed_at = now
    s.add(pet)
    xp = rewards.xp(s, pid)
    level = level_of(xp)
    out = {"fed": amount, "food": rewards.balance(s, pid), "xp": xp, "level": level, "leveled_up": False, "tier": tier_of(level), "tier_up": False, "title": TITLES[level - 1], "fed_xp": pet.fed_xp}
    if request_id:
        s.add(AppieFeed(patient_id=pid, request_id=request_id, response_json=json.dumps(out)))
    s.commit()
    return out


def add_habit(s: Session, pid: int, label: str) -> AppieHabit:
    _prepare(s, pid, baseline.clock())
    norm = label.casefold()
    habit = s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid, AppieHabit.normalized_label == norm)).first()
    if habit is None:
        habit = AppieHabit(patient_id=pid, key=f"h_{uuid.uuid4().hex[:16]}", label=label, normalized_label=norm)
    habit.active = True
    s.add(habit)
    s.commit()
    s.refresh(habit)
    return habit


def change_habit(s: Session, pid: int, key: str, label: str | None = None) -> AppieHabit:
    _prepare(s, pid, baseline.clock())
    habit = s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid, AppieHabit.key == key)).first()
    if habit is None:
        raise LookupError("Habit not found.")
    if label is None:
        habit.active = False
    else:
        duplicate = s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid, AppieHabit.normalized_label == label.casefold(), AppieHabit.id != habit.id)).first()
        if duplicate:
            raise ValueError("You already have a habit with that name.")
        habit.label, habit.normalized_label = label, label.casefold()
    s.add(habit)
    s.commit()
    s.refresh(habit)
    return habit


def tick_habit(s: Session, pid: int, key: str) -> dict:
    now = baseline.clock()
    _prepare(s, pid, now)
    habit = s.exec(select(AppieHabit).where(AppieHabit.patient_id == pid, AppieHabit.key == key, AppieHabit.active == True)).first()  # noqa: E712
    if habit is None:
        raise LookupError("Habit not found. Built-in habits are completed automatically.")
    day = _day(s, pid, now)
    new = _log(s, pid, day, key)
    count = len(s.exec(select(RewardLedger.id).where(RewardLedger.patient_id == pid, RewardLedger.date == day, col(RewardLedger.reason).startswith("appie:custom:"), RewardLedger.amount > 0)).all())
    reward = _award(s, pid, day, now, f"appie:custom:{key}", CUSTOM_REWARD, habit.label) if new and count < CUSTOM_MAX_PER_DAY else None
    s.commit()
    return {"ok": True, "reward": reward}
