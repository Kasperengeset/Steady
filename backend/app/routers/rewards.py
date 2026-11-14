"""Rewards and the habits week view (#24). Contract with the avatar work (#23): see app/rewards.py summary()."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from .. import baseline, rewards
from ..db import get_session
from ..schemas import FeedbackIn
from . import interventions

router = APIRouter(tags=["rewards"])


class ActionDoneIn(BaseModel):
    done: bool  # False = "not today", which is an honest answer and still earns the follow-up reward
    rating: int | None = None  # 1-5, how it felt


@router.get("/patients/{pid}/rewards")
def get_rewards(pid: int, s: Session = Depends(get_session)) -> dict:
    """{balance, xp, level_hint, streak, longest_streak, checked_in_today, days_since_last_checkin, ledger}"""
    rewards.require_patient(s, pid)
    out = rewards.summary(s, pid, baseline.clock())
    s.commit()
    return out


@router.get("/patients/{pid}/habits/week")
def habits_week(pid: int, days: int = 7, s: Session = Depends(get_session)) -> dict:
    """The last `days` days and which built-in habits were done: {days, habits: [{key, label, done}], earned}."""
    rewards.require_patient(s, pid)
    out = rewards.week(s, pid, baseline.clock(), days=max(1, min(days, 31)))
    s.commit()
    return out


@router.post("/patients/{pid}/habits/action")
def mark_action(pid: int, body: ActionDoneIn, s: Session = Depends(get_session)) -> dict:
    """Tick today's small thing. This IS the bandit's follow-up answer (routers/interventions.py)."""
    rewards.require_patient(s, pid)
    now = baseline.clock()
    fb = interventions.feedback(FeedbackIn(patient_id=pid, date=now.date(), completed=body.done, rating=body.rating), s)
    earned = rewards.sync(s, pid, now)
    s.commit()
    return {
        "bandit_reward": fb.reward,
        "earned": [*fb.rewards, *[{"reason": r.reason, "label": rewards.LABELS[r.reason], "amount": r.amount} for r in earned]],
        "balance": rewards.balance(s, pid),
    }
