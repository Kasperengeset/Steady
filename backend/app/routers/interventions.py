from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import update
from sqlmodel import Session, select

from .. import baseline, rewards
from ..db import get_session
from ..engine import bandit
from ..models import Arm, InterventionLog
from ..schemas import FeedbackIn, FeedbackOut

router = APIRouter(tags=["interventions"])


@router.post("/interventions/feedback", response_model=FeedbackOut)
def feedback(body: FeedbackIn, s: Session = Depends(get_session)) -> FeedbackOut:
    log = s.exec(
        select(InterventionLog).where(
            InterventionLog.patient_id == body.patient_id, InterventionLog.date == body.date
        )
    ).first()
    if log is None:
        raise HTTPException(404, "no intervention for that day")
    s.execute(update(InterventionLog).where(InterventionLog.id == log.id).values(arm_key=InterventionLog.arm_key))
    s.refresh(log)
    # TODO: state_improved = S_{t+1} < S_t, needs next day's score (delayed reward)
    r = bandit.reward(body.completed, body.rating, state_improved=None)
    arm = s.exec(select(Arm).where(Arm.patient_id == body.patient_id, Arm.key == log.arm_key)).one()
    if log.reward is not None:
        # A retry or corrected answer replaces this day's evidence, not another trial.
        arm.a = max(1.0, arm.a - log.reward)
        arm.b = max(1.0, arm.b - (1 - log.reward))
    log.completed, log.rating, log.reward = body.completed, body.rating, r
    bandit.update(arm, r)
    s.add_all([log, arm])
    earned = [{"reason": row.reason, "label": rewards.LABELS.get(row.reason, row.reason), "amount": row.amount}
              for row in rewards.sync(s, body.patient_id, baseline.clock())]
    s.commit()
    return FeedbackOut(reward=r, rewards=earned)
