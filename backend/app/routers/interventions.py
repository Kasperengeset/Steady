from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

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
    # TODO(B, lør): state_improved = S_{t+1} < S_t, needs next day's score (delayed reward)
    r = bandit.reward(body.completed, body.rating, state_improved=None)
    log.completed, log.rating, log.reward = body.completed, body.rating, r
    arm = s.exec(select(Arm).where(Arm.patient_id == body.patient_id, Arm.key == log.arm_key)).one()
    bandit.update(arm, r)
    s.add_all([log, arm])
    s.commit()
    return FeedbackOut(reward=r)
