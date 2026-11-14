import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, col, select

from .. import baseline, explanation
from ..db import get_session
from ..models import BaselineStatus, CheckIn, Patient, StateScore
from ..schemas import PatientDetail, PatientSummary

router = APIRouter(tags=["patients"])


def _scores(s: Session, pid: int) -> list[StateScore]:
    return list(s.exec(select(StateScore).where(StateScore.patient_id == pid).order_by(col(StateScore.date))).all())


@router.get("/patients", response_model=list[PatientSummary])
def list_patients(s: Session = Depends(get_session)) -> list[PatientSummary]:
    out = []
    for p in s.exec(select(Patient)).all():
        scores = _scores(s, p.id)
        b = baseline.latest(s, p.id, BaselineStatus.active) or baseline.latest(s, p.id, BaselineStatus.building)
        out.append(PatientSummary(
            id=p.id, name=p.name, level=p.risk_level,
            S=scores[-1].S if scores else None, trend=[x.S for x in scores[-14:]],
            driver=scores[-1].driver if scores else None,
            baseline_status=b.status.value if b else None,
            elevated=bool(scores and scores[-1].elevated),
        ))
    return sorted(out, key=lambda x: (-x.level, -(x.S or 0)))


@router.get("/patients/{pid}", response_model=PatientDetail)
def get_patient(pid: int, s: Session = Depends(get_session)) -> PatientDetail:
    p = s.get(Patient, pid)
    if p is None:
        raise HTTPException(404, "patient not found")
    scores = _scores(s, pid)
    checkins = s.exec(
        select(CheckIn).where(CheckIn.patient_id == pid).order_by(col(CheckIn.ts).desc()).limit(20)
    ).all()
    return PatientDetail(
        patient=p.model_dump(mode="json"),
        scores=[x.model_dump(mode="json") for x in scores],
        checkins=[c.model_dump(mode="json", exclude={"llm_json", "request_key", "response_json", "analysis_json"}) for c in checkins],
        top_features=json.loads(scores[-1].top_features_json) if scores else [],
        explanation=explanation.latest_explanation(s, pid),
    )
