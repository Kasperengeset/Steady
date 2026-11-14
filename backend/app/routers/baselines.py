"""Healthy-baseline management for clinicians (STATUS.md T1)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from .. import baseline
from ..db import get_session
from ..models import BaselineStat, BaselineStatus, Patient
from ..schemas import BaselineConfirmIn, BaselineDetail, BaselineOut, BaselineStartIn

router = APIRouter(tags=["baseline"])


def _patient(s: Session, pid: int) -> Patient:
    p = s.get(Patient, pid)
    if p is None:
        raise HTTPException(404, "patient not found")
    return p


def _detail(s: Session, pid: int) -> BaselineDetail:
    active = baseline.latest(s, pid, BaselineStatus.active)
    building = baseline.latest(s, pid, BaselineStatus.building)
    stats: dict[str, dict[str, dict[str, float]]] = {}
    if active:
        for r in s.exec(select(BaselineStat).where(BaselineStat.baseline_id == active.id)):
            stats.setdefault(r.slot, {})[r.feature] = {"center": r.center, "scale": r.scale, "n": r.n}
    return BaselineDetail(
        active=BaselineOut(**baseline.progress(s, active)) if active else None,
        building=BaselineOut(**baseline.progress(s, building)) if building else None,
        stats=stats,
    )


@router.get("/patients/{pid}/baseline", response_model=BaselineDetail)
def get_baseline(pid: int, s: Session = Depends(get_session)) -> BaselineDetail:
    _patient(s, pid)
    return _detail(s, pid)


@router.post("/patients/{pid}/baseline/start", response_model=BaselineDetail)
def start_baseline(pid: int, body: BaselineStartIn, s: Session = Depends(get_session)) -> BaselineDetail:
    """Start a new healthy period, e.g. after TMS ends, after a new phone, or after a PHQ-9 confirms
    remission. The current active baseline keeps scoring until the new one locks."""
    _patient(s, pid)
    baseline.start(s, pid, period_days=body.period_days, min_checkins=body.min_checkins, note=body.note)
    s.commit()
    return _detail(s, pid)


@router.post("/patients/{pid}/baseline/confirm", response_model=BaselineDetail)
def confirm_baseline(pid: int, body: BaselineConfirmIn, s: Session = Depends(get_session)) -> BaselineDetail:
    """Clinician confirms the patient was in remission during the (active or building) period."""
    _patient(s, pid)
    b = baseline.latest(s, pid, BaselineStatus.building) or baseline.latest(s, pid, BaselineStatus.active)
    if b is None:
        raise HTTPException(404, "no baseline")
    b.confirmed_by, b.note = body.clinician, body.note or b.note
    s.add(b)
    s.commit()
    return _detail(s, pid)


@router.post("/patients/{pid}/baseline/reject", response_model=BaselineDetail)
def reject_baseline(pid: int, body: BaselineStartIn, s: Session = Depends(get_session)) -> BaselineDetail:
    """Clinician: the period was not representative of the healthy state → reject it and start over."""
    _patient(s, pid)
    b = baseline.latest(s, pid, BaselineStatus.building) or baseline.latest(s, pid, BaselineStatus.active)
    if b is None:
        raise HTTPException(404, "no baseline")
    baseline.reject(s, b, note=body.note)
    s.commit()
    return _detail(s, pid)
