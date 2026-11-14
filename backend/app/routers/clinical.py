"""Desktop clinician inspection: measured entries, exact saved analysis and model limitations."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from .. import clinical
from ..db import get_session
from ..models import CheckIn, Patient

router = APIRouter(tags=["clinical-workspace"])


def _patient(s: Session, pid: int) -> Patient:
    patient = s.get(Patient, pid)
    if patient is None:
        raise HTTPException(404, "patient not found")
    return patient


@router.get("/clinician/overview")
def overview(s: Session = Depends(get_session)) -> dict:
    return clinical.overview(s)


@router.get("/clinician/patients/{pid}/workspace")
def workspace(pid: int, days: int = Query(30, ge=7, le=120), s: Session = Depends(get_session)) -> dict:
    return clinical.workspace(s, _patient(s, pid), days)


@router.get("/clinician/patients/{pid}/entries")
def entries(pid: int, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
            s: Session = Depends(get_session)) -> dict:
    _patient(s, pid)
    return clinical.entries(s, pid, limit, offset)


@router.get("/clinician/entries/{checkin_id}")
def entry(checkin_id: int, s: Session = Depends(get_session)) -> dict:
    row = s.get(CheckIn, checkin_id)
    if row is None:
        raise HTTPException(404, "check-in not found")
    return clinical.entry_detail(s, row)
