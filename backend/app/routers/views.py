"""Read endpoints for the patient app and the clinician portal (logic in app/insight.py)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from .. import insight
from ..db import get_session
from ..models import Patient

router = APIRouter(tags=["views"])


def _patient(s: Session, pid: int) -> Patient:
    p = s.get(Patient, pid)
    if p is None:
        raise HTTPException(404, "patient not found")
    return p


# ------------------------------------------------------------------ patient app

@router.get("/patients/{pid}/today")
def today(pid: int, s: Session = Depends(get_session)) -> dict:
    """Status sentence, 14-day line, today's action with its reason, pending follow-up, last wearable sync."""
    _patient(s, pid)
    return insight.today(s, pid)


@router.get("/patients/{pid}/insights")
def insights(pid: int, s: Session = Depends(get_session)) -> list[dict]:
    """What helps: actions ranked by what the bandit learned, with done/helped counts and history."""
    _patient(s, pid)
    return insight.insights(s, pid)


@router.get("/patients/{pid}/contacts")
def contacts(pid: int, s: Session = Depends(get_session)) -> list[dict]:
    _patient(s, pid)
    return insight.contacts(s, pid)


# ------------------------------------------------------------------ clinician portal

@router.get("/clinician/notifications")
def notifications(status: str = Query("open", pattern="^(open|handled|all)$"), s: Session = Depends(get_session)) -> list[dict]:
    """Inbox: crisis flags first, then clinic flags, friend nudges and baselines to confirm."""
    return insight.notifications(s, status)


@router.post("/clinician/notifications/{nid}/handled")
def handled(nid: str, s: Session = Depends(get_session)) -> dict:
    if not insight.mark_handled(s, nid):
        raise HTTPException(404, "notification not found (baseline items are handled by confirming the baseline)")
    return {"ok": True}


@router.get("/patients/{pid}/timeline")
def timeline(pid: int, days: int = Query(30, ge=7, le=120), s: Session = Depends(get_session)) -> dict:
    """Day-by-day: score, check-ins, actions done, and the last check-ins in the patient's own words."""
    _patient(s, pid)
    return insight.timeline(s, pid, days)


@router.get("/patients/{pid}/explain")
def explain(pid: int, days: int = Query(30, ge=7, le=120), s: Session = Depends(get_session)) -> dict:
    """'Why is this changing?': every signal vs the healthy baseline, grouped by source, plus words and habits."""
    _patient(s, pid)
    return insight.explain(s, pid, days)
