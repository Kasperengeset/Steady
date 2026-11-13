from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..db import get_session
from ..models import Patient
from ..schemas import FriendHint

router = APIRouter(tags=["friends"])


@router.get("/friends/{patient_id}/hint", response_model=FriendHint)
def hint(patient_id: int, s: Session = Depends(get_session)) -> FriendHint:
    """Friends only ever get a nudge – never scores, features or transcripts."""
    p = s.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404, "patient not found")
    if p.risk_level >= 1:
        return FriendHint(show_hint=True, message=f"Kanskje en god dag å ta kontakt med {p.name}?")
    return FriendHint(show_hint=False, message="")
