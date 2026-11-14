"""Daily prompt endpoints (issue #22). Logic in app/prompt.py."""

from collections.abc import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from .. import baseline, prompt
from ..db import get_session
from ..models import Patient
from ..schemas_prompt import PromptOut, PromptSettingsIn

router = APIRouter(tags=["prompt"])


def _now():
    """Shared clock in baseline.py (tests replace it)."""
    return baseline.clock()


def _patient(s: Session, pid: int) -> None:
    if s.get(Patient, pid) is None:
        raise HTTPException(404, "patient not found")


def _retry(s: Session, fn: Callable[[], dict]) -> dict:
    """Two first requests of the day can draw at once (e.g. React StrictMode): the unique (patient, date)
    rejects the second insert – roll back and read the one that won."""
    try:
        out = fn()
        s.commit()
    except IntegrityError:
        s.rollback()
        out = fn()
        s.commit()
    return out


@router.get("/patients/{pid}/prompt", response_model=PromptOut)
def get_prompt(pid: int, s: Session = Depends(get_session)) -> dict:
    _patient(s, pid)
    return _retry(s, lambda: prompt.status(s, pid, _now()))


@router.post("/patients/{pid}/prompt/now", response_model=PromptOut)
def send_prompt_now(pid: int, s: Session = Depends(get_session)) -> dict:
    """Demo: "Send today's prompt now", so the video does not wait for a random time."""
    _patient(s, pid)
    now = _now()
    st = prompt.settings_for(s, pid)
    if not st.enabled or st.paused_on == prompt.local_now(s, pid, now).date():
        raise HTTPException(409, "prompts are off or paused today")

    def run() -> dict:
        prompt.send_now(s, pid, now)
        return prompt.status(s, pid, now)

    return _retry(s, run)


@router.put("/patients/{pid}/prompt/settings", response_model=PromptOut)
def update_prompt_settings(pid: int, body: PromptSettingsIn, s: Session = Depends(get_session)) -> dict:
    """Quiet hours, pause for today, turn off. Only the fields sent are changed."""
    _patient(s, pid)
    changes = body.model_dump(exclude_unset=True)
    if changes.get("enabled") is None:
        changes.pop("enabled", None)
    if changes.get("pause_today") is None:
        changes.pop("pause_today", None)
    if "tz" in changes:
        try:
            ZoneInfo(changes["tz"] or "")
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise HTTPException(400, f"unknown time zone: {changes['tz']!r}") from e
    for k in ("quiet_start", "quiet_end"):
        if changes.get(k) is not None:
            try:
                prompt.parse_hm(changes[k])
            except ValueError as e:
                raise HTTPException(400, f"{k}: use HH:MM") from e
    if ("quiet_start" in changes) != ("quiet_end" in changes):
        raise HTTPException(400, "send quiet_start and quiet_end together")
    now = _now()

    def run() -> dict:
        prompt.update_settings(s, pid, now, changes)
        return prompt.status(s, pid, now)

    return _retry(s, run)
