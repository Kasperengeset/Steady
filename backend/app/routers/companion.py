"""The avatar's kind check-ins (#25). Rules in engine/companion.py, lines in companion_lines.json."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from .. import baseline, rewards, safety
from ..db import get_session
from ..engine import companion
from ..models_social import CompanionMessage

router = APIRouter(tags=["companion"])


class ReplyIn(BaseModel):
    message_id: int
    text: str | None = None  # checked for crisis words, never stored


def _out(msg: CompanionMessage | None, glad: bool) -> dict:
    """{message, kind, at} is the contract with the avatar screens (#23); the rest is extra."""
    if msg is None:
        return {"message": None, "kind": None, "at": None, "id": None, "offer_written": False,
                "offer_action": False, "glad_to_see_you": glad}
    line = next((x for x in companion.LINES.get(msg.kind, []) if x["id"] == msg.line_id), {})
    return {
        "message": msg.text, "kind": msg.kind, "at": rewards.aware(msg.ts).isoformat(), "id": msg.id,
        "offer_written": msg.offer_written, "offer_action": bool(line.get("offer_action")),
        "glad_to_see_you": glad,
    }


@router.get("/patients/{pid}/companion")
def get_companion(pid: int, s: Session = Depends(get_session)) -> dict:
    """At most one message a day. Proactive ones appear from the day's prompt time on."""
    rewards.require_patient(s, pid)
    now = baseline.clock()
    msg = companion.evaluate(s, pid, now)
    out = _out(msg, companion.glad_to_see_you(s, pid, now))
    s.commit()
    return out


@router.post("/patients/{pid}/companion/demo")
def demo_companion(pid: int, s: Session = Depends(get_session)) -> dict:
    """Demo: evaluate now without waiting for the prompt time (replaces today's message)."""
    rewards.require_patient(s, pid)
    now = baseline.clock()
    msg = companion.evaluate(s, pid, now, force=True)
    out = _out(msg, companion.glad_to_see_you(s, pid, now))
    s.commit()
    return out


@router.post("/patients/{pid}/companion/seen")
def seen(pid: int, body: ReplyIn, s: Session = Depends(get_session)) -> dict:
    msg = s.get(CompanionMessage, body.message_id)
    if msg is None or msg.patient_id != pid:
        raise HTTPException(404, "message not found")
    msg.seen_at = msg.seen_at or baseline.clock()
    s.add(msg)
    s.commit()
    return {"ok": True}


@router.post("/patients/{pid}/companion/reply")
def reply(pid: int, body: ReplyIn, s: Session = Depends(get_session)) -> dict:
    """A reply to the avatar. Crisis words run the crisis path: the clinician at once, the support screen opens."""
    patient = rewards.require_patient(s, pid)
    msg = s.get(CompanionMessage, body.message_id)
    if msg is None or msg.patient_id != pid:
        raise HTTPException(404, "message not found")
    crisis = safety.check(s, patient, body.text, where="a reply to the avatar")
    msg.replied_at = baseline.clock()
    s.add(msg)
    s.commit()
    return {"ok": True, "crisis": crisis}


@router.get("/patients/{pid}/companion/context")
def clinician_context(pid: int, s: Session = Depends(get_session)) -> dict:
    """For the clinician portal: context, not an alert. {avatar_checked_in, days_without_answer, note, ...}"""
    rewards.require_patient(s, pid)
    return companion.clinician_context(s, pid, baseline.clock())
