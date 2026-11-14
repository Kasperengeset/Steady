"""Web Push endpoints (issue #22, P2). Logic in app/push.py."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from .. import push
from ..config import settings
from ..db import get_session
from ..models import Patient
from ..models_push import PushSubscription

router = APIRouter(tags=["push"])


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscriptionIn(BaseModel):
    """The browser's PushSubscription.toJSON()."""

    endpoint: str
    keys: SubscriptionKeys


class EndpointIn(BaseModel):
    endpoint: str


class PushStatusOut(BaseModel):
    enabled: bool  # VAPID keys configured on the server
    public_key: str | None  # applicationServerKey for pushManager.subscribe()
    devices: int | None = None


def _patient(s: Session, pid: int) -> None:
    if s.get(Patient, pid) is None:
        raise HTTPException(404, "patient not found")


def _devices(s: Session, pid: int) -> int:
    return len(s.exec(select(PushSubscription.id).where(PushSubscription.patient_id == pid)).all())


@router.get("/patients/{pid}/push", response_model=PushStatusOut)
def push_status(pid: int, s: Session = Depends(get_session)) -> PushStatusOut:
    _patient(s, pid)
    return PushStatusOut(enabled=push.enabled(), public_key=settings.vapid_public_key or None, devices=_devices(s, pid))


@router.post("/patients/{pid}/push/subscribe", response_model=PushStatusOut)
def subscribe(pid: int, body: SubscriptionIn, s: Session = Depends(get_session)) -> PushStatusOut:
    _patient(s, pid)
    if not push.enabled():
        raise HTTPException(409, "push is not configured on this server")
    if not body.endpoint.startswith("https://"):
        raise HTTPException(400, "endpoint must be https")
    push.subscribe(s, pid, body.endpoint, body.keys.p256dh, body.keys.auth)
    s.commit()
    return PushStatusOut(enabled=True, public_key=settings.vapid_public_key, devices=_devices(s, pid))


@router.post("/patients/{pid}/push/unsubscribe", response_model=PushStatusOut)
def unsubscribe(pid: int, body: EndpointIn, s: Session = Depends(get_session)) -> PushStatusOut:
    _patient(s, pid)
    push.unsubscribe(s, body.endpoint)
    s.commit()
    return PushStatusOut(enabled=push.enabled(), public_key=settings.vapid_public_key or None, devices=_devices(s, pid))


@router.post("/patients/{pid}/push/test")
def test_push(pid: int, s: Session = Depends(get_session)) -> dict:
    """Send "Notifications are on" to this person's devices (settings screen)."""
    _patient(s, pid)
    if not push.enabled():
        raise HTTPException(409, "push is not configured on this server")
    sent = push.notify(s, pid, push.TEST_PAYLOAD)
    s.commit()
    return {"sent": sent}
