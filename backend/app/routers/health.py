"""Wearable connection, OAuth callback, explicit demo data, and sync controls."""

import base64
import hashlib
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlmodel import Session

from .. import health_sync
from ..config import settings
from ..db import get_session
from ..models import Patient
from ..pipeline.google_health import SCOPES, GoogleHealth

router = APIRouter(tags=["wearables"])
_pending: dict[str, tuple[int, str, float]] = {}
_guard = threading.Lock()


def _patient(s: Session, pid: int) -> None:
    if s.get(Patient, pid) is None:
        raise HTTPException(404, "patient not found")


@router.get("/patients/{pid}/health")
def status(pid: int, s: Session = Depends(get_session)):
    _patient(s, pid)
    return health_sync.status(s, pid)


@router.post("/patients/{pid}/health/connect")
def connect(pid: int, s: Session = Depends(get_session)):
    _patient(s, pid)
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(409, "Google Health is not configured. Add the OAuth client in the backend environment.")
    state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(64)
    with _guard:
        for key, (_, _, expires) in list(_pending.items()):
            if expires <= time.monotonic():
                del _pending[key]
        _pending[state] = (pid, verifier, time.monotonic() + 600)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return {"authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
        "client_id": settings.google_client_id, "redirect_uri": settings.google_redirect_uri,
        "response_type": "code", "scope": " ".join(SCOPES), "state": state,
        "access_type": "offline", "prompt": "consent", "code_challenge": challenge,
        "code_challenge_method": "S256",
    })}


@router.get("/google-health/callback")
def callback(state: str = "", code: str = "", error: str = "", s: Session = Depends(get_session)):
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    with _guard:
        pending = _pending.pop(state, None)
    if pending is None or pending[2] <= time.monotonic():
        raise HTTPException(400, "This connection request expired. Start again in Habits.")
    pid, verifier, _ = pending
    _patient(s, pid)
    destination = settings.frontend_url.rstrip("/") + "/habits?" + urlencode({"p": pid})
    if error or not code:
        return RedirectResponse(destination + "&health=cancelled", status_code=303)
    try:
        response = requests.post("https://oauth2.googleapis.com/token", data={
            "code": code, "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret, "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code", "code_verifier": verifier,
        }, timeout=30)
        response.raise_for_status()
        payload = response.json()
        credentials = Credentials(
            token=payload["access_token"], refresh_token=payload.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token", client_id=settings.google_client_id,
            client_secret=settings.google_client_secret, scopes=payload.get("scope", " ".join(SCOPES)).split(),
            expiry=(datetime.now(timezone.utc) + timedelta(seconds=int(payload.get("expires_in", 3600)))).replace(tzinfo=None),
        )
        # Confirm the Google account is linked before marking it connected.
        GoogleHealth(AuthorizedSession(credentials)).identity()
        with health_sync.patient_lock(pid):
            health_sync.save_credentials(pid, credentials)
        return RedirectResponse(destination + "&health=connected", status_code=303)
    except Exception:
        # No raw provider errors, auth codes, or token values in responses/logs.
        return RedirectResponse(destination + "&health=error", status_code=303)


@router.post("/patients/{pid}/health/sync")
def sync(pid: int, s: Session = Depends(get_session)):
    _patient(s, pid)
    return health_sync.sync(s, pid)


@router.post("/patients/{pid}/health/demo")
def demo(pid: int, s: Session = Depends(get_session)):
    _patient(s, pid)
    return health_sync.demo(s, pid)


@router.delete("/patients/{pid}/health/connection")
def disconnect(pid: int, s: Session = Depends(get_session)):
    _patient(s, pid)
    health_sync.disconnect(s, pid)
    return {"ok": True}
