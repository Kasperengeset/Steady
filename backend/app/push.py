"""Web Push for the daily prompt (issue #22, P2, STATUS.md B23).

When today's prompt fires, every device the person subscribed gets "Time for your check-in".

- Off unless VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY and VAPID_SUBJECT are set in .env (never in git;
  create them with `python scripts/gen_vapid.py`). Without them the app shows the prompt when opened.
- The notification carries no health data: it shows on a lock screen. Fixed text only.
- Sent once per prompt (DailyPrompt.pushed_at), only within the on-time window, never in quiet hours,
  never when paused or off. Subscriptions the push service reports as gone (404/410) are deleted.
- iPhone: the PWA must be added to the home screen first (iOS 16.4 or later).
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from . import prompt
from .config import settings
from .models_prompt import DailyPrompt
from .models_push import PushSubscription

log = logging.getLogger(__name__)

PROMPT_PAYLOAD = {
    "title": "Time for your check-in",
    "body": "Read the phrase, then tell me about your day.",
    "url": "/",
    "tag": "daily-prompt",  # a newer prompt replaces an older one instead of stacking
}
TEST_PAYLOAD = {"title": "Notifications are on", "body": "Your daily check-in will arrive here.", "url": "/", "tag": "test"}
TTL_S = prompt.WINDOW_MIN * 60  # a push that waited longer than the on-time window is not worth showing
LOOP_S = 30


def enabled() -> bool:
    return bool(settings.vapid_public_key and settings.vapid_private_key and settings.vapid_subject)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def subscribe(s: Session, patient_id: int, endpoint: str, p256dh: str, auth: str) -> PushSubscription:
    sub = s.exec(select(PushSubscription).where(PushSubscription.endpoint == endpoint)).first()
    sub = sub or PushSubscription(patient_id=patient_id, endpoint=endpoint, p256dh=p256dh, auth=auth)
    sub.patient_id, sub.p256dh, sub.auth = patient_id, p256dh, auth
    s.add(sub)
    s.flush()
    return sub


def unsubscribe(s: Session, endpoint: str) -> bool:
    sub = s.exec(select(PushSubscription).where(PushSubscription.endpoint == endpoint)).first()
    if sub is not None:
        s.delete(sub)
    return sub is not None


def _webpush(sub: PushSubscription, payload: dict) -> None:
    """One push. Separate function so tests replace it (no network in tests)."""
    from pywebpush import webpush

    webpush(
        subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
        data=json.dumps(payload),
        vapid_private_key=settings.vapid_private_key,
        vapid_claims={"sub": settings.vapid_subject},
        ttl=TTL_S,
        timeout=10,
    )


def _gone(e: Exception) -> bool:
    status = getattr(getattr(e, "response", None), "status_code", None)
    return status in (404, 410)


def notify(s: Session, patient_id: int, payload: dict) -> int:
    """Push to every device of one person. Returns how many accepted it."""
    sent = 0
    for sub in s.exec(select(PushSubscription).where(PushSubscription.patient_id == patient_id)).all():
        try:
            _webpush(sub, payload)
            sent += 1
        except Exception as e:  # noqa: BLE001 – one dead device must not stop the others
            if _gone(e):
                s.delete(sub)
            else:
                log.warning("push failed (%s): %s", type(e).__name__, str(e)[:200])
    return sent


def deliver_due(s: Session, now: datetime, patient_id: int | None = None) -> int:
    """Send today's prompt to everyone whose prompt has fired and who has not been pushed yet."""
    if not enabled():
        return 0
    q = select(PushSubscription.patient_id).distinct()
    if patient_id is not None:
        q = q.where(PushSubscription.patient_id == patient_id)
    sent = 0
    for pid in s.exec(q).all():
        p: DailyPrompt | None = prompt.todays(s, pid, now)
        if p is None or p.answered_at is not None or p.pushed_at is not None:
            continue
        at = _aware(p.at)
        if now < at or now - at > timedelta(minutes=prompt.WINDOW_MIN) or prompt.is_quiet(s, pid, now):
            continue
        if notify(s, pid, PROMPT_PAYLOAD):
            sent += 1
        p.pushed_at = now  # once per prompt, even if every device failed (no retry storm)
        s.add(p)
    s.flush()
    return sent


def run_loop(stop: threading.Event, engine, clock) -> None:
    """Background sender (started in main.py when push is configured)."""
    while not stop.wait(LOOP_S):
        try:
            with Session(engine) as s:
                deliver_due(s, clock())
                s.commit()
        except Exception as e:  # noqa: BLE001 – keep the loop alive
            log.warning("push loop: %s: %s", type(e).__name__, str(e)[:200])
