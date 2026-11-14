"""Patient-bound Google Health connections and idempotent daily ingestion.

OAuth tokens stay in ignored local data files, never API responses. The demo path is
explicit and does not pretend to be a connected watch. No historical state is rewritten.
"""

import json
import logging
import secrets
import threading
from datetime import date, datetime, timedelta, timezone

from fastapi import HTTPException
from sqlmodel import Field, Session, SQLModel, UniqueConstraint, select

from . import baseline
from .config import DATA_DIR, settings
from .models import Patient, Slot
from .pipeline import extract_wearable_features
from .pipeline.google_health import GoogleHealth, SCOPES
from .schemas import WearableDayIn

log = logging.getLogger(__name__)
TOKEN_DIR = DATA_DIR / "health_tokens"
_locks: dict[int, threading.RLock] = {}
_locks_guard = threading.Lock()


class HealthConnection(SQLModel, table=True):
    patient_id: int = Field(foreign_key="patient.id", primary_key=True)
    last_sync_at: datetime | None = None
    last_day: date | None = None
    source: str | None = None
    error: str | None = None


class HealthSyncDay(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("patient_id", "date", name="uq_health_sync_day"),)
    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    date: date
    source: str


def patient_lock(pid: int) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(pid, threading.RLock())


def _token_path(pid: int):
    return TOKEN_DIR / f"{int(pid)}.json"


def _connection(s: Session, pid: int) -> HealthConnection:
    row = s.get(HealthConnection, pid)
    if row is None:
        row = HealthConnection(patient_id=pid)
        s.add(row)
        s.flush()
    return row


def status(s: Session, pid: int) -> dict:
    row = s.get(HealthConnection, pid)
    days = s.exec(select(HealthSyncDay.id).where(HealthSyncDay.patient_id == pid)).all()
    return {
        "connected": _token_path(pid).exists(),
        "configured": bool(settings.google_client_id and settings.google_client_secret),
        "last_sync_at": _iso(row.last_sync_at) if row else None,
        "last_day": row.last_day.isoformat() if row and row.last_day else None,
        "source": row.source if row else None,
        "days_synced": len(days),
        "error": row.error if row else None,
    }


def _iso(ts: datetime | None):
    return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).isoformat() if ts else None


def save_credentials(pid: int, credentials) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    path = _token_path(pid)
    temporary = path.with_suffix(f".{secrets.token_hex(6)}.tmp")
    temporary.write_text(credentials.to_json(), encoding="utf-8")
    temporary.replace(path)


def authorized_client(pid: int):
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    path = _token_path(pid)
    if not path.exists():
        raise HTTPException(409, "Connect your Google Health account first.")
    try:
        credentials = Credentials.from_authorized_user_file(str(path), SCOPES)
        if not credentials.valid:
            credentials.refresh(Request())
            save_credentials(pid, credentials)
    except Exception as exc:
        raise HTTPException(409, "Google Health sign-in expired. Connect again.") from exc
    return GoogleHealth(AuthorizedSession(credentials))


def ingest(s: Session, pid: int, days: dict[date, dict], source: str) -> dict:
    from .routers.checkins import _record_checkin

    patient = s.get(Patient, pid)
    if patient is None:
        raise HTTPException(404, "patient not found")
    now = baseline.clock()
    today = now.date()
    existing = set(s.exec(select(HealthSyncDay.date).where(HealthSyncDay.patient_id == pid)).all())
    count, skipped, rewards = 0, 0, []
    warnings = []
    completed = [day for day in days if day < today]
    newest_completed = max(completed) if completed else None
    for day, values in sorted(days.items()):
        if day != newest_completed or day in existing:
            skipped += 1
            continue
        values = WearableDayIn(patient_id=pid, **values).model_dump(
            exclude={"patient_id", "slot"}, exclude_none=True,
        )
        result = extract_wearable_features(**values)
        if not result.features:
            skipped += 1
            continue
        result.llm_json = json.dumps({"wearable": {"source": source,
            "measurement_date": day.isoformat(), "received_at": _iso(now)}})
        minutes = values.get("minute_steps")
        extra = ({"active_minutes": values["active_minutes"]} if values.get("active_minutes") is not None else
                 {"active_minutes": sum(m >= 100 for m in minutes)} if minutes else {})
        # A day is immutable after ingestion: repeated syncs cannot inflate scores or food.
        s.add(HealthSyncDay(patient_id=pid, date=day, source=source))
        s.flush()
        # Observe a complete measurement day now. Never freeze today's partial steps,
        # backdate an EWMA observation, or append a week of old days as today's signal.
        out = _record_checkin(s, patient, Slot.morning, result, extra, recorded_at=now,
                             data_origin="synthetic" if source == "demo" else "live")
        rewards.extend(out.rewards)
        existing.add(day)
        count += 1
    row = _connection(s, pid)
    row.last_sync_at, row.error = now, None
    if count:
        newest = newest_completed
        row.last_day, row.source = newest, source
    s.add(row)
    s.commit()
    if skipped:
        warnings.append("Only the latest completed wearable day is added. Previously synced days and earlier history stay unchanged.")
    return {"days_synced": count, "skipped": skipped, "rewards": rewards,
            "warnings": warnings, "status": status(s, pid)}


def sync(s: Session, pid: int) -> dict:
    with patient_lock(pid):
        try:
            client = authorized_client(pid)
            client.identity()
            end = baseline.clock().date()
            days = client.fetch_days(end - timedelta(days=7), end)
            if not days:
                raise HTTPException(409, "No wearable data is available yet. Sync your Fitbit, then try again.")
            result = ingest(s, pid, days, "google_health")
            if client.errors:
                result["warnings"].append("Some health measurements were unavailable. Available measurements were synced.")
            return result
        except Exception as exc:
            s.rollback()
            message = exc.detail if isinstance(exc, HTTPException) else "Google Health could not sync. Check your connection and try again."
            row = _connection(s, pid)
            row.error = str(message)
            s.add(row)
            s.commit()
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(502, message) from exc


def demo(s: Session, pid: int) -> dict:
    with patient_lock(pid):
        if _token_path(pid).exists():
            raise HTTPException(409, "Disconnect Google Health before trying sample data.")
        # Explicit sample numbers. The badge and sync source remain visible in the app.
        minutes = [0.0] * 1440
        for i in range(24):
            minutes[600 + i] = 110.0
        for i in range(400):
            minutes[900 + i] = 13.4
        return ingest(s, pid, {baseline.clock().date() - timedelta(days=1): {
            "minute_steps": minutes, "sleep_hours": 7.5, "resting_hr": 63, "hrv_rmssd": 43,
        }}, "demo")


def disconnect(s: Session, pid: int) -> None:
    with patient_lock(pid):
        _token_path(pid).unlink(missing_ok=True)
        row = s.get(HealthConnection, pid)
        if row:
            row.error = None
            s.add(row)
            s.commit()


def sync_connected_patients(engine) -> None:
    """Called hourly by the lifespan worker. Each connected account syncs once per UTC day."""
    if not TOKEN_DIR.exists():
        return
    today = baseline.clock().date()
    for path in TOKEN_DIR.glob("*.json"):
        if not path.stem.isdigit():
            continue
        pid = int(path.stem)
        with Session(engine) as session:
            if session.get(Patient, pid) is None:
                continue
            row = session.get(HealthConnection, pid)
            if (row and row.last_sync_at and row.last_sync_at.date() == today
                    and row.last_day and row.last_day >= today - timedelta(days=1)):
                continue
            try:
                sync(session, pid)
            except Exception:
                log.warning("Daily wearable sync unavailable for patient %s; reconnect or retry in Habits.", pid)
