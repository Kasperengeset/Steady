"""Shared fixtures for the engagement tests (#24 rewards, #26 circle, #25 companion)."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import baseline as baseline_mod
from app import db
from app.main import app
from app.seed import seed

DIARY = Path(__file__).resolve().parents[2] / "ml" / "text" / "diary_sanity.csv"
ENTRIES = {r["id"]: r["text"] for r in csv.DictReader(open(DIARY, encoding="utf-8")) if r["variant"] == "as written"}


@pytest.fixture()
def client(monkeypatch):
    """Fresh DB, patient 1 (Mia), and a test clock: `c.day` and `c.minutes` move time forward."""
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.day, c.minutes = 0, 0
    # Noon UTC, so local days and UTC days agree whatever time the tests run
    start = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    monkeypatch.setattr(baseline_mod, "clock", lambda: start + timedelta(days=c.day, minutes=c.minutes))
    c.now = lambda: start + timedelta(days=c.day, minutes=c.minutes)
    return c


def diary(c, text=None, patient_id=1, slot="evening"):
    r = c.post("/api/checkins/text", json={"patient_id": patient_id, "slot": slot, "text": text or ENTRIES["good_1"]})
    assert r.status_code == 200, r.text
    return r.json()


def checkin_on_days(c, days, text=None):
    """One written check-in on each listed day (relative to day 0)."""
    for d in days:
        c.day = d
        diary(c, text)
