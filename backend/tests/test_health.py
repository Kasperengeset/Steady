"""Health branch (pipeline/health.py, weights/health.json) and the wearable endpoint."""

import json
import random
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import baseline as baseline_mod
from app import db
from app.engine import state
from app.main import app
from app.pipeline import health as hp
from app.seed import seed


def minutes_of_day(rng: random.Random, level: float = 40, wake: float = 7.0, bed: float = 23.0) -> list[float]:
    """Synthetic minute steps: walking bursts while awake, zero while asleep."""
    out = []
    for i in range(hp.MINUTES_PER_DAY):
        awake = wake <= i / 60 < bed
        out.append(max(0.0, rng.gauss(level, 15)) if awake and rng.random() < 0.3 else 0.0)
    return out


def healthy_day(rng):
    return {"minute_steps": minutes_of_day(rng), "sleep_hours": rng.gauss(7.5, 0.4),
            "sleep_midpoint": rng.gauss(3.5, 0.3), "resting_hr": rng.gauss(60, 2), "hrv_rmssd": rng.gauss(45, 5)}


def low_day(rng):
    """RADAR-MDD directions: fewer steps, later and shorter sleep, higher resting HR, lower HRV."""
    return {"minute_steps": minutes_of_day(rng, level=20, wake=9.5), "sleep_hours": rng.gauss(6.0, 0.4),
            "sleep_midpoint": rng.gauss(5.5, 0.3), "resting_hr": rng.gauss(66, 2), "hrv_rmssd": rng.gauss(33, 5)}


# ---------------------------------------------------------------- features

def test_activity_features_on_a_known_day():
    m = [0.0] * 1440
    for i in range(8 * 60, 18 * 60):  # 10 steps a minute from 08:00 to 17:59
        m[i] = 10.0
    f = hp.daily_activity_features(m)
    assert f["steps"] == 6000
    assert f["inactive_share"] == pytest.approx((1440 - 600) / 1440)
    assert f["night_activity"] == 0
    assert f["relative_amplitude"] == pytest.approx(1.0)  # silent nights → strongest possible rhythm
    assert f["m10_onset"] == 8


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="1440"):
        hp.daily_activity_features([1.0] * 100)


def test_daily_totals_without_minutes():
    f = hp.wearable_features(steps=8000, sleep_hours=7, resting_hr=61)
    assert f == {"steps": 8000.0, "sleep_hours": 7.0, "resting_hr": 61.0}


def test_weights_file_matches_the_engine():
    model = state.load_branch("health")
    assert model.trained
    assert sum(model.weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(model.weights) == set(hp.FEATURES) == set(state.DIRECTION["health"])
    assert model.direction == hp.DIRECTION
    wc = model.meta["within_corr"]
    assert wc["features"] == list(hp.ACTIVITY_FEATURES) and len(wc["matrix"]) == 5
    for f in model.meta.get("zero_weight", {}):
        assert model.weights[f] == 0


# ---------------------------------------------------------------- wearable days through the engine

@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    c.day = 0
    start = datetime.now(timezone.utc) + timedelta(minutes=1)
    monkeypatch.setattr(baseline_mod, "clock", lambda: start + timedelta(days=c.day))
    return c


def post_day(c, day: dict):
    r = c.post("/api/checkins/wearable", json={"patient_id": 1, **day})
    c.day += 1
    return r


def test_wearable_days_build_baseline_then_flag_a_decline(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 7})
    rng = random.Random(7)
    for _ in range(7):
        r = post_day(client, healthy_day(rng))
        assert r.status_code == 200, r.text
    out = r.json()
    assert out["baseline"]["status"] == "active"
    assert {f["source"] for f in out["features"]} == {"wearable"}
    assert abs(out["state"]["S_anchored"]) < 2

    for _ in range(3):
        out = post_day(client, low_day(rng)).json()
    assert out["state"]["branches"].get("health") is not None
    assert out["state"]["S_anchored"] > 1.5


def test_wearable_rejects_bad_input(client):
    assert post_day(client, {}).status_code == 400  # nothing to score
    assert post_day(client, {"minute_steps": [1.0] * 10}).status_code == 400
    r = client.post("/api/checkins/wearable", json={"patient_id": 99, "steps": 5000})
    assert r.status_code == 404
