"""Healthy-baseline lifecycle and scoring (STATUS.md T1). Audio analysis is replaced by controlled
feature values, so these tests check the engine, not the signal processing."""

import random

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel

from app import db
from app.engine import state
from app.main import app
from app.models import FeatureSource
from app.pipeline import PipelineResult
from app.routers import checkins as checkins_router
from app.seed import seed

STATS = state.load_branch("acoustic").meta["feature_stats"]
SOURCE = {"arousal": FeatureSource.emotion, "valence": FeatureSource.emotion}


def features(depression: float, rng: random.Random) -> PipelineResult:
    """depression 0 = like corpus controls, 1 = like corpus patients; plus within-person noise."""
    res = PipelineResult()
    for f, st in STATS.items():
        v = st["control_mean"] + depression * (st["patient_mean"] - st["control_mean"]) + rng.gauss(0, st["within_sd"] * 0.5)
        res.features.append((f, v, SOURCE.get(f, FeatureSource.prosody)))
    return res


@pytest.fixture()
def client(monkeypatch):
    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    queue: list[PipelineResult] = []
    monkeypatch.setattr(checkins_router, "extract_features", lambda path: queue.pop(0))
    c = TestClient(app)
    c.queue = queue
    return c


def checkin(c, res: PipelineResult, slot="morning") -> dict:
    c.queue.append(res)
    r = c.post("/api/checkins", data={"patient_id": 1, "slot": slot}, files={"audio": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- pure statistics

def test_robust_stats_ignores_one_outlier():
    vals = [1.0, 1.1, 0.9, 1.05, 0.95, 50.0]
    center, scale = state.robust_stats(vals, prior_sd=None, k=5)
    assert 0.9 < center < 1.1
    assert scale < 1.0  # mean/SD would give ~18


def test_robust_stats_shrinks_toward_corpus_sd():
    vals = [1.0] * 4  # MAD = 0 → personal SD unknowable from 4 identical values
    _, scale = state.robust_stats(vals, prior_sd=2.0, k=5)
    assert scale == pytest.approx((5 * 4 / 9) ** 0.5)  # (n·0 + k·4) / (n + k)
    _, big_n = state.robust_stats([1.0, 3.0] * 200, prior_sd=2.0, k=5)
    assert big_n == pytest.approx(1.4826, rel=0.05)  # personal data dominates with many check-ins


# ---------------------------------------------------------------- lifecycle via the API

def test_first_checkin_starts_building_and_gives_no_score(client):
    out = checkin(client, features(0, random.Random(0)))
    assert out["baseline"]["status"] == "building"
    assert out["baseline"]["period_days"] == 14
    assert out["state"]["S"] is None


def test_locks_after_period_then_scores_and_catches_change(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    rng = random.Random(1)
    for _ in range(5):
        out = checkin(client, features(0, rng))
        assert out["baseline"]["status"] == "building" and out["state"]["S"] is None
    out = checkin(client, features(0, rng))  # 6th check-in → enough → locks
    assert out["baseline"]["status"] == "active"
    assert out["state"]["S_anchored"] is not None and abs(out["state"]["S_anchored"]) < 2
    assert out["state"]["S_rolling"] is None  # needs rolling_min_n same-slot check-ins first

    healthy_S = out["state"]["S_anchored"]
    out = checkin(client, features(1.0, rng))  # a full control→patient shift
    # Detectability on Androids is ~1.6 SD for a full shift (ml/reports/acoustic/summary.md)
    assert out["state"]["S_anchored"] > 1.2
    assert out["state"]["S_anchored"] - healthy_S > 1.0
    assert out["state"]["driver"] == "anchored"

    out = checkin(client, features(2.0, rng))  # far beyond the corpus difference → S > 2 → clinician
    assert out["state"]["S_anchored"] > 2
    assert out["state"]["level"] >= 2

    detail = client.get("/api/patients/1/baseline").json()
    assert detail["active"]["version"] == 1
    assert {"all", "morning"} <= set(detail["stats"])  # evening has too few → pooled fallback
    assert "evening" not in detail["stats"]


def test_new_baseline_keeps_old_one_active_until_it_locks(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    rng = random.Random(2)
    for _ in range(3):
        checkin(client, features(0, rng))
    client.post("/api/patients/1/baseline/start", json={"period_days": 14, "min_checkins": 3, "note": "new phone"})
    d = client.get("/api/patients/1/baseline").json()
    assert d["active"]["version"] == 1 and d["building"]["version"] == 2
    out = checkin(client, features(0, rng))
    assert out["state"]["S_anchored"] is not None  # still scored against v1


def test_clinician_reject_starts_over_and_confirm_is_recorded(client):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 3})
    rng = random.Random(3)
    for _ in range(3):
        checkin(client, features(0, rng))
    d = client.post("/api/patients/1/baseline/confirm", json={"clinician": "Dr. Virtanen"}).json()
    assert d["active"]["confirmed_by"] == "Dr. Virtanen"
    d = client.post("/api/patients/1/baseline/reject", json={"note": "had flu"}).json()
    assert d["active"] is None and d["building"]["version"] == 2
    out = checkin(client, features(0, rng))
    assert out["state"]["S"] is None  # no active baseline → no score until the new one locks
