"""Explanation of S (engine/explain.py, B20) and the clinician-only elevated flag (B21)."""

import random

import pytest

from app.engine import explain, state
from tests.test_baseline import checkin, client, features  # noqa: F401  (client is a fixture)


def test_feature_shares_sum_to_D():
    m = state.load_branch("acoustic")
    z = {"arousal": 2.6, "valence": 2.6, "energy": 1.0, "jitter": -1.1, "shimmer": -1.2, "f0_std": -1.5,
         "speech_rate": 0.5, "pause_ratio": 0.3}
    sh = state.feature_shares(z, m)
    assert sum(sh.values()) == pytest.approx(state.branch_deviation(z, m))
    assert max(sh, key=sh.get) in {"arousal", "valence"}


def test_explain_splits_S_exactly_over_days():
    alpha, Ds = 0.3, [0.2, -0.4, 1.1, 1.6, 1.3]
    S = None
    for d in Ds:
        S = state.ewma(d, S, alpha)
    days = [{"date": f"d{i}", "D": d, "features": {"arousal": d * 0.6, "energy": d * 0.4}} for i, d in enumerate(Ds)]
    out = explain.explain(days, S, alpha)
    assert sum(x["contribution"] for x in out["by_day"]) + out["older"] == pytest.approx(S, abs=1e-3)
    assert out["older"] == pytest.approx(0, abs=1e-3)  # whole history inside the window
    assert sum(out["by_feature"].values()) == pytest.approx(S, abs=1e-3)
    out2 = explain.explain(days, S, alpha, window=2)  # only the last two days shown
    assert len(out2["by_day"]) == 2
    assert sum(x["contribution"] for x in out2["by_day"]) + out2["older"] == pytest.approx(S, abs=1e-3)


def _lock(client, rng):
    client.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": 6})
    for _ in range(6):
        out = checkin(client, features(0, rng))
    assert out["baseline"]["status"] == "active"


def test_api_explains_a_sustained_shift(client):
    rng = random.Random(11)
    _lock(client, rng)
    for _ in range(5):
        out = checkin(client, features(1.2, rng))
    why = out["state"]["explanation"]
    assert why is not None and why["track"] == out["state"]["driver"]
    assert sum(x["contribution"] for x in why["by_day"]) + why["older"] == pytest.approx(out["state"]["S"], abs=1e-3)
    top2 = list(why["by_feature"])[:2]
    assert set(top2) & {"arousal", "energy", "valence"}  # the features that shift most with depression
    assert why["sentence"].startswith("Compared with the healthy normal")
    detail = client.get("/api/patients/1").json()
    assert detail["explanation"]["by_feature"] == why["by_feature"]


def test_elevated_streak_flags_clinician_only(client):
    rng = random.Random(12)
    _lock(client, rng)
    for _ in range(4):
        out = checkin(client, features(-2.0, rng))  # far "better" than the healthy normal
    assert out["state"]["elevated"] is True
    assert out["state"]["level"] == 0  # no friend or clinician alert from the elevated flag
    assert client.get("/api/friends/1/hint").json()["show_hint"] is False
    assert client.get("/api/patients").json()[0]["elevated"] is True
