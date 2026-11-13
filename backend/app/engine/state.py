"""State engine – docs/plan.md §4.

z_i = s_i * (x_i - mu_i) / sigma_i over a personal baseline window,
D_t = sum_i w_i z_i,  S_t = alpha * D_t + (1 - alpha) * S_{t-1}.
"""

import json
import statistics
from pathlib import Path

# +1 = higher value is worse, -1 = higher value is better
DIRECTION: dict[str, int] = {
    "speech_rate": -1, "f0_std": -1, "pause_ratio": +1, "energy": -1,
    "arousal": -1, "valence": -1,
    "mood": -1, "energy_self": -1, "anhedonia": +1, "social_contact": -1,
    "sleep_hours": -1, "hrv_rmssd": -1, "resting_hr": +1, "steps": -1,
}

WEIGHTS_FILE = Path(__file__).with_name("weights.json")


def load_weights() -> dict[str, float]:
    """Weights from ml/train_weights.py; fallback = equal weights."""
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text())["weights"]
    return {k: 1.0 / len(DIRECTION) for k in DIRECTION}


def zscores(today: dict[str, float], history: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, x in today.items():
        hist = history.get(name, [])
        if len(hist) < 3:
            continue  # TODO: population norms during first week
        mu, sd = statistics.fmean(hist), statistics.pstdev(hist) or 1.0
        out[name] = DIRECTION.get(name, 1) * (x - mu) / sd
    return out


def daily_deviation(z: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in z.items())


def ewma(d_t: float, s_prev: float | None, alpha: float) -> float:
    return d_t if s_prev is None else alpha * d_t + (1 - alpha) * s_prev


def top_contributors(z: dict[str, float], weights: dict[str, float], n: int = 3) -> list[dict]:
    contrib = sorted(((k, weights.get(k, 0.0) * v) for k, v in z.items()), key=lambda t: -t[1])
    return [{"feature": k, "contribution": round(c, 3)} for k, c in contrib[:n]]
