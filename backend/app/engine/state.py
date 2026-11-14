"""State engine – docs/plan.md §4, split into three branches.

Per branch b:  z_i = s_i * (x_i - mu_i) / sigma_i   (personal baseline window)
               D_b = sum_i w_i z_i / sqrt(sum_i w_i^2)   (unit variance under no change)
Daily:         D_t = mean of D_b over branches with data today
               S_t = alpha * D_t + (1 - alpha) * S_{t-1}

Weights/directions per branch come from engine/weights/<branch>.json (ml/train_weights.py);
missing file -> literature DIRECTION + equal weights.
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..models import FeatureSource

log = logging.getLogger(__name__)

BRANCHES = ("acoustic", "text", "health")

BRANCH_OF_SOURCE: dict[FeatureSource, str] = {
    FeatureSource.prosody: "acoustic",
    FeatureSource.emotion: "acoustic",
    FeatureSource.llm: "text",
    FeatureSource.text: "text",
    FeatureSource.wearable: "health",
}

# Literature prior: +1 = higher value is worse, -1 = higher value is better
DIRECTION: dict[str, dict[str, int]] = {
    "acoustic": {
        "speech_rate": -1, "f0_std": -1, "pause_ratio": +1, "energy": -1,
        "jitter": +1, "shimmer": +1, "arousal": -1, "valence": -1,
    },
    "text": {"mood": -1, "energy_self": -1, "anhedonia": +1, "social_contact": -1},
    "health": {"sleep_hours": -1, "hrv_rmssd": -1, "resting_hr": +1, "steps": -1},
}

WEIGHTS_DIR = Path(__file__).with_name("weights")


@dataclass
class BranchModel:
    name: str
    weights: dict[str, float]
    direction: dict[str, int]
    trained: bool = False
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=None)
def load_branch(name: str) -> BranchModel:
    prior = DIRECTION[name]
    path = WEIGHTS_DIR / f"{name}.json"
    if not path.exists():
        return BranchModel(name, {k: 1.0 / len(prior) for k in prior}, dict(prior))
    data = json.loads(path.read_text())
    direction = {**prior, **data.get("direction", {})}
    for k, s in data.get("direction", {}).items():
        if k in prior and prior[k] != s:
            log.warning("branch %s: trained direction for %s (%+d) contradicts literature (%+d)", name, k, s, prior[k])
    meta = {k: v for k, v in data.items() if k not in ("weights", "direction")}
    return BranchModel(name, data["weights"], direction, trained=True, meta=meta)


def zscores(today: dict[str, float], history: dict[str, list[float]], direction: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, x in today.items():
        hist = history.get(name, [])
        if len(hist) < 3:
            continue  # TODO: population norms during first week
        mu, sd = statistics.fmean(hist), statistics.pstdev(hist) or 1.0
        out[name] = direction.get(name, 1) * (x - mu) / sd
    return out


def _corr(model: BranchModel, a: str, b: str) -> float:
    """Within-person correlation between two features' direction-adjusted z-scores (0 if unknown)."""
    if a == b:
        return 1.0
    wc = model.meta.get("within_corr")
    if not wc:
        return 0.0
    idx = {f: i for i, f in enumerate(wc["features"])}
    if a not in idx or b not in idx:
        return 0.0
    r = wc["matrix"][idx[a]][idx[b]]
    return r * model.direction.get(a, 1) * model.direction.get(b, 1)


def branch_deviation(z: dict[str, float], model: BranchModel) -> float:
    """Weighted sum of z-scores, rescaled to unit variance under 'no change' (STATUS.md B10, B12).

    Var(sum w*z) = sum_ij w_i w_j R_ij, where R is the within-person correlation of the (direction-
    adjusted) z-scores, measured on Androids and stored in <branch>.json["within_corr"]. Dividing by its
    square root makes D ~ N(0, 1) when nothing changes, so escalation thresholds (S > 1, S > 2) keep a
    fixed meaning even though e.g. energy and arousal move together. Without a correlation matrix the
    features are treated as independent (norm = sqrt(sum w^2)).
    """
    keys = list(z)
    w = {k: model.weights.get(k, 0.0) for k in keys}
    var = sum(w[a] * w[b] * _corr(model, a, b) for a in keys for b in keys)
    if var <= 0:
        return 0.0
    return sum(w[k] * z[k] for k in keys) / var**0.5


def daily_deviation(branch_D: dict[str, float]) -> float | None:
    return statistics.fmean(branch_D.values()) if branch_D else None


def ewma(d_t: float, s_prev: float | None, alpha: float) -> float:
    return d_t if s_prev is None else alpha * d_t + (1 - alpha) * s_prev


def top_contributors(z_by_branch: dict[str, dict[str, float]], n: int = 3) -> list[dict]:
    contrib = [
        (branch, k, load_branch(branch).weights.get(k, 0.0) * v)
        for branch, z in z_by_branch.items()
        for k, v in z.items()
    ]
    contrib.sort(key=lambda t: -t[2])
    return [{"branch": b, "feature": k, "contribution": round(c, 3)} for b, k, c in contrib[:n]]
