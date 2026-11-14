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
    "text": {
        # pipeline/text.py (written diary or transcript)
        "text_logit": +1, "absolutist": +1, "first_person": +1, "sentiment": -1,
        # Gemini content scores (pipeline/llm.py) – weight 0 in weights/text.json until the stub is replaced
        "mood": -1, "energy_self": -1, "anhedonia": +1, "social_contact": -1,
    },
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


MAD_TO_SD = 1.4826  # MAD × 1.4826 ≈ SD for normal data


def robust_stats(values: list[float], prior_sd: float | None, k: float) -> tuple[float, float]:
    """Centre = median, scale = MAD-based SD shrunk toward the corpus within-person SD.

    With ~14–28 baseline check-ins a personal SD is noisy (and MAD can be 0), so
        sigma^2 = (n · s_personal^2 + k · s_corpus^2) / (n + k)
    lets the corpus dominate early and the person's own data take over as n grows.
    Median/MAD keep one odd day (cold, bad recording) from distorting the baseline.
    """
    center = statistics.median(values)
    mad = statistics.median(abs(v - center) for v in values) * MAD_TO_SD
    n = len(values)
    if prior_sd:
        scale = ((n * mad**2 + k * prior_sd**2) / (n + k)) ** 0.5
    else:
        scale = mad or statistics.pstdev(values)
    return center, scale if scale > 0 else 1.0


def prior_sd(model: BranchModel, feature: str) -> float | None:
    """Within-person SD measured on the training corpus (acoustic.json feature_stats), if any."""
    return model.meta.get("feature_stats", {}).get(feature, {}).get("within_sd")


def history_stats(
    history: dict[str, list[float]], model: BranchModel, k: float, min_n: int
) -> dict[str, tuple[float, float, int]]:
    """feature -> (centre, scale, n) for this branch's features with at least min_n values."""
    out = {}
    for f in model.direction:
        vals = history.get(f, [])
        if len(vals) >= min_n:
            c, sc = robust_stats(vals, prior_sd(model, f), k)
            out[f] = (c, sc, len(vals))
    return out


def zscores(today: dict[str, float], stats: dict[str, tuple[float, float, int]], direction: dict[str, int]) -> dict[str, float]:
    """Direction-adjusted z-scores of today's values against a baseline (positive = worse)."""
    return {
        f: direction.get(f, 1) * (x - stats[f][0]) / stats[f][1]
        for f, x in today.items()
        if f in stats
    }


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


def _norm(keys: list[str], model: BranchModel) -> float:
    w = {k: model.weights.get(k, 0.0) for k in keys}
    var = sum(w[a] * w[b] * _corr(model, a, b) for a in keys for b in keys)
    return var**0.5 if var > 0 else 0.0


def feature_shares(z: dict[str, float], model: BranchModel) -> dict[str, float]:
    """How much of D each feature explains: w_i·z_i / √(wᵀRw). The shares sum exactly to D."""
    norm = _norm(list(z), model)
    if norm == 0:
        return {k: 0.0 for k in z}
    return {k: model.weights.get(k, 0.0) * v / norm for k, v in z.items()}


def branch_deviation(z: dict[str, float], model: BranchModel) -> float:
    """Weighted sum of z-scores, rescaled to unit variance under 'no change' (STATUS.md B10, B12).

    Var(sum w*z) = sum_ij w_i w_j R_ij, where R is the within-person correlation of the (direction-
    adjusted) z-scores, measured on Androids and stored in <branch>.json["within_corr"]. Dividing by its
    square root makes D ~ N(0, 1) when nothing changes, so escalation thresholds (S > 1, S > 2) keep a
    fixed meaning even though e.g. energy and arousal move together. Without a correlation matrix the
    features are treated as independent (norm = sqrt(sum w^2)).
    """
    return sum(feature_shares(z, model).values())


def combine_checkins(values: list[float], weights: list[float] | None = None) -> float:
    """Combine several check-ins of one day for one branch: Σ w·D / √(Σ w²) (weighted Stouffer).

    Each check-in's D has SD 1 under "no change", so the combination keeps SD 1 (thresholds keep their
    meaning) while a real shift grows with more check-ins. All weights 1 gives sum(D) / sqrt(n) (B16).
    B31: extra unprompted check-ins get weight EXTRA_WEIGHT, because same-day noise is partly shared
    (a cold, poor sleep), which plain Stouffer ignores and which raised false alarms in the simulator.
    """
    if weights is None or all(w == 1.0 for w in weights):
        return sum(values) / len(values) ** 0.5
    return sum(w * v for w, v in zip(weights, values)) / sum(w * w for w in weights) ** 0.5


EXTRA_WEIGHT = 0.5  # B31: weight of an extra (unprompted, not the day's first) check-in in its branch


def checkin_weight(prompted: bool, earlier_today_in_branch: int) -> float:
    """B31: the check-in that answers today's prompt, or the day's first one in a branch, counts fully."""
    return 1.0 if prompted or earlier_today_in_branch == 0 else EXTRA_WEIGHT


HOUR_VAR = 12.0  # variance of the hour for prompts uniform over 09–21 (one pseudo-value's leverage)


def hour_slope(values: list[float], hours: list[float | None], k: float) -> tuple[float, float, int] | None:
    """B30: a feature's change per hour of the day on the healthy baseline, ridge-shrunk toward 0.

        slope = Σ (h − h̄)(x − x̄) / (Σ (h − h̄)² + k · 12)

    k pseudo-values with no slope (12 = Var of an hour uniform on 09–21) keep a short baseline from
    fitting noise; with all check-ins at the same hour the slope is 0. Returns (slope, h̄, n) or None
    when fewer than two values have an hour.
    """
    pairs = [(x, h) for x, h in zip(values, hours) if h is not None]
    if len(pairs) < 2:
        return None
    h0 = statistics.fmean(h for _, h in pairs)
    x0 = statistics.fmean(x for x, _ in pairs)
    sxy = sum((h - h0) * (x - x0) for x, h in pairs)
    sxx = sum((h - h0) ** 2 for _, h in pairs)
    return sxy / (sxx + k * HOUR_VAR), h0, len(pairs)


def adjust_for_hour(values: dict[str, float], hour: float | None, slopes: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Remove the baseline's time-of-day slope: x − slope·(hour − h̄). No hour → unchanged."""
    if hour is None or not slopes:
        return values
    return {f: v - slopes[f][0] * (hour - slopes[f][1]) if f in slopes else v for f, v in values.items()}


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
