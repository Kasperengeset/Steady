"""Relapse model: an explainable 3-state hidden Markov model on top of the daily deviation D.

Everything up to D is unchanged (features → healthy baseline → z → D per branch, combined per day as
Σ D / √n so it has SD 1 under "no change"). This module replaces the old EWMA + fixed thresholds.

States
  well      D_b ~ N(0, 1)
  relapse   D_b ~ N(+δ_b·√n_b, 1)   – shifted in the depression direction
  elevated  D_b ~ N(−δ_b·√n_b, 1)   – unusually "better" than normal (possible hypomania; clinician only)

δ_b is how far a full control→patient difference moves D (acoustic: held-out detectability on the
Androids Corpus; text: rough estimate from the diary sanity set). n_b = check-ins of that branch today.

Each day: (1) transition – a small daily chance of moving between states, (2) Bayes update with the
day's evidence. In log-odds the relapse-vs-well evidence of one branch is linear in D:

    llr_b = δ·√n·D − (δ·√n)² / 2

so it splits exactly over features (D is a weighted sum of z-scores): each feature gets δ·√n·share_i
and the constant −(δ√n)²/2 is the daily "scepticism" – one odd day is not enough on its own.

Robustness: D is clipped to [−δ√n − 1, δ√n + 1] before the likelihood, so one broken recording or one
cold can multiply the relapse odds by at most exp((δ√n)²/2 + δ√n) (≈ ×17 for one spoken check-in), and
very good days cannot bank unlimited credit.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from ..config import settings
from .state import WEIGHTS_DIR

STATES = ("well", "relapse", "elevated")

# Plain-language phrases for "this feature moved in the worse direction" (used in the clinician sentence)
PHRASES = {
    "arousal": "less energetic-sounding voice",
    "energy": "quieter voice",
    "valence": "more negative-sounding voice",
    "speech_rate": "slower speech",
    "f0_std": "flatter pitch",
    "pause_ratio": "more pauses",
    "jitter": "less steady pitch",
    "shimmer": "less steady loudness",
    "text_logit": "diary language closer to depression",
    "sentiment": "more negative wording",
    "absolutist": "more absolute words (always, never)",
    "first_person": "more self-focused wording",
}


@dataclass(frozen=True)
class Params:
    delta: dict[str, float]
    h_relapse: float  # well → relapse per day
    h_elevated: float  # well → elevated per day
    r_relapse: float  # relapse → well per day
    r_elevated: float  # elevated → well per day
    prior: tuple[float, float, float]
    clip_margin: float = 1.0
    emission_sd: float = 1.0  # > 1 tempers the evidence when D is noisier than SD 1 (baseline error, day-to-day noise)


@lru_cache(maxsize=1)
def params() -> Params:
    delta = dict(settings.relapse_delta)
    path = WEIGHTS_DIR / "acoustic.json"
    if path.exists():  # data: held-out detectability of a full shift with the app's (equal) weights
        det = json.loads(path.read_text()).get("change_model", {}).get("detectability", {})
        delta["acoustic"] = round(det.get("held_out_mean", {}).get("equal", delta["acoustic"]), 3)
    return Params(
        delta=delta,
        h_relapse=settings.relapse_h_relapse,
        h_elevated=settings.relapse_h_elevated,
        r_relapse=settings.relapse_r_relapse,
        r_elevated=settings.relapse_r_elevated,
        prior=tuple(settings.relapse_prior),
    )


@dataclass
class Step:
    posterior: dict[str, float]
    llr_relapse: float  # relapse vs well evidence today (0 if no data)
    llr_elevated: float  # elevated vs well evidence today
    by_branch: dict[str, float] = field(default_factory=dict)  # relapse evidence per branch
    by_feature: dict[str, float] = field(default_factory=dict)  # relapse evidence per feature
    scepticism: float = 0.0  # Σ −(δ√n)²/2: by_feature + scepticism = llr_relapse
    clipped: list[str] = field(default_factory=list)
    sentence: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["posterior"] = {k: round(v, 5) for k, v in self.posterior.items()}
        for k in ("llr_relapse", "llr_elevated", "scepticism"):
            d[k] = round(d[k], 4)
        d["by_branch"] = {k: round(v, 4) for k, v in self.by_branch.items()}
        d["by_feature"] = {k: round(v, 4) for k, v in self.by_feature.items()}
        return d


def transition(p: dict[str, float], days: int, prm: Params) -> dict[str, float]:
    """Apply the daily transition matrix `days` times (0 = no change)."""
    w, r, e = p["well"], p["relapse"], p["elevated"]
    for _ in range(max(days, 0)):
        w, r, e = (
            w * (1 - prm.h_relapse - prm.h_elevated) + r * prm.r_relapse + e * prm.r_elevated,
            w * prm.h_relapse + r * (1 - prm.r_relapse),
            w * prm.h_elevated + e * (1 - prm.r_elevated),
        )
    return {"well": w, "relapse": r, "elevated": e}


def prior(prm: Params | None = None) -> dict[str, float]:
    return dict(zip(STATES, (prm or params()).prior))


def step(
    prev: dict[str, float] | None,
    D_by_branch: dict[str, float],
    n_by_branch: dict[str, int],
    days_elapsed: int = 1,
    shares_by_branch: dict[str, dict[str, float]] | None = None,
    prm: Params | None = None,
) -> Step:
    """One day of the forward filter.

    prev: posterior of the previous scored day (None → prior at baseline lock).
    D_by_branch: the day's combined D per branch (Σ D / √n). n_by_branch: check-ins per branch today.
    shares_by_branch: per feature, the part of D_b it explains (Σ = D_b) – for the explanation only.
    """
    prm = prm or params()
    p = transition(prev or prior(prm), days_elapsed if prev else 0, prm)

    loglik = {s: 0.0 for s in STATES}
    by_branch, by_feature, clipped = {}, {}, []
    scepticism = 0.0
    for b, D in D_by_branch.items():
        n = max(n_by_branch.get(b, 1), 1)
        m = prm.delta.get(b, prm.delta.get("acoustic", 1.5)) * math.sqrt(n)
        lo, hi = -m - prm.clip_margin, m + prm.clip_margin
        Dc = min(max(D, lo), hi)
        if Dc != D:
            clipped.append(b)
        v = prm.emission_sd**2
        for s, mu in (("well", 0.0), ("relapse", m), ("elevated", -m)):
            loglik[s] += -((Dc - mu) ** 2) / (2 * v)
        by_branch[b] = (m * Dc - m * m / 2) / v
        scepticism -= m * m / (2 * v)
        shares = (shares_by_branch or {}).get(b, {})
        scale = (Dc / D) if D else 0.0  # if clipped, shrink all feature contributions proportionally
        for f, sh in shares.items():
            by_feature[f] = by_feature.get(f, 0.0) + m * sh * scale / v

    top = max(loglik.values())
    unnorm = {s: p[s] * math.exp(loglik[s] - top) for s in STATES}
    tot = sum(unnorm.values())
    post = {s: v / tot for s, v in unnorm.items()}
    llr_r = loglik["relapse"] - loglik["well"]
    llr_e = loglik["elevated"] - loglik["well"]
    return Step(post, llr_r, llr_e, by_branch, by_feature, scepticism, clipped, _sentence(llr_r, llr_e, by_feature))


def _sentence(llr_r: float, llr_e: float, by_feature: dict[str, float]) -> str:
    if not by_feature and llr_r == 0 and llr_e == 0:
        return "No check-in today – the estimate only drifts slowly."
    worse = [f for f, v in sorted(by_feature.items(), key=lambda t: -t[1]) if v > 0.2 and f in PHRASES][:2]
    if llr_r > 0 and worse:
        return "Compared with the healthy normal: " + " and ".join(PHRASES[f] for f in worse) + "."
    if llr_e > 1:
        return "Unusually elevated compared with the healthy normal (energy and mood above usual)."
    return "Close to the healthy normal today."


def level_from_posterior(post: dict[str, float] | None, crisis: bool) -> tuple[int, bool]:
    """(level, elevated_flag). 3 = crisis flag (outside the model), 2 = clinician, 1 = friends.
    The elevated flag goes to the clinician only, never to friends."""
    if crisis:
        return 3, False
    if not post:
        return 0, False
    level = 2 if post["relapse"] >= settings.p_clinician else 1 if post["relapse"] >= settings.p_friends else 0
    return level, post["elevated"] >= settings.p_elevated
