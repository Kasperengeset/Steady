"""Synthetic patient for the demo – docs/plan.md §7, STATUS.md.

Runs the *real* engine (engine/state.py, bandit.py, escalation.py, trained acoustic.json) on a
synthetic persona, one spoken morning check-in per day:

- Latent depression d_t in [0, 1] (0 = remission, 1 = like the Androids patients), AR(1) around a
  target that ramps up from `relapse_day` (a slow relapse over ~15 days).
- Acoustic features: x_i = person offset + control_mean_i + d_t * (patient_mean_i - control_mean_i) + noise,
  with corpus means/SDs from acoustic.json["feature_stats"] → realistic effect sizes (Cohen's d 0.2–0.9).
- Self-reported mood (0–10) follows d_t with a lag (ASSUMPTION – see `assumptions` in the output).
- Bandit picks one intervention per day; each arm has a hidden true helpfulness. Doing a helpful
  activity nudges d_t down a little.

Returns both baselines so the demo can show why a fixed post-treatment ("healthy") baseline matters:
  S_rolling  – current engine: z vs the previous 14 days (a slow relapse leaks into the baseline)
  S_anchored – z vs the first 14 days after TMS (STATUS.md T1)
  S_combined – what the app escalates on: max of the two once the healthy baseline is locked
"""

import random
import statistics
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..engine import bandit, escalation, state

TRACKS = ("rolling", "anchored", "combined")
K = settings.sd_prior_strength

ARMS = {  # key: (label, hidden true helpfulness for this persona)
    "walk_outside": ("Tur ute", 0.75),
    "call_friend": ("Ring en venn", 0.65),
    "morning_light": ("Morgenlys", 0.45),
    "music": ("Musikkøkt", 0.40),
    "breathing": ("Pusteøvelse", 0.30),
    "small_task": ("Liten aktivitetsoppgave", 0.55),
}


@dataclass
class SimArm:
    key: str
    a: float = 1.0
    b: float = 1.0


def _first_run(flags: list[bool], start: int, run: int) -> int | None:
    for t in range(start, len(flags) - run + 1):
        if all(flags[t : t + run]):
            return t
    return None


def simulate(
    days: int = 70,
    seed: int = 0,
    relapse_day: int = 40,
    relapse_len: int = 15,
    day_noise_factor: float = 1.0,
    self_report_lag: int = 4,
    baseline_days: int = 14,
) -> dict:
    rng = random.Random(seed)
    model = state.load_branch("acoustic")
    stats = model.meta.get("feature_stats")
    if not stats:
        raise RuntimeError("acoustic.json has no feature_stats – rerun ml/train_weights.py")
    feats = list(stats)
    offset = {f: rng.gauss(0, stats[f]["sd"] * 0.8) for f in feats}  # this person's own voice
    # Day-to-day noise: within-person SD and correlation measured on Androids (clips of one session),
    # scaled by day_noise_factor because real day-to-day noise is larger than within-session noise.
    wc = model.meta["within_corr"]
    order = [feats.index(f) for f in wc["features"]]
    chol = np.linalg.cholesky(np.array(wc["matrix"]))
    wsd = np.array([stats[f]["within_sd"] for f in wc["features"]]) * day_noise_factor
    nrng = np.random.default_rng(seed)
    arms = [SimArm(k) for k in ARMS]

    d, hist_d = 0.1, []
    history: list[dict[str, float]] = []
    out = []
    S_roll = S_anch = None
    anchor: dict[str, list[float]] | None = None

    for t in range(days):
        # --- latent course -------------------------------------------------------------
        ramp = min(1.0, max(0.0, (t - relapse_day) / relapse_len))
        target = 0.1 + 0.75 * ramp
        d = min(1.0, max(0.0, 0.7 * d + 0.3 * target + rng.gauss(0, 0.04)))

        # --- spoken check-in → features -------------------------------------------------
        noise = dict(zip(wc["features"], wsd * (chol @ nrng.standard_normal(len(order)))))
        x = {
            f: offset[f] + s["control_mean"] + d * (s["patient_mean"] - s["control_mean"]) + noise[f]
            for f, s in stats.items()
        }

        # --- engine: rolling baseline (as in the app) and anchored baseline (T1) --------
        # Same functions as the app: median/MAD shrunk toward corpus SD, rolling needs >= rolling_min_n
        window = history[-baseline_days:]
        roll_hist = {f: [h[f] for h in window] for f in feats}
        z_roll = state.zscores(x, state.history_stats(roll_hist, model, K, settings.rolling_min_n), model.direction)
        if len(history) == baseline_days and anchor is None:  # healthy period over → lock (frozen)
            anchor = state.history_stats({f: [h[f] for h in history] for f in feats}, model, K, 1)
        z_anch = state.zscores(x, anchor, model.direction) if anchor else {}

        D_roll = state.branch_deviation(z_roll, model) if z_roll else None
        D_anch = state.branch_deviation(z_anch, model) if z_anch else None
        if D_roll is not None:
            S_roll = state.ewma(D_roll, S_roll, 0.3)
        if D_anch is not None:
            S_anch = state.ewma(D_anch, S_anch, 0.3)
        recent_roll = [r["S_rolling"] for r in out if r["S_rolling"] is not None][-6:] + ([S_roll] if S_roll is not None else [])
        recent_anch = [r["S_anchored"] for r in out if r["S_anchored"] is not None][-6:] + ([S_anch] if S_anch is not None else [])
        # App behaviour: once the healthy baseline is locked, S = max(S_anchored, S_rolling)
        S_comb = max((v for v in (S_anch, S_roll) if v is not None), default=None) if anchor else None
        recent_comb = [r["S_combined"] for r in out if r["S_combined"] is not None][-6:] + ([S_comb] if S_comb is not None else [])

        # --- self-report (lagged) -------------------------------------------------------
        hist_d.append(d)
        d_lag = hist_d[max(0, t - self_report_lag)]
        self_mood = min(10, max(0, round(7.5 - 5 * d_lag + rng.gauss(0, 0.8))))

        # --- intervention loop (real bandit) ---------------------------------------------
        arm = bandit.choose_arm(arms)
        true_p = ARMS[arm.key][1]
        completed = rng.random() < (0.85 - 0.5 * d)
        helped = completed and rng.random() < true_p
        rating = (rng.randint(4, 5) if helped else rng.randint(1, 3)) if completed else None
        r = bandit.reward(completed, rating, state_improved=None)
        bandit.update(arm, r)
        if helped:
            d = max(0.0, d - 0.03)

        history.append(x)
        out.append({
            "day": t,
            "latent": round(d, 3),
            "self_mood": self_mood,
            "S_rolling": None if S_roll is None else round(S_roll, 3),
            "S_anchored": None if S_anch is None else round(S_anch, 3),
            "level_rolling": escalation.evaluate_level(recent_roll, False) if S_roll is not None else 0,
            "level_anchored": escalation.evaluate_level(recent_anch, False) if S_anch is not None else 0,
            "S_combined": None if S_comb is None else round(S_comb, 3),
            "level_combined": escalation.evaluate_level(recent_comb, False) if S_comb is not None else 0,
            "arm": arm.key,
            "completed": completed,
            "reward": round(r, 3),
            "features": {f: round(v, 4) for f, v in x.items()},
        })

    # --- summary: who notices first? -----------------------------------------------------
    base_mood = statistics.fmean(r["self_mood"] for r in out[:baseline_days])
    mood_avg3 = [statistics.fmean(r["self_mood"] for r in out[max(0, t - 2) : t + 1]) for t in range(days)]
    self_flag = _first_run([m <= base_mood - 2 for m in mood_avg3], relapse_day - 5, 1)

    def engine_flag(key: str) -> int | None:
        lv = [r[key] >= 1 for r in out]
        return next((t for t in range(relapse_day - 5, days) if lv[t]), None)

    def false_alarm_days(key: str) -> int:
        return sum(r[key] >= 1 for r in out[: relapse_day - 5])

    posterior = {a.key: round(a.a / (a.a + a.b), 3) for a in arms}
    best_true = max(ARMS, key=lambda k: ARMS[k][1])
    summary = {
        "relapse_starts": relapse_day,
        "self_report_flag_day": self_flag,
        "engine_flag_day": {k: engine_flag(f"level_{k}") for k in TRACKS},
        "lead_days_vs_self_report": {
            k: (None if self_flag is None or v is None else self_flag - v)
            for k, v in {k: engine_flag(f"level_{k}") for k in TRACKS}.items()
        },
        "false_alarm_days_before_relapse": {k: false_alarm_days(f"level_{k}") for k in TRACKS},
        "arms": {k: {"label": ARMS[k][0], "true": ARMS[k][1], "learned": posterior[k],
                     "times_chosen": sum(r["arm"] == k for r in out)} for k in ARMS},
        "bandit_top_arm": max(posterior, key=posterior.get),
        "true_best_arm": best_true,
    }
    assumptions = {
        "day_noise": f"day-to-day noise = {day_noise_factor} × within-person SD measured on Androids (one session; real day-to-day noise is larger)",
        "self_report_lag": f"self-reported mood lags the latent state by {self_report_lag} days (ASSUMPTION, not evidence)",
        "correlated_noise": "feature noise uses the within-person correlation measured on Androids",
        "effect_sizes": "feature shifts from Androids corpus (patients vs controls)",
    }
    return {"days": out, "summary": summary, "assumptions": assumptions, "weights_trained": model.trained}


def demo_lead_time(seeds: int = 50, **kw) -> dict:
    """Robustness over many random personas – use this number, not a single lucky seed."""
    leads = {k: [] for k in TRACKS}
    misses = {k: 0 for k in TRACKS}
    fa = {k: [] for k in TRACKS}
    bandit_hit = 0
    for s in range(seeds):
        r = simulate(seed=s, **kw)["summary"]
        for k in leads:
            v = r["lead_days_vs_self_report"][k]
            if r["engine_flag_day"][k] is None:
                misses[k] += 1
            elif v is not None:
                leads[k].append(v)
            fa[k].append(r["false_alarm_days_before_relapse"][k])
        bandit_hit += r["bandit_top_arm"] == r["true_best_arm"]
    med = lambda v: None if not v else statistics.median(v)  # noqa: E731
    return {
        "seeds": seeds,
        "median_lead_days": {k: med(v) for k, v in leads.items()},
        "missed_relapses": misses,
        "mean_false_alarm_days": {k: round(statistics.fmean(v), 2) for k, v in fa.items()},
        "bandit_finds_best_arm": f"{bandit_hit}/{seeds}",
    }
