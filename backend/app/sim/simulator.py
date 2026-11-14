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
from ..engine import bandit, escalation, relapse, state

TRACKS = ("rolling", "anchored", "combined")
K = settings.sd_prior_strength

ARMS = {  # key: (label, hidden true helpfulness for this persona)
    "walk_outside": ("Walk outside", 0.75),
    "call_friend": ("Call a friend", 0.65),
    "morning_light": ("Morning light", 0.45),
    "music": ("Music session", 0.40),
    "breathing": ("Breathing exercise", 0.30),
    "small_task": ("Small activity task", 0.55),
}


@dataclass
class SimArm:
    key: str
    a: float = 1.0
    b: float = 1.0


def _combine(weighted: list[tuple[float, float]]) -> float | None:
    """Day combination of (weight, D) pairs: Σ w·D / √(Σ w²). All weights 1 → Σ D / √n (B16)."""
    if not weighted:
        return None
    if all(w == 1.0 for w, _ in weighted):
        return state.combine_checkins([v for _, v in weighted])
    return sum(w * v for w, v in weighted) / sum(w * w for w, _ in weighted) ** 0.5


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
    checkins_per_day: int = 1,
    elevated_day: int | None = None,
    elevated_len: int = 10,
    relapse_params: "relapse.Params | None" = None,
    # --- #27 design checks. Defaults reproduce the original simulator exactly. -----------------
    prompt_hours: tuple[float, float] | None = None,  # check-in hour ~ U(a, b); None = no time of day
    tod_effect: float = 0.0,  # person's morning→evening half-range per feature ~ N(0, tod_effect × within SD)
    tod_mode: str = "one_slot",  # "one_slot" | "two_slots" | "hour_covariate"
    slot_split: float = 14.0,
    slot_min_n: int | None = None,  # baseline values a slot needs (app: healthy_min_checkins)
    day_shared: float = 0.0,  # share of the noise variance common to all check-ins of one day
    extra_rate: float = 0.0,  # unprompted extra check-ins per day ~ Poisson(extra_rate)
    extra_weight: float = 1.0,  # 1 = count fully (B16), 0.5 = partly, 0 = context only
    adherence: float | None = None,  # P(scheduled check-in done) on a healthy day; None = always
    adherence_drop: float = 0.0,  # P drops by adherence_drop × latent d
    busy_rate: float = 0.0,  # P(a busy spell starts) per day; a spell lasts ~5 days, adherence − 0.4
    engagement_weight: float = 0.0,  # weight of the engagement signal next to the acoustic D (anchored track)
) -> dict:
    rng = random.Random(seed)
    # Separate stream for the #27 options, so the original streams (and results) are unchanged
    xrng = np.random.default_rng([seed, 27])
    slot_min_n = settings.healthy_min_checkins if slot_min_n is None else slot_min_n
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
    tod_amp = {f: xrng.normal(0, tod_effect * stats[f]["within_sd"]) for f in feats}  # this person's day curve

    d, hist_d = 0.1, []
    history: list[dict[str, float]] = []
    meta: list[tuple[int, float | None]] = []  # (day, hour) per history entry
    out = []
    S_roll = S_anch = None
    post: dict[str, float] | None = None  # relapse model posterior (engine/relapse.py)
    anchor: dict[str, list[float]] | None = None
    anchor_slot: dict[str, dict] = {}
    slope: dict[str, float] = {}
    h0 = 15.0
    answered: list[int] = []  # scheduled check-ins done per day (engagement)
    p_hat: float | None = None
    busy_left = 0
    level = {k: 0 for k in TRACKS}

    def slot_of(h: float | None) -> str:
        return "all" if h is None else ("morning" if h < slot_split else "evening")

    def adj(x: dict[str, float], h: float | None) -> dict[str, float]:
        """Hour as a covariate: remove the slope fitted on the healthy baseline."""
        if tod_mode != "hour_covariate" or h is None or not slope:
            return x
        return {f: v - slope.get(f, 0.0) * (h - h0) for f, v in x.items()}

    for t in range(days):
        # --- latent course -------------------------------------------------------------
        ramp = min(1.0, max(0.0, (t - relapse_day) / relapse_len))
        target = 0.1 + 0.75 * ramp
        if elevated_day is not None and elevated_day <= t < elevated_day + elevated_len:
            target = -0.75  # elevated episode: voice shifts the opposite way (energetic, positive)
        d = min(1.0, max(-1.0, 0.7 * d + 0.3 * target + rng.gauss(0, 0.04)))

        # --- spoken check-ins → features (morning, evening, extras ...) ------------------
        busy = False
        if busy_rate:
            if busy_left == 0 and xrng.random() < busy_rate:
                busy_left = int(xrng.geometric(1 / 5))
            busy, busy_left = busy_left > 0, max(0, busy_left - 1)
        p_do = None if adherence is None else min(1.0, max(0.0, adherence - adherence_drop * max(d, 0.0) - 0.4 * busy))
        e_day = chol @ xrng.standard_normal(len(order)) if day_shared else None
        n_extra = int(xrng.poisson(extra_rate)) if extra_rate else 0
        xs, hs, ws = [], [], []
        done = 0
        for i in range(checkins_per_day + n_extra):
            scheduled = i < checkins_per_day
            e = chol @ (nrng if scheduled else xrng).standard_normal(len(order))
            if e_day is not None:
                e = (1 - day_shared) ** 0.5 * e + day_shared**0.5 * e_day
            noise = dict(zip(wc["features"], wsd * e))
            if scheduled and p_do is not None and xrng.random() >= p_do:
                continue  # missed this check-in
            done += scheduled
            h = None if prompt_hours is None else float(xrng.uniform(*prompt_hours))
            g = 0.0 if h is None else (h - 15.0) / 6.0  # −1 at 09:00, +1 at 21:00
            xs.append({
                f: offset[f] + s["control_mean"] + d * (s["patient_mean"] - s["control_mean"]) + noise[f] + tod_amp[f] * g
                for f, s in stats.items()
            })
            hs.append(h)
            ws.append(1.0 if scheduled else extra_weight)
        answered.append(done)
        x = xs[0] if xs else None

        # --- engine: rolling baseline (as in the app) and anchored baseline (T1) --------
        # Same functions as the app: median/MAD shrunk toward corpus SD, rolling needs >= rolling_min_n
        window = [i for i, (day, _) in enumerate(meta) if day >= t - baseline_days]
        roll_cache: dict[str, dict] = {}

        def roll_stats(h: float | None) -> dict:
            key = slot_of(h) if tod_mode == "two_slots" else "all"
            if key not in roll_cache:
                idx = [i for i in window if key == "all" or slot_of(meta[i][1]) == key]
                vals = [adj(history[i], meta[i][1]) for i in idx]
                roll_cache[key] = state.history_stats({f: [v[f] for v in vals] for f in feats}, model, K, settings.rolling_min_n)
            return roll_cache[key]

        if anchor is None and t >= baseline_days and len(history) >= min(settings.healthy_min_checkins, baseline_days * checkins_per_day):
            hours = [h for _, h in meta]
            if tod_mode == "hour_covariate" and None not in hours:
                # Per-feature slope on the hour, ridge-shrunk toward 0 by K pseudo-values (Var U(9,21) = 12)
                h0 = statistics.fmean(hours)
                hh = np.array(hours) - h0
                for f in feats:
                    v = np.array([e[f] for e in history])
                    slope[f] = float(hh @ (v - v.mean()) / (hh @ hh + K * 12.0))
            base = [adj(e, h) for e, (_, h) in zip(history, meta)]  # healthy period over → lock
            anchor = state.history_stats({f: [e[f] for e in base] for f in feats}, model, K, 1)
            if tod_mode == "two_slots":
                for sl in ("morning", "evening"):
                    sub = [e for e, (_, h) in zip(base, meta) if slot_of(h) == sl]
                    anchor_slot[sl] = state.history_stats({f: [e[f] for e in sub] for f in feats}, model, K, slot_min_n)
            if adherence is not None:
                p_hat = min(0.95, max(0.05, sum(answered[:-1]) / (len(answered[:-1]) * checkins_per_day)))
        # Each check-in gets its own D; the day combines them per branch with sum/sqrt(n) as the app (B16)
        D_rolls, D_anchs = [], []
        for xi, h, w in zip(xs, hs, ws):
            if w == 0:
                continue  # extra check-in kept as context only
            xa = adj(xi, h)
            z_roll = state.zscores(xa, roll_stats(h), model.direction)
            z_anch = state.zscores(xa, {**anchor, **anchor_slot.get(slot_of(h), {})}, model.direction) if anchor else {}
            if z_roll:
                D_rolls.append((w, state.branch_deviation(z_roll, model)))
            if z_anch:
                D_anchs.append((w, state.branch_deviation(z_anch, model)))
        D_roll = _combine(D_rolls)
        D_anch = _combine(D_anchs)
        D_eng = None
        if p_hat is not None:  # engagement: today's share of scheduled check-ins vs the healthy baseline rate
            D_eng = (p_hat - done / checkins_per_day) / (p_hat * (1 - p_hat) / checkins_per_day) ** 0.5
        if D_anch is not None:  # same evidence the app uses: anchored D, combined over the day's check-ins
            post = relapse.step(post, {"acoustic": D_anch}, {"acoustic": len(D_anchs)}, prm=relapse_params).posterior
        if engagement_weight and D_eng is not None:
            we = engagement_weight
            D_anch = D_eng if D_anch is None else (D_anch + we * D_eng) / (1 + we**2) ** 0.5
        scored_roll, scored_anch = D_roll is not None, D_anch is not None
        if scored_roll:
            S_roll = state.ewma(D_roll, S_roll, 0.3)
        if scored_anch:
            S_anch = state.ewma(D_anch, S_anch, 0.3)
        recent_roll = [r["S_rolling"] for r in out if r["S_rolling"] is not None][-6:] + ([S_roll] if scored_roll else [])
        recent_anch = [r["S_anchored"] for r in out if r["S_anchored"] is not None][-6:] + ([S_anch] if scored_anch else [])
        # App behaviour: once the healthy baseline is locked, S = max(S_anchored, S_rolling)
        scored_comb = bool(anchor) and (scored_roll or scored_anch)
        S_comb = max((v for v in (S_anch, S_roll) if v is not None), default=None) if scored_comb else None
        recent_comb = [r["S_combined"] for r in out if r["S_combined"] is not None][-6:] + ([S_comb] if S_comb is not None else [])
        # A day without a score keeps the previous level (the app only re-evaluates on a check-in)
        for k, sc, rec in (("rolling", scored_roll, recent_roll), ("anchored", scored_anch, recent_anch),
                           ("combined", S_comb is not None, recent_comb)):
            if sc:
                level[k] = escalation.evaluate_level(rec, False)

        # --- self-report (lagged) -------------------------------------------------------
        hist_d.append(d)
        d_lag = hist_d[max(0, t - self_report_lag)]
        self_mood = min(10, max(0, round(7.5 - 5 * d_lag + rng.gauss(0, 0.8))))

        # --- intervention loop (real bandit) ---------------------------------------------
        arm = bandit.choose_arm(arms, rng)
        true_p = ARMS[arm.key][1]
        completed = rng.random() < (0.85 - 0.5 * d)
        helped = completed and rng.random() < true_p
        rating = (rng.randint(4, 5) if helped else rng.randint(1, 3)) if completed else None
        r = bandit.reward(completed, rating, state_improved=None)
        bandit.update(arm, r)
        if helped:
            d = max(0.0, d - 0.03)

        history.extend(xs)
        meta.extend((t, h) for h in hs)
        out.append({
            "day": t,
            "latent": round(d, 3),
            "self_mood": self_mood,
            "S_rolling": round(S_roll, 3) if scored_roll else None,
            "S_anchored": round(S_anch, 3) if scored_anch else None,
            "level_rolling": level["rolling"],
            "level_anchored": level["anchored"],
            "S_combined": None if S_comb is None else round(S_comb, 3),
            "D_anchored": None if D_anch is None else round(D_anch, 4),
            "p_relapse": None if post is None else round(post["relapse"], 4),
            "p_elevated": None if post is None else round(post["elevated"], 4),
            "level_combined": level["combined"],
            "checkins": len(xs),
            "D_engagement": None if D_eng is None else round(D_eng, 4),
            "arm": arm.key,
            "completed": completed,
            "reward": round(r, 3),
            "features": None if x is None else {f: round(v, 4) for f, v in x.items()},
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
