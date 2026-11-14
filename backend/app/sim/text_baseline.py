"""Text-branch false alarms for steady patients: a prior within-person SD plus a z cap as the fix (STATUS.md B34).

    backend/.venv/bin/python -m app.sim.text_baseline            # report → ml/reports/engine/text_baseline.md
    backend/.venv/bin/python -m app.sim.text_baseline --write    # also store the prior in weights/text.json

Problem (found by seed_demo, PR #40): a steady patient who only writes good diaries can get alerts from
the text branch alone. Good-day diaries vary a lot in text_logit, and text has no corpus within-person SD
to shrink toward (B15/B17), so a 5–14-diary baseline can get a too small scale.

Data: ml/text/diary_bank.csv, team-written only (never Reddit, never Gemini): three authors' banks of
good and low days (seed_demo, diary_sanity, context_set). A simulated patient writes in ONE author's
style: every day one diary drawn from that author's good bank (steady), or, in a relapse, from the low
bank with a probability that ramps from 0 to 1 over 15 days from day 40.

Prior SD s0 per text feature = pooled within-author SD of good-day diaries. To avoid testing on the data
it is fitted on, a patient of author X uses s0 estimated from the OTHER two authors (leave-one-author-out).
The --write prior uses all three.

Engine as the app: robust_stats (median/MAD, shrunk toward s0 with k = 5 when a prior exists), anchored
baseline locked after `baseline_n` diaries, rolling track (last 14 days, ≥ 7), S = max of the two,
EWMA α = 0.3, escalation.evaluate_level. Calibration as B19: ≤ 1 false clinician onset per patient-year.
"""

import json
import random
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import settings
from ..engine import escalation, state
from ..pipeline import text as text_model
from .evaluate_relapse import _onsets, calibrate

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "ml" / "text" / "diary_bank.csv"
REPORT = ROOT / "ml" / "reports" / "engine" / "text_baseline.md"
K = settings.sd_prior_strength
N = 100


@lru_cache(maxsize=1)
def bank() -> pd.DataFrame:
    df = pd.read_csv(BANK)
    feats = [text_model.analyze_text(t)["features"] for t in df["text"]]
    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def feats() -> list[str]:
    return list(state.load_branch("text").direction)[:4]  # text_logit, absolutist, first_person, sentiment


def prior_sd(authors: list[str]) -> dict[str, float]:
    """Pooled within-author SD of good-day diaries."""
    b = bank()
    g = b[(b["label"] == "good") & b["author"].isin(authors)]
    out = {}
    for f in feats():
        ss = sum(((x - x.mean()) ** 2).sum() for _, x in g.groupby("author")[f])
        dfree = len(g) - g["author"].nunique()
        out[f] = (ss / dfree) ** 0.5
    return out


def _stats(values: dict[str, list[float]], prior: dict[str, float] | None, floor: dict[str, float] | None, min_n: int):
    out = {}
    for f, vals in values.items():
        if len(vals) < min_n:
            continue
        c, sc = state.robust_stats(vals, (prior or {}).get(f), K)
        if floor:
            sc = max(sc, floor[f])
        out[f] = (c, sc, len(vals))
    return out


def patient(seed: int, author: str, variant: str, baseline_n: int, days: int, relapse_day: int | None) -> dict[str, list]:
    b = bank()
    rows = {lab: b[(b["author"] == author) & (b["label"] == lab)][feats()].to_dict("records") for lab in ("good", "low")}
    others = sorted(set(b["author"]) - {author})
    s0 = prior_sd(others)
    prior = s0 if "prior" in variant else None
    floor = s0 if "floor" in variant else None
    centre_unc = "cu" in variant  # T18: the centre is uncertain too → scale × √(1 + 1.57/n)
    cap = 3.0 if "cap" in variant else None  # one diary moves z by at most ±3
    model = state.load_branch("text")
    model = state.BranchModel(model.name, model.weights, model.direction, model.trained,
                              {k: v for k, v in model.meta.items() if k not in ("feature_stats", "z_cap")})  # the variant decides
    rng = random.Random(f"{seed}:{author}")
    hist: list[dict] = []
    anchor = None
    S_a = S_r = None
    rec_a, rec_r, rec_c = [], [], []
    out = {"S": [], "clin": [], "friends": []}
    for t in range(days):
        p_low = 0.0 if relapse_day is None else min(1.0, max(0.0, (t - relapse_day) / 15))
        x = rng.choice(rows["low"] if rng.random() < p_low else rows["good"])
        if anchor is None and len(hist) >= baseline_n:
            anchor = _stats({f: [h[f] for h in hist] for f in feats()}, prior, floor, min(baseline_n, settings.min_feature_n))
            if centre_unc:
                anchor = {f: (c, sc * (1 + 1.57 / n) ** 0.5, n) for f, (c, sc, n) in anchor.items()}
        roll = _stats({f: [h[f] for h in hist[-settings.baseline_days:]] for f in feats()}, prior, floor, settings.rolling_min_n)
        hist.append(x)
        if anchor is None:
            continue
        z_a, z_r = state.zscores(x, anchor, model.direction, cap), state.zscores(x, roll, model.direction, cap)
        if z_a:
            S_a = state.ewma(state.branch_deviation(z_a, model), S_a, settings.ewma_alpha)
            rec_a.append(S_a)
        if z_r:
            S_r = state.ewma(state.branch_deviation(z_r, model), S_r, settings.ewma_alpha)
            rec_r.append(S_r)
        S = max(v for v in (S_a, S_r) if v is not None)
        rec_c.append(S)
        lv = escalation.evaluate_level(rec_c[-7:], False)
        out["S"].append(S)
        out["clin"].append(lv >= 2)
        out["friends"].append(lv >= 1)
    return out


def evaluate(variant: str, baseline_n: int) -> dict:
    authors = sorted(set(bank()["author"]))
    healthy, relapse = [], []
    for a in authors:
        healthy += [patient(10_000 + i, a, variant, baseline_n, baseline_n + 365, None) for i in range(N // len(authors) + 1)]
        relapse += [patient(i, a, variant, baseline_n, 90, 40) for i in range(N // len(authors) + 1)]
    years = sum(len(h["S"]) for h in healthy) / 365
    thr, fa_cal = calibrate([h["S"] for h in healthy], [x / 20 for x in range(10, 200)], 1.0)
    rel = 40 - baseline_n
    miss_cur = miss_cal = 0
    delays = []
    for r in relapse:
        if not any(r["clin"][rel:]):
            miss_cur += 1
        hit = next((t for t in range(rel, len(r["S"])) if r["S"][t] > thr), None)
        if hit is None:
            miss_cal += 1
        else:
            delays.append(hit - rel)
    return {
        "fa_clin": sum(_onsets(h["clin"]) for h in healthy) / years,
        "fa_friends": sum(_onsets(h["friends"]) for h in healthy) / years,
        "max_S_healthy": max(max(h["S"]) for h in healthy),
        "thr": thr, "fa_cal": fa_cal, "missed_cur": miss_cur, "missed_cal": miss_cal, "n_relapse": len(relapse),
        "delay": statistics.median(delays) if delays else None,
    }


LABELS = {"none": "no prior (app now)", "prior": "prior SD s0, shrunk with k = 5 (B15 as for voice)",
          "floor": "minimum scale = s0", "prior_floor": "prior + minimum scale",
          "prior_cu": "prior + centre uncertainty (T18)", "prior_cap": "prior + z capped at ±3",
          "prior_cu_cap": "prior + centre uncertainty + z cap"}


def main() -> None:
    b = bank()
    s_all = prior_sd(sorted(set(b["author"])))
    lines = []
    for n in (5, 14):
        for v in LABELS:
            r = evaluate(v, n)
            print(n, v, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
            cal = "–" if r["fa_cal"] != r["fa_cal"] else f"S > {r['thr']}"
            lines.append(f"| {n} | {LABELS[v]} | {r['fa_clin']:.2f} | {r['fa_friends']:.2f} | {r['max_S_healthy']:.1f} | "
                         f"{r['missed_cur']} / {r['n_relapse']} | {cal} | **{r['missed_cal']} / {r['n_relapse']}** | {r['delay']} |")
    spread = b[b["label"] == "good"].groupby("author")["text_logit"].agg(["mean", "std", "count"]).round(2)
    gap = b.groupby(["author", "label"])["text_logit"].mean().unstack().round(2)
    s_rows = "\n".join(f"| `{f}` | {s_all[f]:.3f} |" for f in feats())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Text branch: false alarms for steady patients and a prior within-person SD (B34)

Generated by `backend/app/sim/text_baseline.py`. Team-written diaries only (`ml/text/diary_bank.csv`,
{len(b)} entries from 3 authors), scored by the local text model; no Gemini, no Reddit. One diary a day,
the text branch alone. Per row: {3 * (N // 3 + 1)} steady patients × 1 year after the baseline locks, and
{3 * (N // 3 + 1)} relapse patients (good → low diaries, ramp over 15 days from day 40). Leave-one-author-out
prior: a patient of author X gets s0 from the other two authors.

| Baseline diaries | Scale | False clinician alerts/yr (current rule) | False friend nudges/yr | Max S, steady patients | Missed relapses (current rule) | Calibrated threshold | Missed (calibrated ≤ 1 FA/yr) | Median delay, calibrated (days) |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

Good-day `text_logit` per author (the spread a steady person shows), and the good/low means:

| Author | Good mean | Good SD | n | Low mean |
|---|---|---|---|---|
""" + "\n".join(f"| {a} | {spread.loc[a, 'mean']} | {spread.loc[a, 'std']} | {int(spread.loc[a, 'count'])} | {gap.loc[a, 'low']} |" for a in spread.index) + f"""

Prior s0 from all three authors (what `--write` stores in `weights/text.json["feature_stats"]`):

| Feature | s0 (pooled within-author SD of good days) |
|---|---|
{s_rows}

**Limits.** Three authors and 26 good-day diaries; each patient reuses a small bank, so day-to-day text
variety is lower than in real life. The point is the relative effect of the scale, not the absolute rates.
""")
    print(f"→ {REPORT}")
    if "--write" in sys.argv:
        path = state.WEIGHTS_DIR / "text.json"
        data = json.loads(path.read_text())
        data["feature_stats"] = {f: {"within_sd": round(s_all[f], 4)} for f in feats()}
        data["z_cap"] = 3.0
        data["feature_stats_source"] = ("pooled within-author SD of good-day diaries in ml/text/diary_bank.csv "
                                        "(team-written, 3 authors) – prior for robust_stats, STATUS.md B34")
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"→ {path}")


if __name__ == "__main__":
    main()
