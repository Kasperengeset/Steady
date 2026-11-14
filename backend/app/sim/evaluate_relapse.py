"""Decision gate for the relapse model (STATUS.md «Relapse model»).

Compares, on the same seeded synthetic patients, the clinician alert of
  current   – the old rule as it is (S > 2, or S > 1 for 7 days; EWMA S = max(anchored, rolling))
  ewma_cal  – EWMA S with its threshold calibrated to the false-alarm target
  hmm_cal   – P(relapse) from engine/relapse.py with its threshold calibrated to the same target
Calibration: healthy patients for one year after the baseline locks; the lowest threshold with
≤ `target_per_year` alert onsets per patient-year. Evaluation: relapse patients (relapse from day 40,
followed to day 90): missed relapses, median delay to the first alert, false onsets before the relapse.

    backend/.venv/bin/python -m app.sim.evaluate_relapse
"""

import statistics
import sys

from ..engine import relapse
from .simulator import simulate

LOCK = 14  # baseline_days in the simulator


def _onsets(flags: list[bool]) -> int:
    return sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))


def _series(r: dict) -> dict[str, list]:
    days = r["days"][LOCK:]
    return {
        "S": [d["S_combined"] if d["S_combined"] is not None else -9 for d in days],
        "P": [d["p_relapse"] if d["p_relapse"] is not None else 0.0 for d in days],
        "cur": [d["level_combined"] >= 2 for d in days],
        "cur_friends": [d["level_combined"] >= 1 for d in days],
    }


def calibrate(values: list[list[float]], grid: list[float], target_per_year: float) -> tuple[float, float]:
    years = sum(len(v) for v in values) / 365
    for thr in grid:
        fa = sum(_onsets([x > thr for x in v]) for v in values) / years
        if fa <= target_per_year:
            return thr, fa
    return grid[-1], float("nan")


def run(noise: float, per_day: int, n_healthy=100, n_relapse=100, target=1.0, prm: relapse.Params | None = None) -> dict:
    kw = dict(day_noise_factor=noise, checkins_per_day=per_day, relapse_params=prm)
    healthy = [_series(simulate(days=LOCK + 365, seed=10_000 + i, relapse_day=10**6, **kw)) for i in range(n_healthy)]
    a_s, fa_s = calibrate([h["S"] for h in healthy], [x / 20 for x in range(10, 120)], target)
    a_p, fa_p = calibrate([h["P"] for h in healthy], [x / 100 for x in range(5, 100)] + [0.995, 0.999], target)
    years = n_healthy * 365 / 365
    fa_cur = sum(_onsets(h["cur"]) for h in healthy) / years
    fa_friends = sum(_onsets(h["cur_friends"]) for h in healthy) / years

    res = {m: {"missed": 0, "delay": [], "pre_fa": 0} for m in ("current", "ewma_cal", "hmm_cal")}
    for i in range(n_relapse):
        s = _series(simulate(days=90, seed=i, relapse_day=40, **kw))
        rel = 40 - LOCK
        alerts = {"current": s["cur"], "ewma_cal": [x > a_s for x in s["S"]], "hmm_cal": [x > a_p for x in s["P"]]}
        for m, fl in alerts.items():
            res[m]["pre_fa"] += _onsets(fl[:rel])
            hit = next((t for t in range(rel, len(fl)) if fl[t]), None)
            if hit is None:
                res[m]["missed"] += 1
            else:
                res[m]["delay"].append(hit - rel)
    out = {"friends_fa_per_year_current": round(fa_friends, 2), "thresholds": {"ewma_S": a_s, "hmm_P": a_p}, "fa_per_year": {"current": round(fa_cur, 2), "ewma_cal": round(fa_s, 2), "hmm_cal": round(fa_p, 2)}}
    for m, r in res.items():
        out[m] = {"missed": r["missed"], "median_delay": statistics.median(r["delay"]) if r["delay"] else None}
    return out


def main() -> None:
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"Clinician alerts calibrated to ≤ {target} false onset per healthy patient-year (100 healthy, 100 relapse patients)\n")
    print(f"{'noise':6s} {'/day':5s} | {'method':9s} {'FA/yr':>6s} {'missed':>7s} {'delay':>6s} | thresholds")
    for noise in (1.0, 1.5, 2.0):
        for per_day in (1, 2):
            r = run(noise, per_day, target=target)
            for m in ("current", "ewma_cal", "hmm_cal"):
                thr = "" if m == "current" else f"S>{r['thresholds']['ewma_S']}" if m == "ewma_cal" else f"P>{r['thresholds']['hmm_P']}"
                print(f"×{noise:<5} {per_day:<5} | {m:9s} {r['fa_per_year'][m]:>6} {r[m]['missed']:>7} {str(r[m]['median_delay']):>6} | {thr}")
            print()


if __name__ == "__main__":
    main()
