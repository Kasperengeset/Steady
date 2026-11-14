"""Engine design checks for GitHub #27 (STATUS.md B30–B32), on the same seeded synthetic patients.

    backend/.venv/bin/python -m app.sim.evaluate_design            # all three, ~5–10 min on 8 cores
    backend/.venv/bin/python -m app.sim.evaluate_design tod        # or: extras, engagement

1. tod         Random prompt times (09–21) with a personal time-of-day effect: one slot vs two slots split
               at 14:00 (as the app is configured: a slot needs 10 baseline values; and with 5) vs one
               slot with the hour as a covariate.
2. extras      One prompted check-in a day plus unprompted extras (Poisson), with part of the noise shared
               within a day: count extras fully (B16), at half weight, or as context only.
3. engagement  Adherence that drops with the latent state (and busy spells that look the same): does an
               engagement signal next to the acoustic D add detection at the same false-alarm rate?

Per condition: 100 healthy patients for a year after the baseline locks (false alarms) and 100 relapse
patients (relapse from day 40, followed to day 90). "Current rule" = the app's escalation as it is.
"Calibrated" = the S threshold set to ≤ 1 false clinician onset per healthy patient-year (as B19).
Writes ml/reports/engine/design_checks.md.
"""

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .evaluate_relapse import LOCK, _onsets, calibrate
from .simulator import simulate

REPORT = Path(__file__).resolve().parents[3] / "ml" / "reports" / "engine" / "design_checks.md"
N = 100


def _series(r: dict) -> dict[str, list]:
    days = r["days"][LOCK:]
    S, last = [], -9.0
    for d in days:  # a day without a check-in keeps yesterday's S (the app shows the last score)
        last = d["S_combined"] if d["S_combined"] is not None else last
        S.append(last)
    return {"S": S, "clin": [d["level_combined"] >= 2 for d in days], "friends": [d["level_combined"] >= 1 for d in days],
            "checkins": sum(d["checkins"] for d in days) / len(days)}


def evaluate(kw: dict) -> dict:
    healthy = [_series(simulate(days=LOCK + 365, seed=10_000 + i, relapse_day=10**6, **kw)) for i in range(N)]
    thr, fa_cal = calibrate([h["S"] for h in healthy], [x / 20 for x in range(10, 160)], 1.0)
    out = {
        "fa_clin": sum(_onsets(h["clin"]) for h in healthy) / N,
        "fa_friends": sum(_onsets(h["friends"]) for h in healthy) / N,
        "thr": thr, "fa_cal": fa_cal, "checkins_per_day": statistics.fmean(h["checkins"] for h in healthy),
    }
    miss_cur = miss_cal = 0
    delay_cur, delay_cal = [], []
    rel = 40 - LOCK
    for i in range(N):
        s = _series(simulate(days=90, seed=i, relapse_day=40, **kw))
        for flags, store, key in ((s["clin"], delay_cur, "cur"), ([x > thr for x in s["S"]], delay_cal, "cal")):
            hit = next((t for t in range(rel, len(flags)) if flags[t]), None)
            if hit is None:
                miss_cur += key == "cur"
                miss_cal += key == "cal"
            else:
                store.append(hit - rel)
    out.update(missed_cur=miss_cur, missed_cal=miss_cal,
               delay_cur=statistics.median(delay_cur) if delay_cur else None,
               delay_cal=statistics.median(delay_cal) if delay_cal else None)
    return out


def _run(configs: list[tuple[str, dict]]) -> list[tuple[str, dict, dict]]:
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(evaluate, [kw for _, kw in configs]))
    return [(name, kw, r) for (name, kw), r in zip(configs, results)]


HEAD = "| Condition | Check-ins/day | False clinician alerts/yr (current rule) | False friend nudges/yr | Missed of 100 (current rule) | Calibrated threshold | Missed of 100 (calibrated ≤ 1 FA/yr) | Median delay, calibrated (days) |\n|---|---|---|---|---|---|---|---|"


def _row(name: str, r: dict) -> str:
    fa = "–" if r["fa_cal"] != r["fa_cal"] else f"S > {r['thr']}"
    return (f"| {name} | {r['checkins_per_day']:.2f} | {r['fa_clin']:.2f} | {r['fa_friends']:.2f} | {r['missed_cur']} | "
            f"{fa} | **{r['missed_cal']}** | {r['delay_cal']} |")


def tod() -> str:
    modes = [("one slot", {"tod_mode": "one_slot"}),
             ("two slots at 14:00, slot needs 10 values (app now)", {"tod_mode": "two_slots", "slot_min_n": 10}),
             ("two slots at 14:00, slot needs 5 values", {"tod_mode": "two_slots", "slot_min_n": 5}),
             ("one slot + hour as covariate", {"tod_mode": "hour_covariate"})]
    cfg = [(f"×{noise}, time-of-day effect {te} SD · {m}", {"day_noise_factor": noise, "prompt_hours": (9.0, 21.0),
            "tod_effect": te, **kw})
           for noise in (1.0, 1.5) for te in (0.0, 0.5, 1.0) for m, kw in modes]
    rows = [_row(n, r) for n, _, r in _run(cfg)]
    return "## 1. Time of day with random prompts (#27)\n\nOne prompted check-in a day at a uniform random time 09–21. Each person gets their own day curve: feature shift = a × (hour − 15)/6, a ~ N(0, effect × within-person SD), so the effect is the typical morning-to-evening half-range.\n\n" + HEAD + "\n" + "\n".join(rows) + "\n"


def extras() -> str:
    cfg = []
    for shared in (0.0, 0.3, 0.6):
        cfg.append((f"day-shared noise {shared} · 2 scheduled/day (B16 as is)", {"day_noise_factor": 1.5, "checkins_per_day": 2, "day_shared": shared}))
        cfg.append((f"day-shared noise {shared} · 1 prompted, no extras", {"day_noise_factor": 1.5, "day_shared": shared}))
        for rate in (0.5, 1.5):
            for w, lab in ((1.0, "count fully (B16)"), (0.5, "half weight"), (0.0, "context only")):
                cfg.append((f"day-shared noise {shared} · 1 prompted + {rate} extras/day · {lab}",
                            {"day_noise_factor": 1.5, "day_shared": shared, "extra_rate": rate, "extra_weight": w}))
    rows = [_row(n, r) for n, _, r in _run(cfg)]
    return "## 2. One prompt a day and extra check-ins (#27 vs B16)\n\nNoise ×1.5. \"Day-shared\" = the share of each check-in's noise variance that all check-ins of one day have in common (a cold, poor sleep, the same room). B16 assumes 0.\n\n" + HEAD + "\n" + "\n".join(rows) + "\n"


def engagement() -> str:
    cfg = [("full adherence, no engagement signal (reference)", {"day_noise_factor": 1.5})]
    for drop in (0.0, 0.3, 0.6):
        for busy in (0.0, 1 / 30):
            for we in (0.0, 0.5, 1.0):
                cfg.append((f"adherence 0.85 − {drop}·d · busy spells {'1/month' if busy else 'none'} · engagement weight {we}",
                            {"day_noise_factor": 1.5, "adherence": 0.85, "adherence_drop": drop, "busy_rate": busy,
                             "engagement_weight": we}))
    rows = [_row(n, r) for n, _, r in _run(cfg)]
    return ("## 3. Engagement as a signal (#27)\n\nNoise ×1.5, one prompted check-in a day done with probability 0.85 − drop × d "
            "(d = latent depression, 0.1 when well, ~0.85 in a full relapse), minus 0.4 during busy spells (~5 days, starting "
            "1/30 days). Engagement D = (baseline rate − done today) / √(p(1−p)), SD 1 when nothing changes; it is combined "
            "with the acoustic anchored D as (D_ac + w·D_eng)/√(1+w²), and alone on a missed day. Weight 0 = today's engine "
            "(a missed day has no score).\n\n" + HEAD + "\n" + "\n".join(rows) + "\n")


def main() -> None:
    which = sys.argv[1:] or ["tod", "extras", "engagement"]
    parts = {"tod": tod, "extras": extras, "engagement": engagement}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    sections = {}
    if REPORT.exists():  # keep sections not rerun
        for chunk in REPORT.read_text().split("\n## ")[1:]:
            sections[{"1": "tod", "2": "extras", "3": "engagement"}[chunk[0]]] = "## " + chunk.rstrip() + "\n"
    for w in which:
        print(f"running {w} …", flush=True)
        sections[w] = parts[w]()
        print(sections[w], flush=True)
    REPORT.write_text("# Engine design checks (#27)\n\nGenerated by `backend/app/sim/evaluate_design.py` (simulator on the real engine, "
                      f"{N} healthy patient-years and {N} relapse patients per row, same seeds in every row). Decisions: STATUS.md B30–B32.\n\n"
                      + "\n".join(sections[k] for k in ("tod", "extras", "engagement") if k in sections))
    print(f"→ {REPORT}")


if __name__ == "__main__":
    main()
