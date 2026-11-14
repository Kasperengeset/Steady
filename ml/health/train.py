"""Health branch: validate the activity features on Depresjon, write engine/weights/health.json.

    python ml/download_data.py --depresjon                 # → data/depresjon/ (gitignored)
    backend/.venv/bin/python ml/health/train.py            # ~1 min

- Day features with the app's own code (backend/app/pipeline/health.py, STATUS.md B9).
- Leave-one-patient-out (asserted): a person's days are never in train and test at the same time.
- Logistic regression (standardised) and a random forest reference. A person's score is the mean of
  their day probabilities. Person-level F1, MCC and ROC-AUC next to the published baseline
  (Garcia-Ceja et al. 2018, leave-one-patient-out: F1 0.73, MCC 0.44).
- Checks: gender + age alone (leakage), unipolar only (the condition group includes bipolar).
- Engine weights follow the acoustic branch (B12): equal weights, literature directions, D
  normalised with the within-person correlation of the activity features measured here (B15).
  Sleep and heart features have no training data here; they get equal weight and literature
  directions, and are treated as independent.

Writes backend/app/engine/weights/health.json and ml/reports/health/summary.md.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import health as hp  # noqa: E402

DATA = ROOT / "data" / "depresjon" / "data"
REPORTS = ROOT / "ml" / "reports" / "health"
WEIGHTS = ROOT / "backend" / "app" / "engine" / "weights" / "health.json"
SEED = 42
FEATS = list(hp.ACTIVITY_FEATURES)
PUBLISHED = {"f1": 0.73, "mcc": 0.44}


def load_days() -> pd.DataFrame:
    if not DATA.exists():
        sys.exit(f"{DATA} missing – run: python ml/download_data.py --depresjon")
    scores = pd.read_csv(DATA / "scores.csv").set_index("number")
    rows = []
    for group, label in (("condition", 1), ("control", 0)):
        for path in sorted((DATA / group).glob(f"{group}_*.csv")):
            pid = path.stem
            x = pd.read_csv(path, parse_dates=["timestamp"])
            x["minute"] = x["timestamp"].dt.hour * 60 + x["timestamp"].dt.minute
            for day, d in x.groupby(x["timestamp"].dt.date):
                if len(d) != hp.MINUTES_PER_DAY or d["minute"].nunique() != hp.MINUTES_PER_DAY:
                    continue  # full days only
                minutes = d.sort_values("minute")["activity"].to_numpy()
                rows.append({"person": pid, "label": label, "day": day,
                             "afftype": scores.loc[pid, "afftype"] if pid in scores.index else np.nan,
                             "gender": scores.loc[pid, "gender"], "age": scores.loc[pid, "age"],
                             **hp.daily_activity_features(minutes)})
    return pd.DataFrame(rows)


def logo_person_scores(days: pd.DataFrame, make_model, cols: list[str]) -> pd.DataFrame:
    """Leave-one-patient-out; returns one row per person with the mean day probability."""
    out = []
    groups = days["person"].to_numpy()
    for tr, te in LeaveOneGroupOut().split(days, groups=groups):
        assert set(groups[tr]).isdisjoint(groups[te]), "person leak between train and test"
        m = make_model().fit(days.iloc[tr][cols], days.iloc[tr]["label"])
        p = m.predict_proba(days.iloc[te][cols])[:, 1]
        out.append({"person": groups[te][0], "label": int(days.iloc[te]["label"].iloc[0]), "p": float(p.mean())})
    return pd.DataFrame(out)


def person_metrics(ps: pd.DataFrame) -> dict:
    pred = (ps["p"] >= 0.5).astype(int)
    return {"f1": f1_score(ps["label"], pred), "mcc": matthews_corrcoef(ps["label"], pred),
            "auc": roc_auc_score(ps["label"], ps["p"]), "n": len(ps), "n_depressed": int(ps["label"].sum())}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1)


def within_person(days: pd.DataFrame) -> tuple[np.ndarray, dict, dict]:
    """Pooled within-person correlation of the raw features, within-person SD and d per feature."""
    centred = days.groupby("person")[FEATS].transform(lambda c: c - c.mean())
    sd_within = centred.std()
    z = centred / sd_within
    corr = np.corrcoef(z.to_numpy().T)
    means = days.groupby(["person", "label"])[FEATS].mean().reset_index()
    diff = means[means.label == 1][FEATS].mean() - means[means.label == 0][FEATS].mean()
    d = (diff / sd_within).to_dict()
    return corr, sd_within.to_dict(), d


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    days = load_days()
    people = days.groupby("person")["label"].first()
    print(f"{len(days)} full days, {len(people)} people ({int(people.sum())} depressed, {int((1 - people).sum())} controls)")

    res = {
        "logistic regression": person_metrics(logo_person_scores(days, lr, FEATS)),
        "random forest": person_metrics(logo_person_scores(days, rf, FEATS)),
    }
    uni = days[(days["label"] == 0) | (days["afftype"] == 2)]
    res["logistic regression, unipolar only"] = person_metrics(logo_person_scores(uni, lr, FEATS))

    # Leakage check: demographics alone (one row per person)
    demo = days.groupby("person").first().reset_index()
    demo["age_mid"] = demo["age"].str.split("-").apply(lambda a: (int(a[0]) + int(a[1])) / 2)
    demo_days = days.merge(demo[["person", "age_mid"]], on="person")
    res["gender + age only (leakage check)"] = person_metrics(logo_person_scores(demo_days, lr, ["gender", "age_mid"]))
    for k, v in res.items():
        print(f"  {k:<38} F1 {v['f1']:.3f}  MCC {v['mcc']:+.3f}  AUC {v['auc']:.3f}")

    full = lr().fit(days[FEATS], days["label"])
    beta = dict(zip(FEATS, full[-1].coef_[0]))
    disagree = [f for f in FEATS if np.sign(beta[f]) != hp.DIRECTION[f]]
    corr, sd_within, d_within = within_person(days)

    # Equal weights (B12) over the features that move in the literature direction here. A feature with
    # no group difference (|d| < 0.1) or one pointing the other way adds only noise to D: weight 0,
    # still computed and shown. Sleep and heart features keep equal weight (no data to judge them).
    no_signal = [f for f in FEATS if abs(d_within[f]) < 0.1 or np.sign(d_within[f]) != hp.DIRECTION[f]]
    scored = [f for f in hp.FEATURES if f not in no_signal]
    weights = {f: (round(1 / len(scored), 4) if f in scored else 0.0) for f in hp.FEATURES}
    WEIGHTS.write_text(json.dumps({
        "branch": "health",
        "weights": weights,
        "zero_weight": {f: f"within-person d {d_within[f]:+.2f}: no signal or against the literature on Depresjon"
                        for f in no_signal},
        "direction": hp.DIRECTION,
        "within_corr": {"features": FEATS, "matrix": [[round(float(v), 4) for v in row] for row in corr]},
        "beta_standardised": {f: round(float(b), 4) for f, b in beta.items()},
        "literature_disagreements": disagree,
        "within_person_d": {f: round(float(v), 3) for f, v in d_within.items()},
        "validation": {k: {m: round(float(x), 3) for m, x in v.items()} for k, v in res.items()},
        "published_baseline": PUBLISHED,
        "data": "Depresjon (Garcia-Ceja et al. 2018), Actiwatch counts per minute, full days only",
        "note": "Equal weights and literature directions (B12), except zero_weight features. within_corr covers the 5 activity features; "
                "sleep and heart features come from the wearable's own summaries, have no training data here "
                "and are treated as independent. No feature_stats: Actiwatch counts and Fitbit steps differ in "
                "scale, so the personal baseline sets the scale.",
        "trained_on": str(date.today()),
    }, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Health branch: activity features on Depresjon", "",
        f"Generated by `ml/health/train.py` on {date.today()}. {len(days)} full days from {len(people)} people "
        f"({int(people.sum())} depressed, {int((1 - people).sum())} controls). Features from "
        "`backend/app/pipeline/health.py`, the same code the app runs on Fitbit data.", "",
        "Leave-one-patient-out (asserted). A person's score is the mean of their day probabilities, cut-off 0.5.", "",
        "| Model | F1 | MCC | ROC-AUC |", "|---|---|---|---|",
    ]
    lines += [f"| {k} | {v['f1']:.3f} | {v['mcc']:+.3f} | {v['auc']:.3f} |" for k, v in res.items()]
    lines += [f"| published baseline (Garcia-Ceja et al. 2018) | {PUBLISHED['f1']:.2f} | {PUBLISHED['mcc']:+.2f} | |", "",
              "## Per feature", "",
              "Standardised beta from logistic regression on all days. Within-person d = (mean of depressed people − "
              "mean of controls) / within-person SD: how far a full depressed-vs-control difference moves a feature "
              "compared with its normal day-to-day swing. Direction is the literature direction the engine uses.", "",
              "| Feature | Direction | Beta | Within-person d | Engine weight |", "|---|---|---|---|---|"]
    lines += [f"| {f} | {'higher' if hp.DIRECTION[f] > 0 else 'lower'} = worse | {beta[f]:+.2f} | {d_within[f]:+.2f} | {weights[f]:.3f} |" for f in FEATS]
    lines += [f"| {f} | {'higher' if hp.DIRECTION[f] > 0 else 'lower'} = worse | (no data) | (no data) | {weights[f]:.3f} |" for f in hp.SUMMARY_FEATURES]
    lines += ["", f"Betas that disagree with the literature: {', '.join(disagree) or 'none'}. A beta can flip because the "
              "features overlap (B6), so the engine judges each feature by its own group difference (d): equal weight "
              f"if d moves in the literature direction, weight 0 if not or if |d| < 0.1 ({', '.join(no_signal) or 'none'}). "
              "Zero-weight features are still computed and can be shown to the clinician.", "",
              "## Limitations", "",
              "- Actiwatch counts are not Fitbit steps. The model shows the patterns exist; in the app the live signal is "
              "the per-person z-score, which removes the scale.",
              "- Between-person classification, used as evidence for within-person change (same caveat as acoustic).",
              "- 55 people, 14 to 45 days each. The condition group includes bipolar patients (unipolar-only row above).",
              "- Sleep, resting heart rate and HRV have no training data here; their directions come from RADAR-MDD and "
              "related literature."]
    (REPORTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"beta {dict((f, round(float(b), 2)) for f, b in beta.items())}  disagree {disagree}")
    print(f"within-person d {dict((f, round(float(v), 2)) for f, v in d_within.items())}")
    print(f"saved {WEIGHTS.relative_to(ROOT)} and ml/reports/health/summary.md")


if __name__ == "__main__":
    main()
