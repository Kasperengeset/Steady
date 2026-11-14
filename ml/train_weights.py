"""Train acoustic-branch weights on Androids Corpus features – docs/plan.md §6.

  backend/.venv/bin/python ml/train_weights.py                  # after extract_features.py
  backend/.venv/bin/python ml/train_weights.py --features X.csv # any clip CSV with the same columns

Speaker-level logistic regression (depressed vs control), speaker-independent CV with the
corpus' fold-list.csv. Writes:
  backend/app/engine/weights/acoustic.json   w_i = |beta_i| (normalised), s_i = sign(beta_i)
  ml/reports/acoustic/{roc.png, coefficients.png, summary.md}

Honest framing: a between-person classifier, used only as a *prior* for which features matter
and in which direction. The personal baseline in the app is what tracks within-person change.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import FEATURES, ROOT, emb_name  # noqa: E402

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

APP_FEATURES = ["speech_rate", "f0_std", "pause_ratio", "energy", "jitter", "shimmer", "arousal", "valence"]
OUT_WEIGHTS = ROOT / "backend" / "app" / "engine" / "weights" / "acoustic.json"
REPORTS = ROOT / "ml" / "reports" / "acoustic"
LITERATURE = {  # +1 = higher is worse (must match engine/state.py DIRECTION["acoustic"])
    "speech_rate": -1, "f0_std": -1, "pause_ratio": +1, "energy": -1,
    "jitter": +1, "shimmer": +1, "arousal": -1, "valence": -1,
}


def _logreg() -> LogisticRegressionCV:
    # C (regularisation) chosen by inner CV on the training part of each outer fold
    return LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12), cv=StratifiedKFold(4, shuffle=True, random_state=0),
        scoring="neg_log_loss", class_weight="balanced", max_iter=5000,  # log-loss → calibrated probabilities (AUC is scale-blind)
        l1_ratios=(0.0,), use_legacy_attributes=False,  # pure L2
    )


def model():
    return make_pipeline(StandardScaler(), _logreg())


def pca_model(n: int):
    return make_pipeline(StandardScaler(), PCA(n_components=n, random_state=0), _logreg())


def speaker_table(clips: pd.DataFrame) -> pd.DataFrame:
    num = clips.select_dtypes("number").columns.difference(["label", "age", "education"])
    agg = clips.groupby("speaker")[list(num)].median()
    meta = clips.groupby("speaker")[["label", "gender", "age", "education"]].first()
    return pd.concat([meta, agg], axis=1).reset_index()


def assign_folds(spk: pd.DataFrame, task: str, n: int = 5) -> tuple[np.ndarray, str]:
    try:
        from corpus import load_folds

        folds = load_folds(task)
    except SystemExit:
        folds = {}
    coverage = spk["speaker"].map(folds).notna().mean() if folds else 0.0
    if coverage >= 0.9:
        f = spk["speaker"].map(folds).fillna(-1).astype(int).to_numpy()
        return f, f"corpus fold-lists.csv ({coverage:.0%} of speakers)"
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0)
    f = np.full(len(spk), -1)
    for k, (_, test) in enumerate(sgkf.split(spk, spk["label"], groups=spk["speaker"])):
        f[test] = k
    return f, f"StratifiedGroupKFold(5) fallback – fold-list.csv covered {coverage:.0%}"


def cv_auc(make, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict:
    oof = np.full(len(y), np.nan)
    per_fold = []
    for k in sorted(set(folds) - {-1}):
        tr, te = folds != k, folds == k
        if len(set(y[te])) < 2:
            continue
        m = make().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        per_fold.append(round(float(roc_auc_score(y[te], oof[te])), 3))
    mask = ~np.isnan(oof)
    return {
        "mean": round(float(np.mean(per_fold)), 3), "std": round(float(np.std(per_fold)), 3),
        "per_fold": per_fold, "pooled": round(float(roc_auc_score(y[mask], oof[mask])), 3),
        "_oof": oof,
    }


def load_embeddings(clips: pd.DataFrame, task: str, speakers: pd.Series) -> np.ndarray | None:
    emb_dir = FEATURES / "emb" / task
    by_spk: dict[str, list[np.ndarray]] = {}
    for clip, spk in zip(clips["clip"], clips["speaker"]):
        p = emb_dir / emb_name(clip)
        if p.exists():
            by_spk.setdefault(spk, []).append(np.load(p))
    if len(by_spk) < len(speakers):
        return None
    return np.stack([np.mean(by_spk[s], axis=0) for s in speakers])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["interview", "reading"], default="interview")
    ap.add_argument("--features", type=Path, help="clip CSV (default data/features/<task>_clips.csv)")
    ap.add_argument("--no-write", action="store_true", help="evaluate only, don't overwrite acoustic.json")
    args = ap.parse_args()

    path = args.features or FEATURES / f"{args.task}_clips.csv"
    if not path.exists():
        sys.exit(f"{path} missing – run ml/extract_features.py --task {args.task}")
    clips = pd.read_csv(path)
    spk = speaker_table(clips)
    feats = [f for f in APP_FEATURES if f"app_{f}" in spk]
    y = spk["label"].to_numpy()
    folds, fold_src = assign_folds(spk, args.task)
    print(f"{len(clips)} clips → {len(spk)} speakers ({y.sum()} depressed / {len(y) - y.sum()} controls); folds: {fold_src}")

    X_app = spk[[f"app_{f}" for f in feats]].to_numpy()
    results = {"app": cv_auc(model, X_app, y, folds)}

    eg_cols = [c for c in spk if c.startswith("eg_")]
    if eg_cols:
        results["egemaps88"] = cv_auc(model, spk[eg_cols].to_numpy(), y, folds)
    demo = np.column_stack([(spk["gender"] == "F").astype(float), spk["age"].astype(float)])
    results["demographics"] = cv_auc(model, demo, y, folds)
    E = load_embeddings(clips, args.task, spk["speaker"])
    if E is not None:
        n_pc = min(20, int((folds != folds.max()).sum()) - 1)  # PCA cannot exceed training-fold size
        results["wav2vec2_probe"] = cv_auc(lambda: pca_model(n_pc), E, y, folds)

    for name, r in results.items():
        print(f"  AUC {name:15s} {r['mean']:.3f} ± {r['std']:.3f}  (pooled {r['pooled']:.3f})  folds {r['per_fold']}")
    if results["app"]["mean"] > 0.95:
        print("  ⚠️  AUC > 0.95 – suspect leakage (are folds speaker-independent?)")

    # Final model on all speakers → standardised coefficients = weights
    final = model().fit(X_app, y)
    beta = final[-1].coef_[0]
    total = np.abs(beta).sum() or 1.0
    weights = {f: round(float(abs(b) / total), 4) for f, b in zip(feats, beta)}
    trained_dir = {f: int(np.sign(b)) or LITERATURE[f] for f, b in zip(feats, beta)}
    disagree = [f for f in feats if trained_dir[f] != LITERATURE[f]]
    # A sign flip in the multivariate fit is usually collinearity (the feature's unique share given
    # correlated ones), not evidence that the marginal effect is reversed. For within-person change we
    # want the marginal direction, so flipped features keep the literature direction (weights stay small).
    direction = {f: LITERATURE[f] if f in disagree else trained_dir[f] for f in feats}
    print("  weights:", weights)
    if disagree:
        print("  ⚠️  multivariate sign contradicts literature for:", disagree, "→ keeping literature direction")

    payload = {
        "branch": "acoustic",
        "weights": weights,
        "direction": direction,
        "beta_standardised": {f: round(float(b), 4) for f, b in zip(feats, beta)},
        "C": float(np.atleast_1d(final[-1].C_)[0]),
        "auc": {k: v for k, v in results["app"].items() if not k.startswith("_")},
        "auc_reference": {k: r["mean"] for k, r in results.items() if k != "app"},
        "literature_disagreements": disagree,
        # Speaker-level distributions (derived stats, no audio) – the simulator uses these for realistic effect sizes
        "feature_stats": {
            f: {
                "control_mean": round(float(spk.loc[y == 0, f"app_{f}"].mean()), 5),
                "patient_mean": round(float(spk.loc[y == 1, f"app_{f}"].mean()), 5),
                "sd": round(float(np.sqrt((spk.loc[y == 0, f"app_{f}"].var() + spk.loc[y == 1, f"app_{f}"].var()) / 2)), 5),
            }
            for f in feats
        },
        "n_speakers": int(len(spk)),
        "n_depressed": int(y.sum()),
        "trained_on": f"androids_{args.task}",
        "folds": fold_src,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not args.no_write:
        OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        OUT_WEIGHTS.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"→ {OUT_WEIGHTS.relative_to(ROOT)}")
    write_reports(results, y, feats, beta, payload, REPORTS / args.task if args.task != "interview" else REPORTS)


def write_reports(results: dict, y: np.ndarray, feats: list[str], beta: np.ndarray, payload: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    labels = {"app": "Our 8 features", "egemaps88": "All 88 eGeMAPS", "wav2vec2_probe": "wav2vec2 probe", "demographics": "Gender + age only"}

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, r in results.items():
        m = ~np.isnan(r["_oof"])
        fpr, tpr, _ = roc_curve(y[m], r["_oof"][m])
        ax.plot(fpr, tpr, lw=2 if name == "app" else 1.2, label=f"{labels.get(name, name)} (AUC {r['mean']:.2f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Depressed vs control – speaker-independent CV")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=160)

    order = np.argsort(np.abs(beta))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    colors = ["#b4443a" if b > 0 else "#2f6f62" for b in beta[order]]
    ax.barh([feats[i] for i in order], beta[order], color=colors)
    ax.axvline(0, c="grey", lw=0.8)
    ax.set(xlabel="Standardised coefficient (→ more likely depressed)", title="What the voice model listens for")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "coefficients.png", dpi=160)

    rows = "\n".join(
        f"| {labels.get(k, k)} | {r['mean']:.3f} ± {r['std']:.3f} | {r['pooled']:.3f} |" for k, r in results.items()
    )
    coef_rows = "\n".join(
        f"| `{f}` | {b:+.3f} | {payload['weights'][f]:.3f} | {'⚠️ sign flips (collinearity) – literature direction kept' if f in payload['literature_disagreements'] else 'agrees'} |"
        for f, b in sorted(zip(feats, beta), key=lambda t: -abs(t[1]))
    )
    (out / "summary.md").write_text(f"""# Acoustic branch – training summary

Trained {payload['trained_at']} on `{payload['trained_on']}` · {payload['n_speakers']} speakers ({payload['n_depressed']} depressed) · folds: {payload['folds']}

## Speaker-independent AUC

| Model | AUC (mean ± sd over folds) | Pooled out-of-fold |
|---|---|---|
{rows}

## Coefficients (standardised; + = higher value → more likely depressed)

| Feature | β | Weight in app | vs. literature |
|---|---|---|---|
{coef_rows}

![ROC](roc.png) ![Coefficients](coefficients.png)

**Limitations:** between-person classifier used as a prior for within-person change · small sample · Italian speech (prosody transfers; content is handled by the text branch).
""")
    print(f"→ {out.relative_to(ROOT)}/{{roc.png, coefficients.png, summary.md}}")


if __name__ == "__main__":
    main()
