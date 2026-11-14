"""Change-detection weights from the Androids Corpus (STATUS.md: «Vekter for endring», T13 + T15).

The app compares a person with their own baseline, so a feature should count by how far depression
moves it *relative to that person's normal clip-to-clip variation*, and correlated features
(energy/arousal) must not be counted twice. That is Fisher's linear discriminant in within-person units:

    Σ_w   pooled within-speaker covariance (Androids has ~5 clips per speaker)
    s     within-speaker SD  (sqrt diag Σ_w),   R = within-speaker correlation
    δ     (mean depressed − mean control) / s   (group difference in within-person SD units)
    w     = (R + shrinkage)⁻¹ δ                 (Fisher direction in z-space)

Signs are constrained to the literature direction (features whose weight flips are dropped and the
rest re-solved): a flip here is a suppressor effect through correlation, which is fragile out of sample.

RESULT (14.11): on held-out folds every weighting (equal / classifier / per-feature / Fisher) gives the
same detectability within noise, so the app uses EQUAL weights with literature directions. What the data
does support: the features and their directions (classifier AUC), and the within-person correlation,
which the engine uses to keep the score's noise at SD 1 (fewer false alarms).

"Detectability" = expected shift of the unit-variance score D for a full control→patient change:
    det = w·Δμ / sqrt(w' Σ_w w)       (raw units; evaluated on held-out fold speakers)
Everything comes from corpus data – nothing from our own recordings.
"""

import numpy as np
import pandas as pd

SHRINK = 0.1
MIN_CLIP_S = 3.0


def _within_stats(clips: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Pooled within-speaker covariance and the group mean difference (depressed − control)."""
    c = clips[clips["duration_s"] >= MIN_CLIP_S]
    X = c[cols].to_numpy(float)
    E = X - c.groupby("speaker")[cols].transform("mean").to_numpy(float)
    n_spk = c["speaker"].nunique()
    cov = E.T @ E / max(len(c) - n_spk, 1)
    spk_mean = c.groupby("speaker")[cols].mean()
    lab = c.groupby("speaker")["label"].first()
    dmu = spk_mean[lab == 1].mean().to_numpy() - spk_mean[lab == 0].mean().to_numpy()
    return cov, dmu, n_spk


def fisher(cov: np.ndarray, dmu: np.ndarray, literature: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """Sign-constrained, shrunk Fisher weights in z-space (z = (x − μ)/s, no direction flip)."""
    s = np.sqrt(np.diag(cov))
    R = cov / np.outer(s, s)
    Rl = (1 - shrink) * R + shrink * np.eye(len(s))
    delta = dmu / s
    active = np.ones(len(s), bool)
    w = np.zeros(len(s))
    while active.any():
        idx = np.flatnonzero(active)
        w = np.zeros(len(s))
        w[idx] = np.linalg.solve(Rl[np.ix_(idx, idx)], delta[idx])
        flipped = active & (np.sign(w) != literature) & (w != 0)
        if not flipped.any():
            break
        active &= ~flipped
    return w


def detectability(w_z: np.ndarray, s_train: np.ndarray, cov: np.ndarray, dmu: np.ndarray) -> float:
    w_raw = w_z / s_train
    return float(w_raw @ dmu / np.sqrt(w_raw @ cov @ w_raw))


def fit(clips: pd.DataFrame, feats: list[str], literature: dict[str, int],
        classifier_weights: dict[str, float], folds: dict[str, int],
        classifier_fold_weights: dict[int, dict[str, float]] | None = None, use: str = "equal") -> dict:
    """classifier_fold_weights[k] = classifier weights refitted without fold k (fair held-out comparison).

    `use` picks the weighting the app gets. Default "equal": on held-out folds all weightings tie
    (1.54–1.61 while folds range 0.8–3.1), so the data cannot tell them apart and the simplest one
    without fitting wins (STATUS.md B12). The others are still computed and reported.
    """
    cols = [f"app_{f}" for f in feats]
    lit = np.array([literature[f] for f in feats], float)
    cls_w = np.array([classifier_weights[f] * literature[f] for f in feats])

    fold_cls = {k: np.array([w[f] * literature[f] for f in feats]) for k, w in (classifier_fold_weights or {}).items()}
    current_fold: list[int] = [-1]
    methods = {
        "equal": lambda cov, dmu: lit.copy(),
        "classifier": lambda cov, dmu: fold_cls.get(current_fold[0], cls_w),
        "per_feature": lambda cov, dmu: np.where(np.sign(dmu) == lit, dmu / np.sqrt(np.diag(cov)), 0.0),
        "fisher": lambda cov, dmu: fisher(cov, dmu, lit),
    }
    held_out = {m: [] for m in methods}
    for k in sorted(set(folds.values())):
        test = clips["speaker"].map(folds) == k
        if clips.loc[test, "label"].nunique() < 2:
            continue
        current_fold[0] = k
        cov_tr, dmu_tr, _ = _within_stats(clips[~test], cols)
        cov_te, dmu_te, _ = _within_stats(clips[test], cols)
        s_tr = np.sqrt(np.diag(cov_tr))
        for m, make in methods.items():
            held_out[m].append(round(detectability(make(cov_tr, dmu_tr), s_tr, cov_te, dmu_te), 3))

    current_fold[0] = -1
    cov, dmu, n_spk = _within_stats(clips, cols)
    s = np.sqrt(np.diag(cov))
    w = methods[use](cov, dmu)
    total = np.abs(w).sum() or 1.0
    R = cov / np.outer(s, s)
    return {
        "weights": {f: round(float(abs(x) / total), 4) for f, x in zip(feats, w)},
        "direction": {f: literature[f] for f in feats},
        "fisher_dropped_sign_flip": [f for f, x in zip(feats, fisher(cov, dmu, lit)) if x == 0],
        "within_corr": {"features": feats, "matrix": [[round(float(v), 4) for v in row] for row in R]},
        "within_sd": {f: round(float(v), 6) for f, v in zip(feats, s)},
        "delta_within": {f: round(float(v), 3) for f, v in zip(feats, dmu / s)},
        "detectability": {
            "in_sample": {m: round(detectability(make(cov, dmu), s, cov, dmu), 3) for m, make in methods.items()},
            "held_out_mean": {m: round(float(np.mean(v)), 3) for m, v in held_out.items()},
            "held_out_per_fold": held_out,
        },
        "used": use,
        "fisher_weights": {f: round(float(x), 4) for f, x in zip(feats, fisher(cov, dmu, lit))},
        "shrinkage": SHRINK,
        "n_speakers": int(n_spk),
        "note": "Held-out detectability ties across weightings, so equal weights are used; Fisher kept for reference. "
                "Within-SD is from clips of one session; day-to-day noise is larger.",
    }
