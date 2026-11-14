"""Text branch model: TF-IDF + logistic regression on the Reddit Mental Health Dataset.

Positive: r/depression. Negative: everyday forums plus two hard negatives (relationships, divorce:
emotional, but not depression). Uses the app's own text code (backend/app/pipeline/text.py).

    backend/.venv/bin/python ml/text/train_v1.py                          # Mac/Linux, ~1 min
    backend\\.venv\\Scripts\\python ml\\text\\train_v1.py                     # Windows
    ... ml/text/train_v1.py --max-posts-per-sub 1500                      # smoke test, no export

Trains two variants on the same chunks: masked (forum and treatment words removed; the app uses
this one) and unmasked (comparison only). Split by author 70/15/15, asserted.

Writes:
    backend/app/pipeline/models/text_tfidf_lr.json.gz   the model the app loads (full run only)
    ml/reports/text/v1.md, ml/reports/text/v1_confusion.png
"""

import argparse
import gzip
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import text as tp  # noqa: E402

RAW = ROOT / "data" / "reddit"
REPORTS = ROOT / "ml" / "reports" / "text"

SEED = 42
POSITIVE = ["depression"]
EVERYDAY = ["fitness", "parenting", "personalfinance", "meditation", "teaching"]
HARD = ["relationships", "divorce"]
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_posts(max_per_sub: int | None) -> pd.DataFrame:
    frames = []
    for sub in POSITIVE + EVERYDAY + HARD:
        path = RAW / f"{sub}_pre_features_tfidf_256.csv"
        if not path.exists():
            sys.exit(f"{path} missing – run: python ml/download_data.py --reddit")
        df = pd.read_csv(path, usecols=["author", "post"]).dropna(subset=["post"])
        df = df[~df["post"].str.strip().isin(["[removed]", "[deleted]"])]
        if max_per_sub and len(df) > max_per_sub:
            df = df.sample(n=max_per_sub, random_state=SEED)
        df["subreddit"] = sub
        frames.append(df)
    posts = pd.concat(frames, ignore_index=True)
    # Cross-posts and copy-paste would otherwise sit in two splits.
    posts = posts.drop_duplicates(subset="post").reset_index(drop=True)
    posts["label"] = posts["subreddit"].isin(POSITIVE).astype(int)
    return posts


def split_by_author(posts: pd.DataFrame) -> pd.DataFrame:
    """70 / 15 / 15 by author. No author may appear in two splits."""
    groups = posts["author"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, rest_idx = next(gss.split(posts, groups=groups))
    rest = posts.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(gss2.split(rest, groups=groups.iloc[rest_idx]))

    posts = posts.copy()
    posts["split"] = ""
    posts.loc[posts.index[train_idx], "split"] = "train"
    posts.loc[rest.index[val_rel], "split"] = "val"
    posts.loc[rest.index[test_rel], "split"] = "test"
    assert_author_split(posts)
    return posts


def assert_author_split(df: pd.DataFrame) -> None:
    authors = {s: set(df.loc[df["split"] == s, "author"].astype(str)) for s in ("train", "val", "test")}
    assert not authors["train"] & authors["val"], "author leak: train/val"
    assert not authors["train"] & authors["test"], "author leak: train/test"
    assert not authors["val"] & authors["test"], "author leak: val/test"


def make_chunks(posts: pd.DataFrame) -> pd.DataFrame:
    """Clean, cut into 50 to 100-word chunks (max 3 per post), then mask.
    Chunking happens before masking so both variants see the same chunks."""
    rows = []
    for r in posts.itertuples(index=False):
        for c in tp.chunk_words(tp.clean_text(r.post)):
            rows.append((r.author, r.subreddit, r.label, r.split, c))
    chunks = pd.DataFrame(rows, columns=["author", "subreddit", "label", "split", "text"])
    chunks["text_masked"] = chunks["text"].map(tp.mask_text)
    return chunks


def balance(chunks: pd.DataFrame) -> pd.DataFrame:
    """Downsample negatives to 1:1 inside each split."""
    parts = []
    for split in ("train", "val", "test"):
        s = chunks[chunks["split"] == split]
        pos, neg = s[s["label"] == 1], s[s["label"] == 0]
        n = min(len(pos), len(neg))
        parts += [pos.sample(n=n, random_state=SEED), neg.sample(n=n, random_state=SEED)]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_dataset(max_per_sub: int | None = None) -> pd.DataFrame:
    data = balance(make_chunks(split_by_author(load_posts(max_per_sub))))
    assert_author_split(data)
    return data


def scores(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "accuracy": accuracy_score(y, pred)}


def train_variant(data: pd.DataFrame, text_col: str, name: str) -> dict:
    tr, va, te = (data[data["split"] == s] for s in ("train", "val", "test"))
    # 1 to 2-grams from the app's analyzer: keeps "i", no bigram crosses a masked word.
    vec = TfidfVectorizer(analyzer=tp.analyze, min_df=5, sublinear_tf=True, max_features=50_000)
    log(f"{name}: fitting TF-IDF on {len(tr)} train chunks")
    Xtr = vec.fit_transform(tr[text_col])
    Xva = vec.transform(va[text_col])

    grid = []
    for C in C_GRID:
        lr = LogisticRegression(C=C, max_iter=2000, solver="liblinear", class_weight="balanced",
                                random_state=SEED).fit(Xtr, tr["label"])
        s = scores(va["label"].to_numpy(), lr.predict_proba(Xva)[:, 1])
        grid.append({"C": C, **s})
        log(f"{name}: C={C:<5} val ROC-AUC {s['roc_auc']:.4f}  F1 {s['f1']:.4f}")
    best = max(grid, key=lambda g: (round(g["roc_auc"], 4), -g["C"]))

    lr = LogisticRegression(C=best["C"], max_iter=2000, solver="liblinear", class_weight="balanced",
                            random_state=SEED).fit(Xtr, tr["label"])
    pipe = Pipeline([("tfidf", vec), ("lr", lr)])

    # Test is touched once, here.
    p_test = pipe.predict_proba(te[text_col])[:, 1]
    test = scores(te["label"].to_numpy(), p_test)
    log(f"{name}: best C={best['C']}, test ROC-AUC {test['roc_auc']:.4f}  F1 {test['f1']:.4f}")

    terms = vec.get_feature_names_out()
    order = np.argsort(lr.coef_[0])
    te = te.assign(p=p_test, pred=(p_test >= 0.5).astype(int))
    per_sub = (te.groupby("subreddit").agg(n=("p", "size"), flagged=("pred", "mean"), mean_score=("p", "mean"))
               .reindex(POSITIVE + EVERYDAY + HARD).dropna())
    return {"name": name, "pipe": pipe, "best_C": best["C"], "grid": grid, "test": test,
            "top_pos": [(terms[i], lr.coef_[0][i]) for i in order[::-1][:30]],
            "top_neg": [(terms[i], lr.coef_[0][i]) for i in order[:30]],
            "per_sub": per_sub, "cm": confusion_matrix(te["label"], te["pred"]), "test_df": te}


def export_model(r: dict, n_posts: int, n_authors: int) -> Path:
    vec, lr = r["pipe"].named_steps["tfidf"], r["pipe"].named_steps["lr"]
    payload = {
        "name": "text_tfidf_lr",
        "trained_on": str(date.today()),
        "data": "Reddit Mental Health Dataset (Low et al. 2020), pre window; r/depression vs "
                + ", ".join(EVERYDAY + HARD),
        "n_posts": n_posts, "n_authors": n_authors,
        "split": "by author 70/15/15, negatives downsampled 1:1 per split",
        "mask": True, "C": r["best_C"],
        "test": {k: round(v, 4) for k, v in r["test"].items()},
        "terms": vec.get_feature_names_out().tolist(),
        "idf": [round(float(x), 6) for x in vec.idf_],
        "coef": [round(float(x), 6) for x in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
    }
    tp.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tp.MODEL_PATH, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    tp._model.cache_clear()

    # The app scores in pure Python. Check it matches scikit-learn.
    sample = r["test_df"].sample(n=300, random_state=SEED)["text_masked"].tolist()
    sk = r["pipe"].decision_function(sample)
    ours = np.array([tp.text_logit(t) for t in sample])
    diff = float(np.max(np.abs(sk - ours)))
    assert diff < 1e-3, f"exported model disagrees with scikit-learn (max diff {diff})"
    log(f"exported {tp.MODEL_PATH.relative_to(ROOT)} ({tp.MODEL_PATH.stat().st_size / 1e6:.1f} MB), "
        f"max logit diff vs scikit-learn {diff:.1e}")
    return tp.MODEL_PATH


def save_confusion(results: list[dict]) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    for ax, r in zip(np.atleast_1d(axes), results):
        ConfusionMatrixDisplay(r["cm"], display_labels=["other", "depression"]).plot(
            ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"Text v1, {r['name']}\nF1 {r['test']['f1']:.3f}, ROC-AUC {r['test']['roc_auc']:.3f}")
    fig.tight_layout()
    path = REPORTS / "v1_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(results, data, posts, args) -> Path:
    def n(split, label=None):
        d = data[data["split"] == split]
        return len(d) if label is None else int((d["label"] == label).sum())

    lines = [
        "# Text v1 results", "",
        f"Generated by `ml/text/train_v1.py` on {date.today()}."
        + (f" Smoke test: at most {args.max_posts_per_sub} posts per subreddit." if args.max_posts_per_sub else ""),
        "",
        "TF-IDF (1 to 2-grams, min_df 5, 50k features) + logistic regression. Chunks of 50 to 100 words, "
        "at most 3 per post. Split by author 70/15/15, then negatives downsampled to 1:1 inside each split. "
        "C tuned on validation, test used once.",
        "", "## Data", "",
        f"- Posts after cleaning and de-duplication: {len(posts)}. Authors: {posts['author'].nunique()}.",
        "", "| Split | Chunks | Depression | Other |", "|---|---|---|---|",
    ]
    lines += [f"| {s} | {n(s)} | {n(s, 1)} | {n(s, 0)} |" for s in ("train", "val", "test")]
    lines += [
        "", "## Test results", "",
        "Test is balanced 1:1, so chance is 0.5 for accuracy, ROC-AUC and PR-AUC. "
        "Predicting \"depression\" for every chunk gives F1 0.667.", "",
        "| Variant | C | F1 | ROC-AUC | PR-AUC | Accuracy |", "|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r["test"]
        lines.append(f"| {r['name']} | {r['best_C']} | {t['f1']:.3f} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} | {t['accuracy']:.3f} |")
    lines += [
        "| baseline (all \"depression\") | | 0.667 | 0.500 | 0.500 | 0.500 |", "",
        "Masked replaces forum names (`depress*`, `subreddit`, `meditat*`, `divorc*`, `fitness`, `parenting`, "
        "`teaching`, `personal finance`) and treatment words (therapy, meds, doctor, symptoms, side effects, "
        "mental health, TMS, ECT, ketamine, antidepressant names) with a break that no bigram crosses. Both "
        "variants drop Reddit formatting (tl;dr, edit:, age/gender tags). The app uses the masked model. "
        "Full list in `backend/app/pipeline/text.py`.",
        "", "## Share flagged per subreddit (test, masked)", "",
        "For r/depression this is recall. For the others it is the false positive rate.", "",
        "| Subreddit | Chunks | Flagged | Mean score |", "|---|---|---|---|",
    ]
    masked = next(r for r in results if r["name"] == "masked")
    for sub, row in masked["per_sub"].iterrows():
        lines.append(f"| {sub} | {int(row['n'])} | {row['flagged']:.1%} | {row['mean_score']:.3f} |")
    lines += ["", "## Validation grid", "", "| Variant | C | val ROC-AUC | val F1 |", "|---|---|---|---|"]
    for r in results:
        lines += [f"| {r['name']} | {g['C']} | {g['roc_auc']:.4f} | {g['f1']:.4f} |" for g in r["grid"]]
    lines += ["", "## Top 30 coefficients (masked)", "", "Check: these should be mood words, not topic words.", "",
              "| Towards depression | coef | Away from depression | coef |", "|---|---|---|---|"]
    for (tpos, cp), (tneg, cn) in zip(masked["top_pos"], masked["top_neg"]):
        lines.append(f"| {tpos} | {cp:+.2f} | {tneg} | {cn:+.2f} |")
    path = REPORTS / "v1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-posts-per-sub", type=int, default=None)
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    REPORTS.mkdir(parents=True, exist_ok=True)

    log("loading posts")
    posts = split_by_author(load_posts(args.max_posts_per_sub))
    log(f"{len(posts)} posts, {posts['author'].nunique()} authors")
    chunks = make_chunks(posts)
    assert_author_split(chunks)
    data = balance(chunks)
    assert_author_split(data)
    log(f"{len(chunks)} chunks, {len(data)} after 1:1 balancing")

    results = [train_variant(data, "text_masked", "masked"), train_variant(data, "text", "unmasked")]
    if args.max_posts_per_sub:
        log("smoke test: model not exported")
    else:
        export_model(results[0], len(posts), posts["author"].nunique())
    png, md = save_confusion(results), write_report(results, data, posts, args)
    log(f"saved {png.relative_to(ROOT)} and {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
