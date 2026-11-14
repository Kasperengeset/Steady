"""Text branch weights: v1 log-odds + 3 lexicon features, on the same author split as train_v1.

    backend/.venv/bin/python ml/text/train_lexicon.py        # after train_v1.py, ~1 min

Scorers, all tested once on the v1 test split:
- lexicon only:  logistic regression on [absolutist, first_person, sentiment], fit on train.
- v1 + lexicon:  logistic regression on [text_logit + the three], fit on validation
                 (v1 never saw validation, so its log-odds there are honest).
- equal weights: mean of signed z-scores scaled on validation; no labels (what the engine does
                 without a weights file).

Branch weights (engine/weights/text.json) follow the acoustic convention (STATUS.md):
weights = |standardised beta| of "v1 + lexicon", normalised to sum 1; direction = literature
(a sign that flips because of collinearity keeps the literature direction, B6).

Writes backend/app/engine/weights/text.json and ml/reports/text/lexicon.md.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_v1 import REPORTS, ROOT, SEED, build_dataset, log, tp

LEX = ["absolutist", "first_person", "sentiment"]
ALL = ["text_logit"] + LEX
DIRECTION = {"text_logit": 1, "absolutist": 1, "first_person": 1, "sentiment": -1}  # literature
DIARY = ROOT / "ml" / "text" / "diary_sanity.csv"
WEIGHTS = ROOT / "backend" / "app" / "engine" / "weights" / "text.json"


def add_features(df: pd.DataFrame, prepared_col: str) -> pd.DataFrame:
    feats = pd.DataFrame([{"text_logit": tp.text_logit(t), **tp.lexicon_features(t)}
                          for t in df[prepared_col]], index=df.index)
    return pd.concat([df, feats], axis=1)


def equal_weight(df: pd.DataFrame, stats: dict) -> np.ndarray:
    return np.mean([DIRECTION[c] * (df[c] - stats[c][0]) / stats[c][1] for c in ALL], axis=0)


def pairs_low_above_good(s: pd.Series, labels: pd.Series) -> int:
    low, good = s[labels == "low"], s[labels == "good"]
    return int(sum(lo > g for lo in low for g in good))


def main() -> None:
    tp._model()  # fail early if the model was not exported
    log("rebuilding the v1 split")
    data = build_dataset()
    log(f"computing features for {len(data)} chunks (pure-Python model, as in the app)")
    data = add_features(data, "text_masked")
    tr, va, te = (data[data["split"] == s] for s in ("train", "val", "test"))

    lex_only = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=SEED))
    lex_only.fit(tr[LEX], tr["label"])
    stacked = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=SEED))
    stacked.fit(va[ALL], va["label"])
    stats = {c: (float(va[c].mean()), float(va[c].std())) for c in ALL}

    y = te["label"].to_numpy()
    rows = []
    for name, p, has_f1 in [
        ("v1 alone", 1 / (1 + np.exp(-te["text_logit"])), True),
        ("lexicon only", lex_only.predict_proba(te[LEX])[:, 1], True),
        ("v1 + lexicon", stacked.predict_proba(te[ALL])[:, 1], True),
        ("equal weights", equal_weight(te, stats), False),
    ]:
        rows.append({"scorer": name, "f1": f1_score(y, (p >= 0.5).astype(int)) if has_f1 else np.nan,
                     "roc_auc": roc_auc_score(y, p)})
    rows += [{"scorer": f"{c} alone", "f1": np.nan, "roc_auc": roc_auc_score(y, DIRECTION[c] * te[c])} for c in LEX]
    reddit = pd.DataFrame(rows)
    log("Reddit test:\n" + reddit.round(3).to_string(index=False))
    means = te.groupby("label")[LEX].mean().rename(index={0: "other", 1: "depression"})

    # Diary sanity set
    diary = pd.read_csv(DIARY)
    diary["prepared"] = diary["text"].map(tp.prepare)
    diary = add_features(diary, "prepared")
    scorers = {
        "v1 alone": diary["text_logit"],
        "lexicon only": pd.Series(lex_only.predict_proba(diary[LEX])[:, 1]),
        "v1 + lexicon": pd.Series(stacked.predict_proba(diary[ALL])[:, 1]),
        "equal weights": pd.Series(equal_weight(diary, stats)),
        **{f"{c} alone": DIRECTION[c] * diary[c] for c in LEX},
    }
    written = diary["variant"] == "as written"
    rewrite = (diary["label"] == "low") | ~written
    drows = []
    for name, s in scorers.items():
        p = 1 / (1 + np.exp(-s)) if name == "v1 alone" else s
        has_prob = name in ("v1 alone", "lexicon only", "v1 + lexicon")
        drows.append({
            "scorer": name,
            "correct_of_10": int(((p[written] >= 0.5) == (diary.loc[written, "label"] == "low")).sum()) if has_prob else None,
            "pairs_as_written": pairs_low_above_good(s[written], diary.loc[written, "label"]),
            "pairs_I_added": pairs_low_above_good(s[rewrite], diary.loc[rewrite, "label"]),
        })
    diary_res = pd.DataFrame(drows)
    log("Diary set (pairs of 25):\n" + diary_res.to_string(index=False))

    # Branch weights for the engine
    beta = pd.Series(stacked[-1].coef_[0], index=ALL)
    w = beta.abs() / beta.abs().sum()
    disagree = [c for c in ALL if np.sign(beta[c]) != DIRECTION[c]]
    v1_auc = float(reddit.loc[reddit.scorer == "v1 alone", "roc_auc"].iloc[0])
    WEIGHTS.write_text(json.dumps({
        "branch": "text",
        "weights": {c: round(float(w[c]), 4) for c in ALL},
        "direction": DIRECTION,
        "beta_standardised": {c: round(float(beta[c]), 4) for c in ALL},
        "fit_on": "Reddit MH validation split (v1 out-of-sample), tested on the v1 test split",
        "auc": {"v1_alone": round(v1_auc, 4),
                "v1_plus_lexicon": round(float(reddit.loc[reddit.scorer == 'v1 + lexicon', 'roc_auc'].iloc[0]), 4),
                "lexicon_only": round(float(reddit.loc[reddit.scorer == 'lexicon only', 'roc_auc'].iloc[0]), 4)},
        "literature_disagreements": disagree,
        "note": "Lexicon features add no accuracy over v1 on Reddit (v1 already uses those words), so "
                "text_logit carries almost all weight. They stay in the output to explain the score. "
                "Gemini content features (mood, energy_self, anhedonia, social_contact) get weight 0 "
                "until the real Gemini call replaces the stub.",
        "trained_on": str(date.today()),
    }, indent=2) + "\n", encoding="utf-8")
    log(f"weights {w.round(3).to_dict()} -> {WEIGHTS.relative_to(ROOT)}")

    lines = [
        "# Text branch: v1 + lexicon features", "",
        f"Generated by `ml/text/train_lexicon.py` on {date.today()}. Same author split and chunks as v1. "
        "Features are computed on the masked text with the app's own code.", "",
        "- **absolutist**: absolutist words per 100 words (Al-Mosaiwi and Johnstone 2018), skipping a word after a negation",
        "- **first_person**: i, me, my, mine, myself per 100 words",
        "- **sentiment**: VADER compound, -1 to +1", "",
        "## Mean feature values, Reddit test", "",
        "| | " + " | ".join(LEX) + " |", "|---|" + "---|" * len(LEX),
    ]
    lines += [f"| {i} | " + " | ".join(f"{r[c]:.2f}" for c in LEX) + " |" for i, r in means.iterrows()]
    lines += ["", "## Reddit test (balanced 1:1, chance ROC-AUC 0.5)", "", "| Scorer | F1 | ROC-AUC |", "|---|---|---|"]
    lines += [f"| {r.scorer} | {'' if np.isnan(r.f1) else f'{r.f1:.3f}'} | {r.roc_auc:.3f} |" for r in reddit.itertuples()]
    lines += ["", "Engine weights (`backend/app/engine/weights/text.json`): "
              + ", ".join(f"{c} {w[c]:.3f}" for c in ALL) + ". Standardised betas: "
              + ", ".join(f"{c} {beta[c]:+.2f}" for c in ALL) + ".", "",
              "## Diary sanity set", "",
              "Pairs: how many of the 25 (low day, good day) pairs put the low day higher. Chance 12.5. "
              "\"I added\" uses the good days rewritten with I and my.", "",
              "| Scorer | Correct of 10 | Pairs, as written | Pairs, I added |", "|---|---|---|---|"]
    for r in diary_res.itertuples():
        c = "" if r.correct_of_10 is None or pd.isna(r.correct_of_10) else int(r.correct_of_10)
        lines.append(f"| {r.scorer} | {c} | {r.pairs_as_written} | {r.pairs_I_added} |")
    (REPORTS / "lexicon.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("saved ml/reports/text/lexicon.md")


if __name__ == "__main__":
    main()
