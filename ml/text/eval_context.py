"""Context test for the text branch (#28): TF-IDF alone vs Gemini content scores vs both.

    backend/.venv/bin/python ml/text/eval_context.py              # needs GEMINI_API_KEY (.env)
    backend/.venv/bin/python ml/text/eval_context.py --cached     # reuse the last Gemini replies

ONLY team-written text (ml/text/context_set.csv). Never Reddit or Androids data (STATUS.md B2).
Each entry is rated REPEATS times (consistency). Raw replies are cached locally in
data/text/context_ratings.json (gitignored), the report has only summary numbers:
ml/reports/text/context_check.md.

Scores, higher = worse for all three:
  tfidf     text_logit from the app's model (pipeline/text.py)
  gemini    badness = mean(10 − mood, 10 − energy_self, anhedonia, 10 − social_contact), 0–10
  combined  (z_tfidf + z_gemini) / 2, equal weight on z-scores (as B12)
Each score is put on a z-scale with the plain anchors: z = (x − midpoint of the good and low anchor
means) / pooled anchor SD, so z > 0 means "on the low side". Same rule for every scorer.

Gate (#28, set before running): weight only if the combined scorer
  1. is right on ≥ 75 % of the context entries (non-anchor) and beats TF-IDF there,
  2. is no worse than TF-IDF on the plain anchors,
  3. orders ≥ 90 % of the pairs (low above its good partner),
  4. has a mean SD over repeats ≤ 0.5 points and < 10 % of the anchors' good/low gap.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import llm, text  # noqa: E402

DATA = ROOT / "ml" / "text" / "context_set.csv"
CACHE = ROOT / "data" / "text" / "context_ratings.json"
REPORT = ROOT / "ml" / "reports" / "text" / "context_check.md"
REPEATS = 5


def badness(scores: dict[str, float]) -> float:
    return statistics.fmean([10 - scores["mood"], 10 - scores["energy_self"], scores["anhedonia"], 10 - scores["social_contact"]])


def rate_all(df: pd.DataFrame) -> dict[str, list[dict]]:
    jobs = [(r.id, r.text) for r in df.itertuples() for _ in range(REPEATS)]

    def one(job):
        t0 = time.time()
        out = llm.rate_text(job[1])
        return job[0], out, time.time() - t0

    res: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (eid, out, dt) in enumerate(ex.map(one, jobs), 1):
            res.setdefault(eid, []).append({"rating": out, "latency_s": round(dt, 2)})
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    return res


def zscale(df: pd.DataFrame, col: str) -> pd.Series:
    a = df[df["category"] == "anchor"]
    good, low = a.loc[a["label"] == "good", col], a.loc[a["label"] == "low", col]
    sd = ((good.var() + low.var()) / 2) ** 0.5 or 1.0
    return (df[col] - (good.mean() + low.mean()) / 2) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    if args.cached:
        ratings = json.loads(CACHE.read_text())
    else:
        ratings = rate_all(df)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(ratings))

    df["tfidf"] = [text.analyze_text(t)["features"]["text_logit"] for t in df["text"]]
    rep_bad, rep_sd, lat, failed = [], [], [], 0
    sd_per_score = {f: [] for f in llm.RATE_FIELDS}
    for eid in df["id"]:
        rs = [r["rating"] for r in ratings[eid] if r["rating"]]
        failed += REPEATS - len(rs)
        lat += [r["latency_s"] for r in ratings[eid]]
        rep_bad.append(statistics.fmean(badness(r["scores"]) for r in rs) if rs else float("nan"))
        for f in llm.RATE_FIELDS:
            if len(rs) > 1:
                sd_per_score[f].append(statistics.stdev(r["scores"][f] for r in rs))
    df["gemini"] = rep_bad
    for c in ("tfidf", "gemini"):
        df[f"z_{c}"] = zscale(df, c)
    df["z_combined"] = (df["z_tfidf"] + df["z_gemini"]) / 2
    low = df["label"] == "low"
    scorers = ("tfidf", "gemini", "combined")
    for c in scorers:
        df[f"ok_{c}"] = (df[f"z_{c}"] > 0) == low

    cats = [c for c in df["category"].unique() if c != "anchor"]
    ctx = df["category"] != "anchor"
    share = {c: {k: float(df.loc[ctx, f"ok_{k}"].mean()) for k in scorers} for c in ["context"]}
    share["anchor"] = {k: float(df.loc[~ctx, f"ok_{k}"].mean()) for k in scorers}
    per_cat = {cat: {k: float(df.loc[df["category"] == cat, f"ok_{k}"].mean()) for k in scorers} for cat in cats}

    pairs = df[df["pair"].notna()].groupby("pair")
    pair_ok = {k: [] for k in scorers}
    for _, g in pairs:
        if set(g["label"]) == {"good", "low"}:
            for k in scorers:
                pair_ok[k].append(float(g.loc[g["label"] == "low", f"z_{k}"].iloc[0] > g.loc[g["label"] == "good", f"z_{k}"].iloc[0]))
    pair_share = {k: statistics.fmean(v) for k, v in pair_ok.items()}

    a = df[~ctx]
    gap = float(a.loc[a["label"] == "low", "gemini"].mean() - a.loc[a["label"] == "good", "gemini"].mean())
    mean_sd = statistics.fmean(v for vs in sd_per_score.values() for v in vs) if any(sd_per_score.values()) else float("nan")
    r = float(df["tfidf"].corr(df["gemini"]))

    gate = {
        "context ≥ 75 % and beats TF-IDF": share["context"]["combined"] >= 0.75 and share["context"]["combined"] > share["context"]["tfidf"],
        "anchors no worse than TF-IDF": share["anchor"]["combined"] >= share["anchor"]["tfidf"],
        "pairs ≥ 90 %": pair_share["combined"] >= 0.9,
        "repeat SD ≤ 0.5 and < 10 % of gap": mean_sd <= 0.5 and mean_sd < 0.1 * gap,
    }
    passed = all(gate.values())

    def pct(x):
        return f"{100 * x:.0f} %"

    cat_rows = "\n".join(f"| {c} | {int((df['category'] == c).sum())} | " + " | ".join(pct(per_cat[c][k]) for k in scorers) + " |" for c in cats)
    sd_rows = " · ".join(f"{f} {statistics.fmean(v):.2f}" for f, v in sd_per_score.items() if v)
    no = df["lang"] == "no"
    lang_row = " | ".join(pct(float(df.loc[no, f"ok_{k}"].mean())) for k in scorers)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Context check: TF-IDF vs Gemini content scores (#28)

Generated by `ml/text/eval_context.py` on `ml/text/context_set.csv` ({len(df)} team-written entries,
{int(no.sum())} Norwegian; never Reddit or Androids data). Gemini: `{llm.settings.rate_model}`, temperature 0,
strict JSON schema, `store=False`, each entry rated {REPEATS} times. Failed or rejected replies: {failed} of {REPEATS * len(df)}.

**Gate: {'PASSED' if passed else 'NOT PASSED'}** → {'text_logit 0.5 and each Gemini score 0.125 (per #28)' if passed else 'the Gemini scores stay at weight 0'} (STATUS.md B33).

| Gate criterion (combined scorer) | Result |
|---|---|
""" + "\n".join(f"| {k} | {'✅' if v else '❌'} |" for k, v in gate.items()) + f"""

## Share on the correct side

All scorers are put on the same anchor-based z-scale (z > 0 = low side), so the comparison is fair.

| | n | TF-IDF | Gemini | Combined |
|---|---|---|---|---|
| **Context entries (all non-anchor)** | {int(ctx.sum())} | {pct(share['context']['tfidf'])} | {pct(share['context']['gemini'])} | {pct(share['context']['combined'])} |
| Plain anchors | {int((~ctx).sum())} | {pct(share['anchor']['tfidf'])} | {pct(share['anchor']['gemini'])} | {pct(share['anchor']['combined'])} |
{cat_rows}
| Norwegian entries | {int(no.sum())} | {lang_row} |

Pair order (low above its good partner, {len(pair_ok['tfidf'])} pairs): TF-IDF {pct(pair_share['tfidf'])}, Gemini {pct(pair_share['gemini'])}, combined {pct(pair_share['combined'])}.

## Consistency and agreement

- Mean SD over {REPEATS} repeats: **{mean_sd:.2f} points** ({sd_rows}); anchors' good/low gap on the Gemini badness score: {gap:.1f} points.
- Correlation TF-IDF vs Gemini over the set: r = {r:+.2f}.
- Latency per call: median {statistics.median(lat):.1f} s, 90th percentile {sorted(lat)[int(0.9 * len(lat)) - 1]:.1f} s.

**Limits.** 40 entries written by the team, who also know what the model should do; a pass is necessary,
not sufficient. Gemini sees the text, so the app must keep `store=False` and never log text or replies.
""")
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
