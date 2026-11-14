"""Reading task vs interview task on the Androids Corpus (STATUS.md T7, GitHub #21).

Question: is read speech a better reference than free talk for the fixed daily phrase?

  backend/.venv/bin/python ml/voice/reading_vs_interview.py            # extract chunks (cached) + report
  backend/.venv/bin/python ml/voice/reading_vs_interview.py --no-extract

Needs data/features/interview_clips.csv and reading_clips.csv (ml/extract_features.py --task …).

The reading task has ONE recording per speaker (~48 s of "The North Wind and the Sun"), so a
within-person SD cannot come from separate clips. Each recording is cut into K = 5 equal parts by
duration (~10 s each, close to the ~9 s daily phrase). Part k covers roughly the same text for every
speaker, so the text effect can be removed with a two-way model (speaker + part). That is the closest
the corpus gets to "the same words, read again". Interview clips are separate turns with different
content, so they get a one-way model (speaker only), exactly as ml/change_weights.py.

Both are within ONE session and one microphone. Real day-to-day noise is larger for both tasks.

Writes (derived numbers only, no audio):
  data/features/reading_chunks_clips.csv   gitignored, resumable
  ml/reports/voice/reading_vs_interview.md
"""

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _venv import ensure_backend_venv  # noqa: E402

ensure_backend_venv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "backend"))
import change_weights  # noqa: E402
import train_weights as tw  # noqa: E402
from corpus import FEATURES, ROOT, corpus_dir, list_files, parse_meta  # noqa: E402

K = 5
MIN_PART_S = 5.0
FEATS = tw.APP_FEATURES
LIT = tw.LITERATURE
REPORT = ROOT / "ml" / "reports" / "voice" / "reading_vs_interview.md"
CHUNKS_CSV = FEATURES / "reading_chunks_clips.csv"


# --------------------------------------------------------------------------- extraction
def extract_chunks() -> None:
    from app.pipeline import audio, emotion, prosody

    base = corpus_dir()
    files = list_files("reading")
    done: set[str] = set()
    header = None
    if CHUNKS_CSV.exists():
        with CHUNKS_CSV.open() as f:
            r = csv.DictReader(f)
            header = list(r.fieldnames or [])
            done = {row["clip"] for row in r}
    fh = CHUNKS_CSV.open("a", newline="")
    writer, t0 = None, time.time()
    try:
        for i, path in enumerate(files, 1):
            meta = parse_meta(path)
            if meta is None:
                continue
            rel = str(path.relative_to(base))
            if all(f"{rel}#{k}" in done for k in range(K)):
                continue
            x = audio.load(str(path))
            edges = np.linspace(0, len(x), K + 1).astype(int)
            for k in range(K):
                clip = f"{rel}#{k}"
                seg = x[edges[k]:edges[k + 1]]
                if clip in done or len(seg) < MIN_PART_S * audio.SR:
                    continue
                full = prosody.extract_full_signal(seg)
                row = {"clip": clip, **meta, "part": k, "duration_s": len(seg) / audio.SR}
                row.update({f"app_{n}": v for n, v in prosody.from_full(full).items()})
                _, emo = emotion.run_signal(seg)
                row.update({f"app_{n}": v for n, v in emo.items()})
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=header or list(row))
                    if header is None:
                        writer.writeheader()
                writer.writerow(row)
            fh.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(files)} recordings  {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        fh.close()


# --------------------------------------------------------------------------- statistics
def within_stats(clips: pd.DataFrame, two_way: bool) -> dict:
    """Pooled within-speaker covariance; two_way also removes the part (text) effect."""
    cols = [f"app_{f}" for f in FEATS]
    c = clips[clips["duration_s"] >= change_weights.MIN_CLIP_S].copy()
    c = c[c.groupby("speaker")["clip"].transform("count") >= 2]
    X = c[cols].to_numpy(float)
    E = X - c.groupby("speaker")[cols].transform("mean").to_numpy(float)
    n_spk = c["speaker"].nunique()
    df = len(c) - n_spk
    if two_way:
        # additive speaker + part model: residual = x − speaker mean − part mean + grand mean
        E = E - (c.groupby("part")[cols].transform("mean").to_numpy(float) - X.mean(axis=0))
        df -= c["part"].nunique() - 1
    cov = E.T @ E / max(df, 1)
    s = np.sqrt(np.diag(cov))
    spk = c.groupby("speaker")[cols].mean()
    lab = c.groupby("speaker")["label"].first()
    dmu = spk[lab == 1].mean().to_numpy() - spk[lab == 0].mean().to_numpy()
    between_var = spk.var().to_numpy() - s**2 / c.groupby("speaker").size().mean()
    icc = np.clip(between_var, 0, None) / (np.clip(between_var, 0, None) + s**2)
    w = np.array([LIT[f] for f in FEATS], float)  # equal weights, literature directions (B12)
    R = cov / np.outer(s, s)
    det = float((w / s) @ dmu / np.sqrt((w / s) @ cov @ (w / s)))
    return {
        "n_clips": len(c), "n_speakers": n_spk, "median_s": float(c["duration_s"].median()),
        "within_sd": dict(zip(FEATS, s)), "between_sd": dict(zip(FEATS, np.sqrt(np.clip(between_var, 0, None)))),
        "icc": dict(zip(FEATS, icc)), "delta": dict(zip(FEATS, dmu / s)), "R": R, "det_equal": det,
        "norm_equal": float(np.sqrt(np.full(len(FEATS), 1 / len(FEATS)) @ (R * np.outer(w, w)) @ np.full(len(FEATS), 1 / len(FEATS)))),
    }


def auc_table(clips: pd.DataFrame, task: str) -> dict:
    spk = tw.speaker_table(clips)
    y = spk["label"].to_numpy()
    folds, src = tw.assign_folds(spk, task)
    out = {"app": tw.cv_auc(tw.model, spk[[f"app_{f}" for f in FEATS]].to_numpy(), y, folds)}
    eg = [c for c in spk if c.startswith("eg_")]
    if eg:
        out["egemaps88"] = tw.cv_auc(tw.model, spk[eg].to_numpy(), y, folds)
    demo = np.column_stack([(spk["gender"] == "F").astype(float), spk["age"].astype(float)])
    out["demographics"] = tw.cv_auc(tw.model, demo, y, folds)
    return {"res": out, "folds": src, "spk": spk, "y": y, "n": len(spk), "n_dep": int(y.sum())}


def directions(spk: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """Between-person Cohen's d (patients − controls) and its sign in 'worse' terms."""
    y = spk["label"]
    out = {}
    for f in FEATS:
        a, b = spk.loc[y == 1, f"app_{f}"], spk.loc[y == 0, f"app_{f}"]
        sd = np.sqrt((a.var() + b.var()) / 2)
        d = (a.mean() - b.mean()) / sd
        out[f] = (float(d), int(np.sign(d)))
    return out


def combined_auc(iv: pd.DataFrame, rd: pd.DataFrame) -> dict | None:
    """Both tasks' 8 features side by side, speakers present in both (improvement plan item 9)."""
    a, b = tw.speaker_table(iv), tw.speaker_table(rd)
    cols = [f"app_{f}" for f in FEATS]
    m = a[["speaker", "label", "gender", "age"] + cols].merge(b[["speaker"] + cols], on="speaker", suffixes=("_iv", "_rd"))
    if len(m) < 40:
        return None
    folds, _ = tw.assign_folds(m, "interview")
    y = m["label"].to_numpy()
    X = m[[c + "_iv" for c in cols] + [c + "_rd" for c in cols]].to_numpy()
    return {"n": len(m), "both": tw.cv_auc(tw.model, X, y, folds),
            "iv_only": tw.cv_auc(tw.model, m[[c + "_iv" for c in cols]].to_numpy(), y, folds),
            "rd_only": tw.cv_auc(tw.model, m[[c + "_rd" for c in cols]].to_numpy(), y, folds),
            "merged": m}


# --------------------------------------------------------------------------- report
def fmt_auc(r: dict) -> str:
    return f"{r['mean']:.3f} ± {r['std']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    if not args.no_extract:
        extract_chunks()

    iv = pd.read_csv(FEATURES / "interview_clips.csv")
    rd = pd.read_csv(FEATURES / "reading_clips.csv")
    ch = pd.read_csv(CHUNKS_CSV)

    A = {"Interview (clips)": auc_table(iv, "interview"), "Reading (whole file)": auc_table(rd, "reading"),
         "Reading (median of 5 parts)": auc_table(ch, "reading")}
    W = {"Interview clips (one-way)": within_stats(iv.assign(part=0), two_way=False),
         "Reading parts (one-way)": within_stats(ch, two_way=False),
         "Reading parts (two-way, text removed)": within_stats(ch, two_way=True)}
    dirs = {"Interview": directions(A["Interview (clips)"]["spk"]), "Reading": directions(A["Reading (whole file)"]["spk"])}
    comb = combined_auc(iv, rd)

    for name, a in A.items():
        print(f"{name:30s} n={a['n']} AUC app {fmt_auc(a['res']['app'])}")
    for name, w in W.items():
        print(f"{name:40s} det(equal) {w['det_equal']:.2f}  median ICC {np.median(list(w['icc'].values())):.2f}")

    # ---- tables
    keys = ["app", "egemaps88", "demographics"]
    labels = {"app": "Our 8 features", "egemaps88": "All 88 eGeMAPS", "demographics": "Gender + age only"}
    auc_rows = "\n".join(
        f"| {labels[k]} | " + " | ".join(fmt_auc(a["res"][k]) if k in a["res"] else "–" for a in A.values()) + " |"
        for k in keys
    )
    n_row = "| Speakers (depressed) | " + " | ".join(f"{a['n']} ({a['n_dep']})" for a in A.values()) + " |"

    dir_rows = []
    for f in FEATS:
        di, si = dirs["Interview"][f]
        dr, sr = dirs["Reading"][f]
        lit = "higher" if LIT[f] > 0 else "lower"
        agree = "✅" if sr == LIT[f] else "❌ flips"
        dir_rows.append(f"| `{f}` | {lit} | {di:+.2f} | {dr:+.2f} | {agree} |")

    wi, wr1, wr2 = W.values()
    sd_rows = []
    for f in FEATS:
        ratio = wr2["within_sd"][f] / wi["within_sd"][f]
        sd_rows.append(
            f"| `{f}` | {wi['within_sd'][f]:.4g} | {wr1['within_sd'][f]:.4g} | {wr2['within_sd'][f]:.4g} | {ratio:.2f} | "
            f"{wi['icc'][f]:.2f} / {wr2['icc'][f]:.2f} | {wi['delta'][f]:+.2f} / {wr2['delta'][f]:+.2f} |"
        )
    det_rows = "\n".join(f"| {n} | {w['n_clips']} | {w['median_s']:.1f} | {w['det_equal']:.2f} | {w['norm_equal']:.3f} |" for n, w in W.items())

    dR = np.abs(wi["R"] - wr2["R"])[np.triu_indices(len(FEATS), 1)]
    big = sorted(((abs(wi["R"][i, j] - wr2["R"][i, j]), FEATS[i], FEATS[j], wi["R"][i, j], wr2["R"][i, j])
                  for i in range(len(FEATS)) for j in range(i + 1, len(FEATS))), reverse=True)[:4]
    big_rows = "\n".join(f"| `{a}` × `{b}` | {ri:+.2f} | {rr:+.2f} |" for _, a, b, ri, rr in big)

    cross_rows = ""
    if comb is not None:
        m = comb["merged"]
        cross_rows = "\n".join(
            f"| `{f}` | {m[f'app_{f}_iv'].corr(m[f'app_{f}_rd'], method='spearman'):+.2f} | "
            f"{m[f'app_{f}_rd'].median() / m[f'app_{f}_iv'].median():.2f} |"
            for f in FEATS
        )

    within_gain = np.median([wi["within_sd"][f] / wr2["within_sd"][f] for f in FEATS])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Reading task vs interview task (Androids Corpus)

Question (#21, T7): is read speech a better reference than free talk for the fixed daily phrase?
Generated by `ml/voice/reading_vs_interview.py`. Derived numbers only; no audio leaves the machine.

**Setup.** Interview: {len(iv)} clips (separate turns, different content). Reading: one recording per
speaker of the same fairy tale (~48 s), cut into {K} equal parts by duration (~10 s each, like the ~9 s
phrase). Part k covers roughly the same text for every speaker, so a two-way model (speaker + part)
removes the text effect. That is the closest the corpus gets to "the same words, read again".
All within one session and one microphone, so day-to-day noise is larger for both tasks.

## 1. Does read speech separate depressed from healthy speakers?

Speaker-level L2 logistic regression, the corpus' own folds (Read block for reading), same code as
`ml/train_weights.py`.

| Model (AUC, mean ± SD over 5 folds) | Interview (clips) | Reading (whole file) | Reading (median of 5 parts) |
|---|---|---|---|
{n_row}
{auc_rows}
""" + (f"""
Both tasks together (speakers in both, n = {comb['n']}, 16 features): **{fmt_auc(comb['both'])}**
(interview features only {fmt_auc(comb['iv_only'])}, reading features only {fmt_auc(comb['rd_only'])} on the same speakers).
""" if comb else "") + f"""
## 2. Directions: does depression move the read voice the same way?

Between-person Cohen's d (depressed − control), speaker medians. The engine's direction is the literature one.

| Feature | Worse when | d, interview | d, reading | Reading agrees with the engine's direction? |
|---|---|---|---|---|
{chr(10).join(dir_rows)}

## 3. Within-person variation: is read speech steadier?

Within-person SD in raw feature units (pooled over speakers). The ratio compares like with like only
for features whose level is similar in the two tasks (see §4, level ratio). ICC = share of the variance
that is between people (higher = a person is more consistent with themselves relative to how people differ).
δ = (depressed mean − control mean) / within-person SD: the shift of a full relapse in z units.

| Feature | Within SD, interview | Within SD, reading one-way | Within SD, reading two-way | Ratio reading two-way / interview | ICC interview / reading | δ interview / reading |
|---|---|---|---|---|---|---|
{chr(10).join(sd_rows)}

Median ratio interview SD / reading SD (two-way): **{within_gain:.2f}×**.

**Detectability** = expected shift of the equal-weight score D (noise SD 1, measured correlation) for a
full control → patient change, in-sample. The norm √(wᵀRw) is what the engine divides by.

| Within-person model | Clips | Median duration (s) | Detectability, equal weights | √(wᵀRw) |
|---|---|---|---|---|
{det_rows}

## 4. Correlation

Within-person correlation of the features (what the engine uses to normalise D): mean absolute
difference interview vs reading (two-way) over the 28 pairs = **{dR.mean():.2f}**, max {dR.max():.2f}.
Largest differences:

| Pair | r, interview | r, reading (two-way) |
|---|---|---|
{big_rows}
""" + (f"""
Across tasks, per speaker (Spearman, speakers in both): does a person's reading voice rank them the
same way as their free talk? Level ratio = median reading / median interview.

| Feature | ρ (reading vs interview) | Level ratio |
|---|---|---|
{cross_rows}
""" if comb else "") + """
## 5. What this means for the daily phrase (decision B28)

- **Read speech is steadier, not more informative.** Within a person, every feature varies less when
  reading (median 1.5× smaller SD, ICC higher for 6 of 8), even though the reading parts are shorter
  (~10 s vs ~16 s). But depression also moves the read voice less on several features, so the net
  signal-to-noise of the equal-weight score is about the same (detectability 1.5 vs 1.5 in-sample;
  held-out 1.69 vs 1.58 with folds from 0.5 to 3.5, i.e. a tie). Between people, reading separates
  worse (AUC 0.68–0.71 vs 0.77), and adding reading to interview features does not help.
- **Steadiness is what a fixed phrase is for.** A daily comparison against one's own baseline gains
  from a smaller within-person SD, so read speech is the right corpus reference for the phrase, not
  free talk. It does **not** show that the phrase beats free talk; that needs day-to-day data
  (`phrase_check.py`).
- **Not every feature carries over.** In reading, `pause_ratio` flips direction (pauses follow the
  punctuation), and `jitter`/`shimmer` keep almost no group difference (δ 0.09 and 0.03). The
  candidates for the phrase are `phrase_energy`, `phrase_arousal`, `phrase_valence`, `phrase_f0_std`
  and `phrase_speech_rate`. The other three stay at weight 0 even if they are steadier.
- **Priors.** If the phrase features get weight, their corpus prior (within-person SD for the B15
  shrinkage, the within-person correlation for B12) comes from the reading parts (two-way), never
  from the interview clips. Energy and arousal are ~2× steadier when reading, so the interview SD
  would make a phrase baseline far too wide.

**Limits.** One session per speaker, so this is within-session, not day-to-day. The parts of the
passage are different sentences; the two-way model removes the average text effect, not each
speaker's interaction with the text. Italian speakers, one microphone setup.
""")
    print(f"→ {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
