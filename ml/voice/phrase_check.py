"""Phrase check (#21): is the fixed daily phrase steadier than 30 s of free talk, for our own voices?

    backend/.venv/bin/python ml/voice/phrase_check.py recorded_voice/phrase_check
    backend/.venv/bin/python ml/voice/phrase_check.py recorded_voice/phrase_check --no-extract   # reuse features.csv

ONLY our own recordings. Never Androids audio (the script refuses paths inside the corpus).

How to record (per person, 5 sessions, at different times of day, e.g. 09, 12, 15, 18, 21):
  1. Read the phrase once, at a normal pace:
       "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
        The rainbow is a division of white light into many beautiful colors."
  2. Then talk freely about your day for ~30 s.
  Same phone, same distance (~20 cm), a quiet room. Do not act a mood.

File names: <person>_<YYYYMMDD>-<HHMM>_<kind>.<ext>
  person  lowercase letters, e.g. kasper, mahdan
  date    the session's date and local start time, e.g. 20261114-0930
  kind    "phrase" or "free"
  ext     wav, flac, m4a, mp3, aac, caf (non-wav is converted with macOS afconvert)
  Example: kasper_20261114-0930_phrase.m4a and kasper_20261114-0930_free.m4a (one session = one pair)

Keep the folder outside git: recorded_voice/ is gitignored. The extracted features are cached in
<folder>/features.csv (also stays local). The report contains only summary statistics (SDs, ratios),
never per-session values.

What it computes, per feature (the 8 app features, same extraction code as the app, B9):
  - within-person SD over sessions, phrase vs free talk (sample SD, pooled over people)
  - the variance ratio phrase/free with a one-sided F-test and a 90 % CI for the SD ratio
  - per-person SD ratios (must point the same way for everyone)
  - the within-person correlation phrase vs free over sessions (do they move together?)
  - the time-of-day slope of each feature (per hour, in within-person SDs), phrase and free (#27)

Gate (decided BEFORE recording, STATUS.md B29): a phrase feature may get weight only if
  1. pooled SD ratio phrase/free < 1 with the upper bound of its 90 % CI < 1 (one-sided p < 0.05),
  2. the ratio is < 1 for every person, and
  3. there are ≥ 2 people with ≥ 4 complete sessions each, and
  4. the feature is a phrase candidate (B28: depression moves it the engine's way in read speech).
Otherwise it stays at weight 0. With 2 people × 5 sessions (8 + 8 df) that needs an SD ratio below
about 0.54: only a clearly steadier phrase passes. That is deliberate (B13: never tune to 2 people).
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEATS = ["speech_rate", "f0_std", "pause_ratio", "energy", "jitter", "shimmer", "arousal", "valence"]
# B28 (reading_vs_interview.md): in read speech pause_ratio flips direction and jitter/shimmer keep almost no
# group difference, so they stay at weight 0 however steady they are.
CANDIDATES = {"speech_rate", "f0_std", "energy", "arousal", "valence"}
NAME = re.compile(r"^(?P<person>[a-z]+)_(?P<date>\d{8})-(?P<hhmm>\d{4})_(?P<kind>phrase|free)$")
AUDIO_EXT = {".wav", ".flac", ".m4a", ".mp3", ".aac", ".caf", ".ogg"}
REPORT = HERE.parents[1] / "ml" / "reports" / "voice" / "phrase_check.md"
MIN_PEOPLE, MIN_SESSIONS = 2, 4
PHRASE_S = (4.0, 20.0)  # a read of the two sentences takes ~9 s
FREE_MIN_S = 15.0


def parse_name(path: Path) -> dict | None:
    m = NAME.match(path.stem)
    if not m:
        return None
    hh, mm = int(m["hhmm"][:2]), int(m["hhmm"][2:])
    return {"person": m["person"], "session": f"{m['date']}-{m['hhmm']}", "kind": m["kind"], "hour": hh + mm / 60}


def _refuse_corpus(folder: Path) -> None:
    p = str(folder.resolve()).lower()
    if "androids" in p or "interview-task" in p or "reading-task" in p:
        sys.exit("phrase_check takes only our own recordings, never the Androids Corpus (licence, B2).")


def extract(folder: Path) -> pd.DataFrame:
    """One row per file: person, session, kind, hour, duration_s, the 8 app features."""
    from _venv import ensure_backend_venv

    ensure_backend_venv()
    sys.path.insert(0, str(HERE.parents[1] / "backend"))
    from app.pipeline import audio, emotion, prosody
    from score_clip import to_wav

    rows, bad = [], []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in AUDIO_EXT:
            continue
        meta = parse_name(path)
        if meta is None:
            bad.append(path.name)
            continue
        x = audio.load(str(to_wav(path)))
        feats = prosody.from_full(prosody.extract_full_signal(x))
        feats.update({k: v for k, v in emotion.run_signal(x)[1].items() if k in FEATS})
        rows.append({**meta, "duration_s": len(x) / audio.SR, **{f: feats[f] for f in FEATS}})
        print(f"  {path.name}: {rows[-1]['duration_s']:.1f} s")
    if bad:
        print("⚠️  skipped (name does not follow <person>_<YYYYMMDD>-<HHMM>_<phrase|free>):", ", ".join(bad))
    return pd.DataFrame(rows)


def quality(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep complete sessions (both a phrase and a free-talk file) with plausible durations."""
    notes = []
    ok = ((df["kind"] == "phrase") & df["duration_s"].between(*PHRASE_S)) | (
        (df["kind"] == "free") & (df["duration_s"] >= FREE_MIN_S))
    for _, r in df[~ok].iterrows():
        notes.append(f"{r['person']} {r['session']} {r['kind']}: {r['duration_s']:.1f} s is outside the expected length → dropped")
    df = df[ok]
    both = df.groupby(["person", "session"])["kind"].transform("nunique") == 2
    for (p, s), _ in df[~both].groupby(["person", "session"]):
        notes.append(f"{p} {s}: only one of phrase/free → session dropped")
    return df[both].copy(), notes


def analyse(df: pd.DataFrame) -> dict:
    """Pure statistics on a feature table (no audio). See the module docstring for the gate."""
    from scipy import stats

    df, notes = quality(df)
    people = df.groupby("person")["session"].nunique()
    enough = int((people >= MIN_SESSIONS).sum()) >= MIN_PEOPLE
    out = {"people": people.to_dict(), "notes": notes, "enough_data": enough, "features": {}}
    for f in FEATS:
        wide = df.pivot_table(index=["person", "session"], columns="kind", values=f).dropna()
        per_person, ss, dfree = {}, {"phrase": 0.0, "free": 0.0}, 0
        cov_pf, slopes = 0.0, {"phrase": [], "free": []}
        hours = df.drop_duplicates(["person", "session"]).set_index(["person", "session"])["hour"]
        for p, g in wide.groupby(level="person"):
            if len(g) < 2:
                continue
            sd = {k: float(g[k].std(ddof=1)) for k in ("phrase", "free")}
            per_person[p] = {"sd_phrase": sd["phrase"], "sd_free": sd["free"],
                             "ratio": sd["phrase"] / sd["free"] if sd["free"] > 0 else float("inf")}
            c = g - g.mean()
            for k in ("phrase", "free"):
                ss[k] += float((c[k] ** 2).sum())
                h = hours.loc[g.index].to_numpy(float)
                if np.ptp(h) > 0 and sd[k] > 0:  # slope per hour in within-person SDs
                    slopes[k].append(float(np.polyfit(h, g[k].to_numpy(float), 1)[0] / sd[k]))
            cov_pf += float((c["phrase"] * c["free"]).sum())
            dfree += len(g) - 1
        if dfree == 0:
            continue
        var_p, var_f = ss["phrase"] / dfree, ss["free"] / dfree
        F = var_p / var_f if var_f > 0 else float("inf")
        lo, hi = (np.sqrt(F / stats.f.ppf(q, dfree, dfree)) for q in (0.95, 0.05))
        p_one = float(stats.f.cdf(F, dfree, dfree))  # H1: phrase variance < free variance
        r = cov_pf / np.sqrt(ss["phrase"] * ss["free"]) if ss["phrase"] * ss["free"] > 0 else float("nan")
        every = bool(per_person) and all(v["ratio"] < 1 for v in per_person.values())
        out["features"][f] = {
            "sd_phrase": float(np.sqrt(var_p)), "sd_free": float(np.sqrt(var_f)), "sd_ratio": float(np.sqrt(F)),
            "ci90": (float(lo), float(hi)), "p_steadier": p_one, "df": dfree, "per_person": per_person,
            "r_phrase_free": float(r), "every_person_steadier": every,
            "slope_phrase": float(np.mean(slopes["phrase"])) if slopes["phrase"] else float("nan"),
            "slope_free": float(np.mean(slopes["free"])) if slopes["free"] else float("nan"),
            "candidate": f in CANDIDATES,
            "passes": bool(enough and hi < 1 and every and f in CANDIDATES),
        }
    return out


def write_report(res: dict, path: Path = REPORT) -> str:
    rows = []
    for f, r in res["features"].items():
        pp = " · ".join(f"{p} {v['ratio']:.2f}" for p, v in r["per_person"].items())
        rows.append(
            f"| `{f}` | {r['sd_ratio']:.2f} | {r['ci90'][0]:.2f}–{r['ci90'][1]:.2f} | {r['p_steadier']:.3f} | {pp} | "
            f"{r['r_phrase_free']:+.2f} | {r['slope_phrase']:+.2f} / {r['slope_free']:+.2f} | {'✅ weight allowed' if r['passes'] else 'weight 0' + ('' if r['candidate'] else ' (not a candidate, B28)')} |"
        )
    passed = [f for f, r in res["features"].items() if r["passes"]]
    people = ", ".join(f"{p} ({n} sessions)" for p, n in res["people"].items())
    notes = "\n".join(f"- {n}" for n in res["notes"]) or "- none"
    verdict = (
        "**Not enough data for a decision** (needs ≥ 2 people with ≥ 4 complete sessions). All phrase features stay at weight 0."
        if not res["enough_data"] else
        f"**Passed the gate:** {', '.join(f'`{f}`' for f in passed)}." if passed else
        "**No phrase feature passed the gate.** All phrase features stay at weight 0."
    )
    text = f"""# Phrase check: fixed phrase vs free talk (#21)

Generated by `ml/voice/phrase_check.py` on our own recordings ({people}). Only summary statistics are
in this file; audio and per-session values stay on the recording machine.

{verdict}

Gate (B29, set before recording): pooled SD ratio phrase/free < 1 with the 90 % CI below 1
(one-sided F-test p < 0.05), the ratio < 1 for every person, ≥ 2 people with ≥ 4 sessions.

| Feature | SD ratio phrase/free (pooled) | 90 % CI | p (phrase steadier) | Per person | r phrase vs free | Hour slope phrase / free (SD per h) | Decision |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- SD ratio < 1 = the phrase is steadier from session to session than free talk.
- r = within-person correlation over sessions. High r means the phrase repeats what free talk already says.
- Hour slope: change per hour of the day in within-person SDs (input to the time-of-day decision, #27).
  With 5 sessions it is a rough check only.

Quality notes:
{notes}

**Limits.** 2 people, 5 sessions each: only a large difference can pass. Healthy voices only, so this
tests noise, not whether depression still shows in a read phrase (for that, see
`reading_vs_interview.md`). One phone and room per person.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--no-extract", action="store_true", help="reuse <folder>/features.csv")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    folder = args.folder.expanduser()
    _refuse_corpus(folder)
    cache = folder / "features.csv"
    if args.no_extract:
        df = pd.read_csv(cache)
    else:
        df = extract(folder)
        if df.empty:
            sys.exit(f"no recordings matching <person>_<YYYYMMDD>-<HHMM>_<phrase|free>.<ext> in {folder}")
        df.to_csv(cache, index=False)
        print(f"features → {cache} (local only)")
    res = analyse(df)
    write_report(res, args.report)
    for f, r in res["features"].items():
        print(f"  {f:12s} SD ratio {r['sd_ratio']:.2f}  90% CI {r['ci90'][0]:.2f}–{r['ci90'][1]:.2f}  "
              f"{'PASS' if r['passes'] else 'weight 0'}")
    print(f"→ {args.report}")


if __name__ == "__main__":
    main()
