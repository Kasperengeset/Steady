"""Check that the three datasets in data/ are present and load. Notes on each: ml/DATA.md.

    python ml/check_data.py

Prints a short summary per dataset and exits with code 1 if something is missing.
"""

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
problems: list[str] = []


def check_reddit() -> None:
    print("== Reddit MH (text branch)")
    files = sorted((DATA / "reddit").glob("*_pre_features_tfidf_256.csv"))
    if not files:
        problems.append("reddit: no CSVs in data/reddit/ – python ml/download_data.py --reddit")
        return
    cols = list(pd.read_csv(files[0], nrows=0).columns)
    if not {"author", "post"} <= set(cols):
        problems.append(f"reddit: expected author and post columns, got {cols[:10]}")
        return
    for f in files:
        df = pd.read_csv(f, usecols=["author", "post"])
        words = df["post"].fillna("").str.split().str.len()
        print(f"   {f.name:<45} posts {len(df):>6}  authors {df['author'].nunique():>6}  50+ words {(words >= 50).sum():>6}")


def check_depresjon() -> None:
    print("== Depresjon (health branch)")
    root = DATA / "depresjon"
    scores = next(root.rglob("scores.csv"), None)
    cond, ctrl = sorted(root.rglob("condition_*.csv")), sorted(root.rglob("control_*.csv"))
    if scores is None or not cond or not ctrl:
        problems.append("depresjon: missing – python ml/download_data.py --depresjon")
        return
    print(f"   condition files {len(cond)} (expect 23), control files {len(ctrl)} (expect 32)")
    print(f"   scores.csv columns: {list(pd.read_csv(scores).columns)}")


def check_androids() -> None:
    print("== Androids (acoustic branch)")
    root = DATA / "androids"
    folds = next(root.rglob("fold-list*.csv"), None)
    interview = next(root.rglob("Interview-Task"), None)
    if folds is None or interview is None:
        problems.append("androids: missing or not unzipped – bash ml/download_androids.sh")
        return
    for group in ("HC", "PT"):
        print(f"   Interview-Task/audio/{group}: {len(list((interview / 'audio' / group).glob('*.wav')))} wav files")


def main() -> None:
    check_reddit()
    check_depresjon()
    check_androids()
    print()
    if problems:
        print("PROBLEMS:\n" + "\n".join(f" - {p}" for p in problems))
        sys.exit(1)
    print("All three datasets load.")


if __name__ == "__main__":
    main()
