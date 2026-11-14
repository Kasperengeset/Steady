"""Download the text and health datasets into data/ (gitignored). Androids: ml/download_androids.sh.

    python ml/download_data.py              # Reddit + Depresjon
    python ml/download_data.py --reddit     # text branch: Reddit Mental Health Dataset, ~400 MB
    python ml/download_data.py --depresjon  # health branch: Depresjon actigraphy, ~5 MB

Licences: Reddit MH is PDDL, re-use subject to Reddit API terms (short quotes only, never commit).
Depresjon is research and education only. Never commit either.
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA = Path(__file__).resolve().parents[1] / "data"

REDDIT_RECORD = "https://zenodo.org/records/3941387/files/{name}?download=1"
REDDIT_WINDOW = "pre"  # Dec 2018 to Dec 2019, avoids the COVID-19 topic shift
REDDIT_SUBREDDITS = [
    "depression",                                                      # positive
    "fitness", "parenting", "personalfinance", "meditation", "teaching",  # everyday
    "relationships", "divorce",                                        # hard negatives
]
DEPRESJON_URL = "https://zenodo.org/records/1219550/files/depresjon-dataset.zip?download=1"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.name} (exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def get_reddit() -> None:
    for sub in REDDIT_SUBREDDITS:
        name = f"{sub}_{REDDIT_WINDOW}_features_tfidf_256.csv"
        download(REDDIT_RECORD.format(name=name), DATA / "reddit" / name)


def get_depresjon() -> None:
    out = DATA / "depresjon"
    zpath = out / "depresjon-dataset.zip"
    download(DEPRESJON_URL, zpath)
    if not (out / "data" / "scores.csv").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    print(f"Depresjon in {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reddit", action="store_true")
    p.add_argument("--depresjon", action="store_true")
    a = p.parse_args()
    both = not (a.reddit or a.depresjon)
    if a.reddit or both:
        get_reddit()
    if a.depresjon or both:
        get_depresjon()


if __name__ == "__main__":
    main()
