"""Androids Corpus layout helpers: file discovery, labels from filenames, speaker folds.

Filename convention (corpus README): nn_XGmm_t.wav
  nn = id within group, X = P (patient) / C (control) / X (unknown), G = M/F, mm = age, t = education 1–4
A speaker is uniquely identified by "nn_X".
"""

import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Env overrides are for smoke tests on a synthetic corpus
CORPUS = Path(os.environ.get("ANDROIDS_DIR", ROOT / "data" / "androids"))
FEATURES = Path(os.environ.get("ANDROIDS_FEATURES", ROOT / "data" / "features"))

_NAME = re.compile(r"(?P<nn>\d+)_(?P<X>[PCX])(?P<G>[MFX])(?P<mm>\d{2})_(?P<t>\d)")


def corpus_dir() -> Path:
    """Unzipped corpus may sit in data/androids/ or one level below (e.g. Androids-Corpus/)."""
    for cand in [CORPUS, *CORPUS.glob("*/")]:
        if (cand / "Interview-Task").exists() or (cand / "Reading-Task").exists():
            return cand
    sys.exit(f"Androids Corpus not found under {CORPUS} – run ml/download_androids.sh")


def parse_meta(path: Path) -> dict | None:
    """Labels from the file name, falling back to parent dir names (interview clips live in per-speaker dirs)."""
    for part in [path.stem, *(p.name for p in path.parents)]:
        m = _NAME.search(part)
        if m:
            if m["X"] == "X":
                return None  # condition unknown -> unusable for supervised training
            return {
                "speaker": f"{int(m['nn']):02d}_{m['X']}",
                "label": int(m["X"] == "P"),
                "gender": m["G"],
                "age": int(m["mm"]),
                "education": int(m["t"]),
            }
    return None


def list_files(task: str) -> list[Path]:
    base = corpus_dir()
    sub = {"interview": "Interview-Task/audio_clip", "reading": "Reading-Task"}[task]
    return sorted((base / sub).rglob("*.wav"))


def load_folds() -> dict[str, int]:
    """speaker -> fold (0-based) from fold-list.csv. Parsed defensively: any cell that contains a
    speaker-style name is assigned the fold of its column (wide format) or of a 'fold' column (long)."""
    path = corpus_dir() / "fold-list.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]
    folds: dict[str, int] = {}
    fold_col = next((i for i, h in enumerate(header) if "fold" in h and len(header) <= 3), None)
    for row in rows[1:] if any(not _NAME.search(h) for h in header) else rows:
        for j, cell in enumerate(row):
            m = _NAME.search(cell)
            if not m or m["X"] == "X":
                continue
            spk = f"{int(m['nn']):02d}_{m['X']}"
            if fold_col is not None and fold_col != j:
                digits = re.findall(r"\d+", row[fold_col])
                folds[spk] = int(digits[0]) - 1 if digits else 0
            else:
                folds[spk] = j
    return folds
