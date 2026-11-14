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


def load_folds(task: str) -> dict[str, int]:
    """speaker -> fold (0-based) from fold-lists.csv.

    Real layout: row 0 = block titles ("Read" … "Interview"), row 1 = fold1..fold5 per block,
    then one speaker id per cell (e.g. '01_CF56_1'). The fold is the column's position in its block.
    """
    base = corpus_dir()
    path = next((p for p in (base / "fold-lists.csv", base / "fold-list.csv") if p.exists()), None)
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return {}
    want = {"interview": "interview", "reading": "read"}[task]
    titles, fold_names = rows[0], rows[1]
    block, cols = None, []
    for j, name in enumerate(fold_names):
        if j < len(titles) and titles[j].strip():
            block = titles[j].strip().lower()
        if block == want and name.strip().lower().startswith("fold"):
            cols.append(j)
    folds: dict[str, int] = {}
    for row in rows[2:]:
        for k, j in enumerate(cols):
            m = _NAME.search(row[j]) if j < len(row) else None
            if m and m["X"] != "X":
                folds[f"{int(m['nn']):02d}_{m['X']}"] = k
    return folds
