"""Train feature weights w_i on the Androids Corpus – docs/plan.md §6.

Pipeline (TODO B, lør):
  1. For each speaker in data/androids: extract the same features as backend/app/pipeline
     (prosody via parselmouth, arousal/valence via audEERING wav2vec2).
  2. Standardise → L2-regularised logistic regression (depressed vs. control).
  3. Speaker-independent CV with the corpus' own fold-list.csv → report AUC.
  4. Write coefficients to backend/app/engine/weights.json:
       {"weights": {"speech_rate": 0.12, ...}, "auc": 0.71, "n": 116}
     The backend falls back to equal weights if the file is missing.

Run:  backend/.venv/bin/python ml/train_weights.py   (after pip install -r backend/requirements-ml.txt)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "androids"
OUT = ROOT / "backend" / "app" / "engine" / "weights.json"


def main() -> None:
    if not DATA.exists():
        sys.exit(f"Missing {DATA} – run ml/download_androids.sh first")
    raise NotImplementedError("TODO: feature extraction + logistic regression")
    OUT.write_text(json.dumps({"weights": {}, "auc": None, "n": 0}, indent=2))


if __name__ == "__main__":
    main()
