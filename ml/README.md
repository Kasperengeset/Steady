# ml/ – acoustic branch training (Androids Corpus)

The app's state engine has three branches – **acoustic** (spoken check-in), **text** (written diary) and **health** (wearable). This folder trains the acoustic branch's weights.

**The corpus must never be committed or sent to any external API.** `data/` is gitignored.

- Source: <https://github.com/androidscorpus/data> (Tao, Esposito & Vinciarelli, Interspeech 2023)
- 3.69 GB · Italian speakers (64 depressed in the interview task) · `fold-list.csv` for speaker-independent CV
- Licence: academic/non-commercial only, no redistribution

## Run

```bash
bash ml/download_androids.sh                                  # → data/androids/ (accepts the licence)
pip install -r backend/requirements-ml.txt                    # into backend/.venv
backend/.venv/bin/python ml/extract_features.py --limit 5     # smoke test
backend/.venv/bin/python ml/extract_features.py               # all interview clips (resumable)
backend/.venv/bin/python ml/train_weights.py                  # → backend/app/engine/weights/acoustic.json
                                                              #   ml/reports/acoustic/{roc,coefficients}.png, summary.md
```

**Which Python?** `extract_features.py` and `score_clip.py` import the app's pipeline, so they need the backend venv (`backend/.venv`, has opensmile/transformers/sqlmodel). If you start them with another interpreter they re-run themselves with `backend/.venv/bin/python` automatically (`ml/_venv.py`). `train_weights.py` only needs pandas, numpy, scikit-learn and matplotlib, so any venv with those works.

`--no-emotion` skips wav2vec2 (prosody only, minutes). `--task reading` does the reading task as a sanity check (reports in `ml/reports/acoustic/reading/`; use `--no-write` so it doesn't overwrite the interview weights).

## How it works

- **Same code as the app:** features come from `backend/app/pipeline/{prosody,emotion}.py` – openSMILE eGeMAPSv02 + audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` (CC-BY-NC-SA 4.0). So trained weights are valid for the app's features by construction.
- **8 interpretable features** (`speech_rate`, `f0_std` in semitones, `pause_ratio`, `energy`, `jitter`, `shimmer`, `arousal`, `valence`), median per speaker.
- **Model:** standardise → L2 logistic regression, C picked by inner CV, outer CV over the corpus folds. Weights = |β| normalised, direction = sign(β).
- **References:** all 88 eGeMAPS, a wav2vec2-embedding probe (PCA 20 → logreg), and gender+age only (leakage check).
- AUC > 0.95 triggers a leakage warning.

**Limitation to say out loud:** a between-person classifier is used as a *prior* for within-person change; the personal baseline is the core.

Smoke-test without the real corpus: point `ANDROIDS_DIR` / `ANDROIDS_FEATURES` at a synthetic corpus with the same file-name convention.
