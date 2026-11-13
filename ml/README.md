# ml/ – voice-feature weights from the Androids Corpus

**The corpus must never be committed.** `data/` is gitignored.

- Source: <https://github.com/androidscorpus/data> (Tao, Esposito & Vinciarelli, Interspeech 2023)
- 3.69 GB · 118 Italian speakers (64 depressed) · Reading task + interview task · `fold-list.csv` for speaker-independent 5-fold CV
- Licence: academic/non-commercial only, no redistribution

```bash
bash ml/download_androids.sh          # → data/androids/
pip install -r backend/requirements-ml.txt
python ml/train_weights.py            # → backend/app/engine/weights.json (+ AUC)
```

Limitation to say out loud: a between-person classifier is used as a *prior* for within-person change; the personal baseline is the core.
