# ml/text/ – text branch (written diary or transcript)

Input: diary text, ideally 50 to 100 words (about 30 s of speech). English only.
Output, from `backend/app/pipeline/text.py`:

| Feature | What | Direction |
|---|---|---|
| `text_logit` | log-odds that the text reads like r/depression (TF-IDF + logistic regression) | higher = worse |
| `absolutist` | absolutist words per 100 ("always", "nothing", "never"), skipping a word after "not" | higher = worse |
| `first_person` | i, me, my, mine, myself per 100 | higher = worse |
| `sentiment` | VADER compound, -1 to +1 | lower = worse |

Plus a deterministic **crisis phrase check** (English and Norwegian) that sets the check-in's crisis flag. It never waits for a model.

**Same code in training and in the app** (STATUS.md B9): the scripts here import `backend/app/pipeline/text.py`. The model is exported as JSON (`backend/app/pipeline/models/text_tfidf_lr.json.gz`) and scored in pure Python, so the app needs no scikit-learn and no pickle. `train_v1.py` checks that the export matches scikit-learn.

## Run

```bash
python ml/download_data.py --reddit                         # → data/reddit/ (~400 MB, gitignored)
backend/.venv/bin/python ml/text/train_v1.py                # ~1 min → model JSON + ml/reports/text/v1.md
backend/.venv/bin/python ml/text/train_lexicon.py           # ~1 min → engine/weights/text.json + lexicon.md
backend/.venv/bin/python ml/text/eval_diary.py              # diary sanity set → ml/reports/text/diary.png
backend/.venv/bin/python ml/text/predict.py "some text"     # try it; no argument = interactive
```

On Windows use `backend\.venv\Scripts\python`.

## Results (details in `ml/reports/text/`)

Reddit Mental Health Dataset (Low et al. 2020), `pre` window. r/depression against fitness, parenting, personalfinance, meditation, teaching (everyday) and relationships, divorce (hard negatives). 115,814 posts, 112,861 authors. Split **by author** 70/15/15 (asserted), chunks of 50 to 100 words, negatives downsampled 1:1 per split, C tuned on validation, test used once.

| Model | F1 | ROC-AUC | Baseline |
|---|---|---|---|
| v1, masked (the app uses this) | 0.908 | 0.970 | 0.667 / 0.500 |
| v1, unmasked | 0.924 | 0.977 | |
| lexicon only (3 features) | 0.715 | 0.790 | |
| v1 + lexicon | 0.909 | 0.970 | no gain over v1 |

**Masking.** Forum names (`depress*`, `subreddit`, the negative forums' names) and treatment words (therapy, meds, doctor, symptoms, TMS, ECT, antidepressant names) are replaced by a break that no bigram crosses. TMS patients mention treatment on good days too.

**Diary sanity set** (`diary_sanity.csv`, 10 team-written entries, each low day mirrors a good day): 9 of 10 correct at 0.5, a low day scored above its good day in 23 of 25 pairs. Good days 0.20 on average, low days 0.61.

**Engine weights** (`weights/text.json`): |standardised beta| of v1 + lexicon, as for the acoustic branch. `text_logit` carries almost all weight. The lexicon features stay in the output to explain the score to a clinician.

## Limitations – say these out loud

- **Genre, not only mood.** r/depression differs from everyday forums in genre as well as mood. On diary text the model scores personal, feelings-focused writing high even on a good day (a good day mentioning "feel more like myself again" scored 0.95). The useful signal is change against the person's own baseline, which is how the engine uses it.
- **Masked low mood.** Sentiment rates "I'm fine, it's okay" as positive. The ranking of low against good days held up better than any fixed cut-off.
- **Reddit is not a diary**, and the diary sanity set is 10 entries from one writer.
- **English only.** A Norwegian diary needs a Norwegian model or translation first.
- Not validated on relapse after TMS.
