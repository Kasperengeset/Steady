# Datasets

All raw data lives in `data/` (gitignored, and the pre-commit hook blocks it). Never commit it, never upload it elsewhere, never send it to Gemini. Check what you have with `python ml/check_data.py`.

| Dataset | Branch | Download | Use | Redistribute | In the video |
|---|---|---|---|---|---|
| Reddit Mental Health | text | `python ml/download_data.py --reddit` | PDDL, subject to Reddit API terms | no, link to Zenodo | short quotes only |
| Depresjon | health | `python ml/download_data.py --depresjon` | research and education only | no | plots ok |
| Androids Corpus | acoustic | `bash ml/download_androids.sh` | non-commercial academic only | **no** | **no audio** |

## Reddit Mental Health Dataset → `data/reddit/`

- Low et al. 2020, JMIR. Zenodo record 3941387. `pre` window (Dec 2018 to Dec 2019), which avoids the COVID-19 topic shift.
- One CSV per subreddit: `<subreddit>_pre_features_tfidf_256.csv`. Columns: `subreddit`, `author`, `date`, `post` (the text), about 90 precomputed features on the whole post (readability, VADER, LIWC-style, lexicon totals), then 256 `tfidf_*` columns. We compute our own features on 50 to 100-word chunks and use only `author` and `post`.
- One post per author per file, but an author can appear in several subreddits, so the split is by author across all files.

| File | Posts | 50+ words | Role |
|---|---|---|---|
| depression | 21,209 | 17,545 | positive |
| fitness | 22,164 | 18,841 | everyday |
| parenting | 17,371 | 16,067 | everyday |
| personalfinance | 23,440 | 21,613 | everyday |
| meditation | 7,856 | 6,162 | everyday |
| teaching | 2,350 | 1,954 | everyday |
| relationships | 15,037 | 14,822 | hard negative (emotional, not depression) |
| divorce | 6,419 | 5,880 | hard negative |

## Depresjon → `data/depresjon/data/`

- Garcia-Ceja et al. 2018, MMSys. Zenodo record 1219550.
- 23 depressed (unipolar and bipolar) and 32 controls, Actiwatch activity counts per minute.
- Per person: `condition/condition_N.csv` or `control/control_N.csv` with `timestamp`, `date`, `activity`. `scores.csv`: `number`, `days`, `gender`, `age`, `afftype`, `melanch`, `inpatient`, `edu`, `marriage`, `work`, `madrs1`, `madrs2`.
- Published baseline (leave-one-patient-out): F1 0.73, MCC 0.44.

## Androids Corpus → `data/androids/`

- Tao, Esposito and Vinciarelli, Interspeech 2023. 3.69 GB zip, 4.5 GB unzipped (skip the zip's `__MACOSX/` folder).
- `Interview-Task/audio/HC` (52) and `PT` (64): spontaneous speech, the part we use. `Reading-Task/` is read speech.
- File names `nn_XGmm_t.wav`: `X` = P (patient), C (control) or X (unknown, dropped), `G` = gender, `mm` = age, `t` = education. Speaker id is `nn_X`.
- `fold-lists.csv` (note the s) has two header rows: `Read` over the first five columns and `Interview` over columns 8 to 12, each with `fold1` to `fold5`. Cells are quoted file stems such as `'01_CF56_1'`. The interview folds differ from the reading folds.
