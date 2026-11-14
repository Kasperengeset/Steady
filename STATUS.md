# STATUS – Beyond the Clinic

> Living document: what is built, what we have learned about the model, which decisions we made and why, and what is left.
> **Update this whenever something changes.** Original plan: [`docs/plan.md`](docs/plan.md). Last update: Sat 14 Nov 2026 (branch `acoustic-model`).

## Core message

*We do not diagnose. We start when the patient is well after TMS and detect when they drift back – measured against their own healthy normal, not a population average.*

---

## 1. Architecture – three branches, one engine

| Branch | When | Dataset for weights | Status |
|---|---|---|---|
| **Acoustic** | the user *speaks* the check-in | Androids Corpus | ✅ trained, in use |
| **Text** | the user *writes* a diary entry, or the transcript of a spoken check-in | Reddit Mental Health Dataset | ✅ trained, in use (`pipeline/text.py`, see §2b) |
| **Health** | daily, automatic | Fitbit / Google Health (candidate: Depresjon dataset) | ⏳ not started |

Per check-in:
1. Extract features per branch.
2. **z-scores against the patient's own baselines** – two tracks (see §3 «Healthy baseline»): `z = direction × (today − centre) / scale`.
3. Branch score per check-in: `D_branch = Σ w·z / √(wᵀ R w)`, where R is the within-person correlation measured on the corpus → D has SD 1 when nothing changes (B10, B12).
4. **All check-ins of a day count** (B16): per branch the day's check-ins are combined as `Σ D / √n` (still SD 1, signal grows with √n); daily score `D_t` = mean over branches with data today. A later check-in no longer overwrites an earlier one.
5. Smoothing: `S_t = 0.3·D_t + 0.7·S_(t−1)`, per track; **S = max(S_anchored, S_rolling)**.
6. Escalation: friends at S > 1 for 3 days · clinician at S > 2 · immediately on the crisis flag.
7. Interventions are chosen with Thompson sampling (bandit) and learn from evening feedback.

Weights per branch live in `backend/app/engine/weights/<branch>.json`. If the file is missing, literature directions and equal weights are used.

---

## 2. Acoustic model

### ⭐ Current model (as used by the app)

**Equal weights (0.125 each), literature directions, D normalised with the within-person correlation** (`acoustic.json`, decision B12).

| Feature | Source | Weight | "Worse" when … | Typical shift with depression (within-person SD) | Share of D in a full relapse |
|---|---|---|---|---|---|
| `arousal` | wav2vec2 | 0.125 | lower | −1.42 | **27 %** |
| `energy` | openSMILE | 0.125 | lower | −1.19 | **23 %** |
| `valence` | wav2vec2 | 0.125 | lower | −0.88 | **17 %** |
| `jitter` | openSMILE | 0.125 | higher | +0.54 | 10 % |
| `pause_ratio` | openSMILE | 0.125 | higher | +0.44 | 8 % |
| `speech_rate` | openSMILE | 0.125 | lower | −0.30 | 6 % |
| `f0_std` | openSMILE | 0.125 | lower | −0.24 | 5 % |
| `shimmer` | openSMILE | 0.125 | higher | +0.28 | 5 % |

- **Score:** `D = Σ 0.125 · z / 0.450`. 0.450 = √(wᵀRw), the noise of the weighted sum given the measured correlation (it would be 0.354 if the features were independent). A single feature moving 1 SD on its own moves D by 0.28.
- **What drives the score:** with equal weights, a feature's contribution is decided by how much depression shifts it. **Arousal, energy and valence carry ~2/3 of the signal** without being up-weighted. A full control→patient shift gives **D ≈ 1.5**.
- **Why equal weights:** on held-out folds four weightings tie (§3 «Change weights»), so the data cannot justify anything finer. The data does support *which* features and *which direction*.
- **Reference weightings kept in `acoustic.json` (not used):**
  - classifier: arousal 0.18 · jitter 0.18 · valence 0.17 · energy 0.14 · f0_std 0.13 · shimmer 0.10 · speech_rate 0.07 · pause_ratio 0.04
  - Fisher (sign-constrained): arousal −1.03 · jitter +0.55 · energy −0.54 · f0_std −0.41 · shimmer +0.39 · valence −0.22 · speech_rate −0.14 · pause_ratio 0
- ⚠️ `energy` is ~¼ of the signal **and** the feature most sensitive to microphone and distance → recording protocol/quality check (T14) matters before real use.

### Features (8)
| Feature | Source | Method | What it measures |
|---|---|---|---|
| `speech_rate` | openSMILE eGeMAPS | FFT-based | voiced segments/s (≈ syllables/s) |
| `f0_std` | openSMILE | FFT (pitch in semitones) | pitch variation (monotony) |
| `pause_ratio` | openSMILE | voiced/unvoiced | share of unvoiced time (pauses + unvoiced consonants) |
| `energy` | openSMILE | FFT, auditory bands | loudness |
| `jitter` | openSMILE | time domain, period by period | pitch irregularity |
| `shimmer` | openSMILE | time domain | amplitude irregularity |
| `arousal` | wav2vec2 (audEERING) | neural net on the raw waveform, **no FFT** | how energetic the voice sounds |
| `valence` | wav2vec2 | same | how positive the voice sounds |

openSMILE and wav2vec2 are **off-the-shelf tools that are not trained**. The only thing we fit is on top of the 8 numbers.

### Training and validation
- **Data:** Androids Corpus, interview clips. 866 labelled clips → median per speaker → **114 speakers (62 depressed / 52 controls)**. 2 speakers with unknown condition dropped.
- **Classifier (validation of the features):** standardisation → L2 logistic regression, C chosen by inner CV on **log-loss**. The corpus' own `fold-lists.csv`, 5 folds, no speaker in both train and test.
- **Change-detection weights:** `ml/change_weights.py` – within-speaker covariance from ~5 clips per speaker; compares weightings on held-out folds (B12).

### Results (AUC, mean ± SD over 5 folds)
| Model | AUC | Used in the app? |
|---|---|---|
| **Our 8 features** | **0.77 ± 0.08** | ✅ (features + directions) |
| All 88 eGeMAPS | 0.88 ± 0.06 | reference |
| wav2vec2 embedding (1024 → PCA 20) | 0.91 ± 0.06 | reference |
| Gender + age only | 0.48 | confound check ✅ |

First model version (C chosen by AUC): openSMILE features only 0.76, wav2vec2 features only 0.73, both 0.76 (they overlap a lot).

### Known weaknesses and checks
- **Between- vs within-person:** the classifier learns differences *between* people; the app tracks change *within* one person. The corpus gives a prior; the personal baseline is the core.
- **Recording conditions:** depressed speakers have longer clips (median 18.6 s vs 10.5 s) and ~2 dB quieter audio.
  - Duration alone gives AUC 0.64 but barely helps on top of the features (0.76 → 0.77), so the features are not just a duration proxy.
  - Loudness alone gives 0.68. Without `energy`, AUC only drops to 0.74.
- **Microphone:** `energy` depends on mic and distance. Phone recordings sit around the corpus' 98th percentile.
- **The wav2vec2 reference (0.91)** may partly capture mic and room. Use **0.77** as the honest external number.
- **Language:** the corpus is Italian. Prosody transfers reasonably across languages; content is handled by the text branch.
- **Licence:** both Androids and the audEERING model are non-commercial only.

---

## 2b. Text model (Mahdan)

Details and how to run: [`ml/text/README.md`](ml/text/README.md). Reports: `ml/reports/text/`.

### Features (4) + crisis flag
| Feature | What | Direction |
|---|---|---|
| `text_logit` | log-odds that the text reads like r/depression (TF-IDF + logistic regression) | higher = worse |
| `absolutist` | absolutist words per 100 words ("always", "nothing", "never") | higher = worse |
| `first_person` | I/me/my per 100 words | higher = worse |
| `sentiment` | VADER, −1 to +1 | lower = worse |

The crisis flag is a fixed phrase list (English and Norwegian). It never waits for a model and goes straight to the clinician.

### Training
- **Data:** Reddit Mental Health Dataset, `pre` window. r/depression against everyday forums (fitness, parenting, personalfinance, meditation, teaching) and hard negatives (relationships, divorce). 115,814 posts, 112,861 authors.
- **Split by author** 70/15/15 (asserted in the code). Posts are cut into 50–100-word chunks, like a diary entry. 1:1 balance per split. C chosen on validation, test used once.
- **Masking:** forum names and treatment words (therapy, meds, doctor, TMS, ECT, drug names) are removed, and no bigram crosses a removed word. Patients after TMS mention treatment on good days too.
- **Same code in training and app** (B9). The model is exported as JSON and scored in pure Python: no scikit-learn or pickle in the app.

### Results
| Model | F1 | ROC-AUC |
|---|---|---|
| **v1, masked (used in the app)** | **0.908** | **0.970** |
| the 3 lexicon features only | 0.715 | 0.790 |
| v1 + lexicon | 0.909 | 0.970 (no gain) |
| baseline | 0.667 | 0.500 |

**Diary check** (10 team-written entries, each low day mirrors a good day): 9 of 10 correct, and the low day scored above its good day in 23 of 25 pairs.

### Known weaknesses
- **Genre, not only mood.** The model scores personal writing about feelings high, even on good days. The useful signal is change against the person's own baseline, not the absolute level.
- **Masked low mood** ("I'm fine") fools the sentiment feature.
- **English only.** The demo is in English.
- **Weights differ from B12.** `weights/text.json` puts 0.95 on `text_logit` (|standardised beta|, the original acoustic convention), not equal weights. Reason: the lexicon features add nothing on Reddit, `first_person` overlaps with `text_logit`, and there is no within-person text data to measure correlation. They stay in the output to explain the score to the clinician.

---

## 3. Key concepts (things we have discussed)

### What the model is – and is not
**A single clip on its own gives almost nothing usable.**
- **In the app** one clip only yields 8 feature values. A score (`S`) appears once a healthy baseline exists.
- **`ml/score_clip.py`** gives `P(depressed-like voice)` against the corpus. AUC 0.77 means a random depressed/healthy pair is ranked correctly 77 % of the time, but for one person the error margin is large. Italian speech, mic-dependent, **not a diagnosis**.

Training gives two things, neither of them a prediction:
1. **Validation of the instrument:** the features carry depression signal on a clinical corpus (AUC 0.77, all directions match the literature).
2. **Structure for change detection:** directions, within-person noise and correlation.

The useful output is the **time series S_t over days**, not a single clip.

**Not shown:** that the system detects relapse in real patients. That needs data following the same people through a relapse; Androids has one session per person.
**Pitch:** say "the features are validated on a clinical corpus", **not** "the model detects depression with 77 % accuracy".

### We measure change, not diagnosis
The app compares the patient with themselves. **If someone is already depressed when we start, that becomes the baseline and we see no deviation.** It still fits this challenge:
- Patients start after **successful TMS**, i.e. in remission, and the clinic knows this from their depression scores during treatment.
- The baseline is therefore the "healthy normal", and we look for **relapse**.

### ⭐ Healthy baseline – how it is built (T1)
**Lifecycle:** `building` → (period over and enough check-ins) → `active` → (a newer one locks) → `retired`. The clinician can reject: `rejected`, and a new one starts.
- **Start:** the first check-in without a baseline starts one automatically. The clinician can start a new one with `POST /api/patients/{id}/baseline/start` (e.g. after a new phone or confirmed remission).
- **Period:** 14 days and at least 10 check-ins (`config.py`: `healthy_period_days`, `healthy_min_checkins`). Overridable per baseline; **demo: `{"period_days": 0, "min_checkins": 3}`**. If the period ends with too few check-ins, it is extended.
- **Minimum values per feature (B17, B18):** a feature only gets baseline stats with ≥ `min(min_checkins, min_feature_n = 5)` values. Text features have no corpus SD to shrink toward, so 1–2 written entries would give a guessed scale. A branch without stats gets no anchored score (rolling track and crisis flag still work). **A baseline only locks if at least one feature qualifies** – otherwise the period is extended (e.g. 3 spoken + 3 written check-ins). Mixed use needs ≥ 5 of each kind for both branches to be scored.
- **No score during the period.** The patient sees "Building your normal: day x of 14". The crisis flag works throughout.
- **Re-baselining:** while a new baseline builds, the response has `baseline` (the active one, still scoring) and `baseline_building` (progress of the new one); the patient sees "Updating your normal". An active baseline's check-in count is frozen at lock.
- **Stored per feature and per slot** (`morning` / `evening` / `all`):
  - centre = median
  - scale = MAD-based SD shrunk toward the corpus within-person SD: `σ² = (n·s² + k·s₀²)/(n + k)`, k = 5
  - slots with too few check-ins fall back to `all`
- **Two tracks per check-in:**
  - `S_anchored` vs the healthy baseline catches slow drift
  - `S_rolling` vs the last 14 days, same slot, ≥ 7 check-ins, catches abrupt change
  - **S = the higher of the two**, and the clinician sees which (`driver`)
- **Never re-anchored automatically.** A new baseline builds while the old one keeps scoring; the old one is retired only when the new one locks. Rejected baseline → no score until a new one is locked.
- **Code:**
  - `app/baseline.py` (lifecycle)
  - `engine/state.py` (`robust_stats`, `history_stats`, `zscores`, `branch_deviation`)
  - `routers/checkins.py` (two tracks)
  - `routers/baselines.py` (API: `GET …/baseline`, `POST …/baseline/start|confirm|reject`)
  - tests in `backend/tests/test_baseline.py` (`backend/.venv/bin/python -m pytest backend/tests`); one shared clock `baseline.clock()` that tests replace to advance days
- ⚠️ **After data-model changes, delete `data/app.db`** (`create_all` does not add new columns). It only holds test data.

### Simulator (`backend/app/sim/simulator.py`)
Runs the **real engine** (same functions as the app, trained `acoustic.json`) on a synthetic patient for 70 days, with 1 or 2 spoken check-ins per day (`checkins_per_day`). Fully reproducible per seed (the bandit uses the seeded RNG too).
- Latent depression `d_t` in [0, 1] follows an AR(1) process. A slow relapse starts on day 40 and builds over ~15 days.
- Features = personal offset + `d_t` × (patient mean − control mean from the corpus) + noise. Noise uses the **measured** within-person SD and correlation from Androids, × `day_noise_factor`.
- The bandit picks interventions; each has a hidden "true" helpfulness.
- API: `POST /api/sim/run` gives one patient. `GET /api/sim/robustness?seeds=50` gives numbers over many patients – **use those externally, not one lucky seed.**

**Results over 100 synthetic patients** (14 Nov, reproducible; "combined" = what the app escalates on):

| Day-to-day noise | Check-ins/day | Missed, rolling | Missed, anchored | **Missed, combined (app)** | False alarms, combined (mean days before relapse) |
|---|---|---|---|---|---|
| ×1.0 (measured within one session) | 1 | 78 | 24 | **24 / 100** | 0.33 |
| ×1.0 | **2** | 62 | 7 | **6 / 100** | 0.55 |
| ×1.5 | 1 | 75 | 46 | **40 / 100** | 0.56 |
| ×1.5 | **2** | 73 | 20 | **18 / 100** | 0.75 |
| ×2.0 | 1 | 72 | 56 | **46 / 100** | 0.67 |
| ×2.0 | **2** | 81 | 39 | **32 / 100** | 0.78 |

- **Morning + evening (T16) roughly halves the missed relapses**, at the cost of slightly more false alarms.
- Earlier 50-patient numbers (14/50, 25/50, 28/50) varied by a few relapses between runs because the bandit used an unseeded RNG; these are reproducible.

⚠️ **Earlier numbers (0/50 missed) were too optimistic:** they used a guessed noise level (0.6 × between-person SD) and assumed independent features.

Interpretation:
- **The fixed baseline is essential.** The rolling one "absorbs" a slow relapse (boiling frog): ~41/50 missed on its own.
- **One spoken check-in a day catches only about half of relapses** at realistic noise (×1.5–2). A full relapse moves D only ~1.5 SD and the alarm needs S > 1 for 3 days. What helps:
  - **two check-ins a day** (done, B16): 40 → 18 missed of 100 at ×1.5
  - calibrated thresholds – T12
  - the text and health branches
- **We do not detect relapse earlier than self-report** (median ~0 days). Do not claim early warning in the pitch.
- **The bandit learns moderately in 70 days**: it ends on the truly best intervention in 45/100 patients (chance ≈ 17/100). Reward is capped at 0.5 because "did the state improve?" is not wired in (T8).
- **Assumptions** (returned by the API): measured within-session noise × `day_noise_factor`, measured correlation, self-report lagging 4 days.

### What gives an absolute level rather than just change?
- The text branch and the crisis flag (e.g. "I feel hopeless")
- The corpus model (`ml/score_clip.py`) – too weak to stand alone (AUC 0.77, mic-dependent)
- An occasional questionnaire (e.g. PHQ-9 every other week) as a clinical anchor

### How many clips are needed?
The model is **already trained** (114 people). A user's own clips are for testing and for the baseline, not for training.
- **Testing the model:** 2–5 clips in different states (engaged vs tired and flat), same mic and distance, 30–60 s of free speech. Check the features move in the expected direction (`ml/score_clip.py`).
- **Personal baseline:** an SD from 3 measurements is very uncertain. **~7 is usable, ~14 (two weeks) is stable** – which matches the healthy period after TMS. The app requires ≥ 10 for the healthy baseline and ≥ 7 for the rolling track, and shrinks the SD toward the corpus value.
- **Morning and evening** have separate baselines because the voice differs systematically through the day.
- **A better model = more *people*, not more clips.** Many unlabelled clips from one person do not help the classifier. Personal weights would need months of data paired with depression scores.
- **Demo:** two weeks do not exist. Use a short demo baseline and the simulator.

### Speech vs text
Speech goes to the acoustic branch, a written diary to the text branch. Gemini is used in the text/content part.

### Change weights ≠ classification weights (measured 14 Nov)
The classifier learns to separate *people*. The app needs to detect *change within a person*, so a feature should count by how far depression moves it relative to that person's normal variation: **d_within = (patient mean − control mean) / within-person SD**. Androids has ~5 clips per person, so this can be measured.

First estimate (mean of per-speaker SDs):

| Feature | d_within | ICC (stability within a person) | Classifier weight |
|---|---|---|---|
| `energy` | **2.13** | 0.89 | 0.14 |
| `arousal` | **1.88** | 0.85 | 0.18 |
| `valence` | 1.15 | 0.63 | 0.17 |
| `jitter` | 0.71 | 0.68 | 0.18 |
| `pause_ratio` | 0.56 | 0.66 | 0.04 |
| `speech_rate` | 0.39 | 0.58 | 0.07 |
| `shimmer` | 0.36 | 0.74 | 0.10 |
| `f0_std` | 0.30 | 0.64 | 0.13 |

The pooled-covariance version used by `ml/change_weights.py` gives somewhat smaller values (arousal −1.42, energy −1.19, valence −0.88, …; see «Current model»).

- Median ratio of within- to between-person variation: **0.70**.
- **Caveat:** clips come from one session and one mic, so day-to-day variation is larger – especially for `energy`, which depends on phone distance. `energy` and `arousal` numbers are optimistic.
- **Follow-up (B12):** on held-out folds no weighting beats equal weights:

| Weighting | Detectability on held-out speakers (D shift for a full relapse) |
|---|---|
| Equal | 1.58 |
| Classifier (refit per fold) | 1.58 |
| Per-feature sensitivity | 1.61 |
| Fisher (sign-constrained) | 1.54 |

Folds range from 0.8 to 3.1, so the differences are noise. Fisher looked best in-sample (1.66) – that was overfitting. Details: `ml/reports/acoustic/summary.md` part 2.

### Smoke test on our own recordings (14 Nov, 8 clips: Kasper and Mahdan, 2 good and 2 bad each)
Same DJI mic, same day. Audio and features from these recordings are **not** in the repo.

**Against the corpus** (`score_clip.py`, P(depressed-like)):
| | Good 1 | Good 2 | Bad 1 | Bad 2 |
|---|---|---|---|---|
| Kasper | 0.31 | 0.14 | 0.33 | 0.34 |
| Mahdan | 0.29 | 0.18 | 0.55 | 0.59 |

Mahdan is separated, Kasper is not.

**Against their own baseline** (mean of the 2 good clips; within-person SD from the corpus; each good clip compared with the other good clip):

| Engine version | Kasper: normal variation | Kasper: bad 1 / 2 | Mahdan: normal variation | Mahdan: bad 1 / 2 |
|---|---|---|---|---|
| Classifier weights, independence assumption | ±2.28 | 1.85 / 1.91 | ±1.29 | 3.71 / 4.30 |
| Per-feature sensitivity weights | ±1.70 | 3.34 / 2.96 | ±0.84 | 4.27 / 4.77 |
| **Current engine** (equal weights + correlation, B12) | ±1.07 | **0.71 / 0.84 → not separated** | ±0.53 | **1.90 / 2.42 → separated** |

- Kasper's change was mostly in arousal and valence, which are strongly correlated; the current engine counts them as one signal, not two.
- **We do not tune weights to these clips (B13):** the held-out test on 114 people outweighs 2 people.
- Caveats: n = 2; the "bad" clips may have been acted (exaggerated); one session (day-to-day variation is larger); SD taken from the corpus.

### Preconditions for this to work
A good healthy baseline is **necessary but not sufficient**.
1. **The baseline needs 1–2 weeks**, not one day (~7–14 check-ins across weekdays, weekend, morning and evening).
2. **The clinic confirms the starting point** (e.g. MADRS at the end of TMS), since remission often has residual symptoms.
3. **Stable recording conditions.** A new phone, another distance or a noisy room looks like change. This is the biggest practical risk (T14).
4. **Voice is not specific to depression.** Colds, sleep, alcohol, stress and medication cause false alarms, so we need more signal sources (T3, T4).
5. **The baseline ages.** It should be renewed periodically, but **only when a healthy state is confirmed** (e.g. by PHQ-9). Otherwise the boiling frog returns.
6. **Not everyone shows depression in their voice.** The average effect is moderate.
7. **The patient must check in regularly.**
8. **Simulator numbers rest on assumptions.** Real performance needs a longitudinal pilot.

### Improvement plan (prioritised)
1. ~~**T1** fixed healthy baseline~~ ✅ built. Simulator: 41/50 → 14/50 missed.
2. ~~**T13** change weights~~ ✅ tested: all weightings tie on held-out data, so the app uses equal weights (B12). The simulator uses measured noise.
3. **T14** recording protocol and quality check:
   - same question, ≥ 20 s, fixed distance
   - reject recordings with little speech, clipping or low SNR
   - loudness relative to background noise
4. ~~**T10** more robust baseline~~ ✅ median/MAD, morning and evening separately, ≥ 7 for the rolling track.
5. ~~**T15** correlation between features~~ ✅ D normalised with the measured within-person correlation (B12). Noise in D had been underestimated by 27 %.
6. ~~**T16** morning + evening together~~ ✅ all check-ins of a day are combined per branch (B16). Simulator: 40 → 18 missed of 100 at ×1.5.
7. **T12** calibrate thresholds in the simulator against an acceptable number of false alarms per patient-month.
8. **T3/T4** text and health branches: independent sources are the biggest real gain.
9. Higher AUC (more eGeMAPS features, gender normalisation, reading + interview tasks together). Least important for the product.
10. **After the hackathon:** a longitudinal pilot with PHQ-9. The only way to validate early warning and learn personal weights.

---

## 4. Decision log

| # | Decision | Why |
|---|---|---|
| B1 | Three branches, each with its own model and dataset | Different signals and datasets; easier to develop in parallel |
| B2 | **No corpus audio is sent to Gemini or any other API** | The Androids licence forbids redistribution |
| B3 | The Google requirement is met with Gemini in the text branch, not Vertex for acoustics | An M1 with MPS (Apple GPU) is fast enough (~0.1 s per 3 s of audio) |
| B4 | 8 interpretable features, not 88 or the wav2vec2 embedding | Explainable to clinicians; less risk of learning mic/room with 114 people |
| B5 | C chosen by log-loss, not AUC | AUC is scale-blind, so C ended at 0.001 and all probabilities were ~0.5 |
| B6 | Features whose sign flips against the literature keep the literature direction | The flip comes from collinearity; for within-person change we want the marginal direction |
| B7 | The browser converts recordings to 16 kHz mono WAV | No ffmpeg needed in the backend |
| B8 | Raw audio is deleted right after feature extraction | Privacy |
| B9 | Same extraction code in training (`ml/`) and app (`backend/app/pipeline/`) | Weights stay valid for the app's features |
| B10 | Branch scores are rescaled to unit variance | With weights summing to 1, D was a weighted mean with SD ≈ 0.36, so thresholds (S > 1, S > 2) were almost never reached |
| B11 | The simulator uses the real engine and corpus effect sizes | The demo shows what the system actually does; robustness measured over 50 seeds |
| B12 | **Equal weights + normalisation with within-person correlation** (`ml/change_weights.py`) | On held-out folds: equal 1.58, classifier 1.58, per-feature 1.61, Fisher 1.54 (D shift for a full relapse). Fold-to-fold range (0.8–3.1) is far larger, so the data cannot tell weightings apart → simplest wins. The correlation *is* measurable and keeps D at SD 1 (independence gave 1.27 → too many false alarms) |
| B13 | Weights are **not** chosen from our own test recordings | 2 people are a test, not training data. Tuning to them looks good but is overfitting |
| B14 | Healthy baseline: locks automatically after 14 days (clinician can reject), S = max(anchored, rolling), never re-anchored automatically | Locking without a clinician keeps the app from stalling. "Max of two" catches both drift and abrupt drops (simulator: 14/50 missed vs 15/50 anchored only). Automatic re-anchoring would bring back the boiling frog |
| B15 | Baseline scale = MAD shrunk toward the corpus within-person SD (k = 5) | 14–28 measurements give an uncertain personal SD, and one odd day must not ruin the baseline |
| B16 | All check-ins of a day are combined per branch as `Σ D / √n` (Stouffer), then averaged over branches | The day score was overwritten by the last check-in (a negative diary was erased by a neutral spoken check-in later the same day). `Σ D / √n` keeps SD 1 so thresholds keep their meaning, while a real shift grows with √n: two check-ins a day halve the missed relapses in the simulator. Assumes independent noise between check-ins |
| B17 | A feature needs ≥ `min(min_checkins, 5)` baseline values to get stats | Text features have no corpus SD to shrink toward; 1–2 written entries would give a guessed scale (fallback 1.0) |
| B18 | A baseline only locks if at least one feature has enough values; otherwise the period is extended | Without this, a mixed baseline (3 spoken + 3 written) locked with no stats → an active baseline that could never score and is never re-anchored automatically |

---

## 5. Open tasks

*T-number = task here. B-number = decision in section 4.*

- [x] **T1** Fixed healthy baseline in the app, with API, patient/clinician UI, tests and simulator (B14, B15)
- [x] **T2** The Gemini placeholder no longer adds random numbers to the score (`llm.ContentAnalysis.stub`, `pipeline/__init__.py`)
- [x] **T3** Text branch: `pipeline/text.py`, `weights/text.json`, `POST /api/checkins/text` for written diaries (§2b)
- [ ] **T3b** Real Gemini transcription (`gemini-3.5-transcribe`, verbatim, `store=False`) → the text branch then runs on spoken check-ins automatically (GitHub issue #3)
- [ ] **T4** Health branch: Fitbit/Google Health adapter, weights from the Depresjon dataset (minute-level activity, MADRS) (GitHub issues #5, #10)
- [ ] **T5** What to show before the healthy baseline is locked (population norms?)
- [ ] **T6** Consider down-weighting `energy` because of mic dependence
- [ ] **T7** Run the reading task as an extra check (`--task reading --no-write`)
- [ ] **T8** Bandit reward: "did the state improve?" (`S_{t+1} < S_t`) is not wired in (`routers/interventions.py`)
- [x] **T9** Test with our own recordings: 8 clips from Kasper and Mahdan (see «Smoke test»)
- [x] **T10** ≥ 7 check-ins for the rolling track; morning and evening separately in both tracks
- [x] Simulator on the real engine (`sim/simulator.py`, `/api/sim/run`, `/api/sim/robustness`)
- [ ] **T11** Demo view in the frontend: slider over days, S (rolling vs anchored) against self-report, what the bandit learned (types exist in `frontend/src/lib/api.ts`)
- [x] **T13** Change weights tested on held-out data: none beats equal weights, so equal weights are used (B12). The simulator uses measured noise and correlation
- [ ] **T14** Recording protocol and quality check: little speech, clipping, SNR; loudness relative to background noise
- [x] **T15** D normalised with the measured within-person correlation (`state.branch_deviation`)
- [x] **T16** All check-ins of a day combined per branch (B16); simulator: 2 check-ins/day roughly halve missed relapses
- [x] Flow fixes after the main merge: day score no longer overwritten (B16), minimum values per feature (B17), lock only with stats (B18), correct re-baselining progress (`baseline_building`)
- [ ] **T17** A branch with too few baseline values stays unscored until a new baseline. Consider letting a branch "join" an active baseline later once it has ≥ 5 values (only if the clinician confirms the patient is still well)
- [ ] **T12** Calibrate escalation thresholds: trade-off between detection speed and false alarms, over many simulator seeds

---

## 6. Running

```bash
# app
cd backend && .venv/bin/uvicorn app.main:app --reload     # :8000/docs
cd frontend && npm run dev                                # :5173
backend/.venv/bin/python -m pytest backend/tests          # engine + baseline tests

# acoustic model
bash ml/download_androids.sh                              # 3.7 GB → data/androids/ (accepts the licence)
backend/.venv/bin/python ml/extract_features.py           # ~15 min on an M1, resumable
backend/.venv/bin/python ml/train_weights.py              # → weights/acoustic.json + ml/reports/acoustic/
backend/.venv/bin/python ml/score_clip.py <file.m4a>      # score your own recording against the corpus
```

| What | Where |
|---|---|
| Reports and figures | `ml/reports/acoustic/` (`roc.png`, `coefficients.png`, `summary.md`) |
| Data | `data/` is gitignored and must **never** be committed |
