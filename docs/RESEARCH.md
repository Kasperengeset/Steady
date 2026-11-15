# Research, evidence and design decisions

[Project overview](../README.md) · [Function guide](TECHNICAL_GUIDE.md) · [Clinical walkthrough](CLINICIAN.md)

Steady connects three kinds of evidence: published research that motivates the problem and methods, repository experiments on model components, and product decisions that still need testing. They answer different questions. A good public-dataset classifier, a convincing synthetic demonstration and a useful clinical follow-up tool are not interchangeable claims.

This guide describes the integrated implementation. [STATUS.md](../STATUS.md) preserves the team's longer decision history, including earlier approaches and experiments that are not active in the app.

## Follow-up after TMS

**Decision:** start from a post-treatment follow-up use case and compare a person with a reference collected during an intended period of wellness.

**Published basis:** Senova and colleagues' review examines how antidepressant response to rTMS is sustained after the acute course. It supports the importance of the follow-up question; it does not test smartphone monitoring or Steady. [Senova et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30344109/)

**Project interpretation:** a person's previous measurements may be a useful reference for noticing change. That is a design hypothesis, not evidence that the current state score predicts recurrence. The software can lock a reference after coverage requirements are met, but cannot verify clinical remission. The clinician can review, reject and replace it.

**Implementation:** [baseline.py](../backend/app/baseline.py), [baseline endpoints](../backend/app/routers/baselines.py).

## Daily check-ins and prompts

**Decision:** invite one short check-in at a random allowed time, with additional voluntary entries and a written alternative.

Ecological momentary assessment studies experiences repeatedly in ordinary settings, close to when they happen. It motivates the sampling approach and attention to recall and selection effects. It does not establish that a particular daily memo schedule is effective after TMS. [Shiffman, Stone & Hufford, 2008](https://doi.org/10.1146/annurev.clinpsy.3.022806.091415)

The default **09:00–21:00 local window**, **120-minute bonus window**, quiet-hour controls and roughly **30-second memo** are project choices. A missed or late prompt is not a negative clinical observation. Late entries still count; the reward bonus is separate from clinical scoring.

Random timing also creates a measurement problem: observations taken at different hours may differ for ordinary reasons. The engine fits a shrunk hour-of-day slope from the reference data. The [timing and repeated-entry simulations](../ml/reports/engine/design_checks.md) compare alternatives under explicit synthetic assumptions; they do not measure real-world alert rates.

**Implementation:** [prompt.py](../backend/app/prompt.py), [state.hour_slope / adjust_for_hour](../backend/app/engine/state.py), [push.py](../backend/app/push.py).

## The fixed reading passage

**Decision:** record the fixed passage separately from the free memo, check that it was read, and keep its measurements separate from spontaneous speech.

Holding words constant is intended to reduce variation due to changing content. The team's Androids reading/interview analysis found lower within-speaker variability for reading segments, but also smaller depression-related shifts; overall detectability did not clearly improve. This is a within-session corpus experiment, not proof of better day-to-day monitoring. [Repository analysis](../ml/reports/voice/reading_vs_interview.md); [Androids paper](https://www.isca-archive.org/interspeech_2023/tao23_interspeech.html).

The current implementation stores eligible `phrase_*` features without giving them drift weight. The fixed words never enter the diary classifier. A failed optional reading check does not discard the free memo. The proposed phrase-validation gate and the recording protocol are recorded in [STATUS.md](../STATUS.md).

**Implementation:** [pipeline.add_phrase](../backend/app/pipeline/__init__.py), [phrase.reading_check](../backend/app/pipeline/phrase.py), [phrase_check.py](../ml/voice/phrase_check.py).

## Voice and language

### Acoustic measurements

GeMAPS/eGeMAPS provides a documented acoustic feature family and a common extraction approach. Steady uses openSMILE to derive six measurements, then adds arousal and valence from an existing audEERING wav2vec2 emotion model. This mixes interpretable signal measurements with a pretrained neural representation; Steady did not train that emotion network. [Eyben et al., 2016](https://doi.org/10.1109/TAFFC.2015.2457417); [audEERING model card](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim).

The Androids Corpus supplies speech-based depression research data. The repository's acoustic experiment uses 114 eligible Italian interview speakers and speaker-independent folds. Its eight-feature classifier has mean fold AUC 0.770; the live app instead uses equal-weight personal deviations. Candidate weighting methods performed similarly in the team's held-out comparison, so the app uses the simpler weighting. [Corpus paper](https://www.isca-archive.org/interspeech_2023/tao23_interspeech.html); [experiment and weighting results](../ml/reports/acoustic/summary.md).

Microphone, distance, room, language and illness can affect the same measurements. In particular, energy is sensitive to recording conditions. Corpus clip-to-clip variability can underestimate real day-to-day variability. Neither the eight-feature classifier nor the pretrained emotion output is a clinical probability of relapse.

### Language measurements

The Reddit Mental Health Dataset provides the text training source. Its original research concerns language in online communities. The project's model learns to distinguish r/depression writing from selected comparison forums; forum membership is not a clinical diagnosis. [Low et al., 2020](https://www.jmir.org/2020/10/e22635/).

The repository separates authors before splitting data into chunks, masks forum names and treatment vocabulary, tunes on validation data and reports a held-out test. Masking is an engineering defence against shortcuts: mentioning treatment can occur during recovery as well as distress. The deployed classifier is exported as JSON and evaluated with the same preprocessing used in the app. [Text training guide](../ml/text/README.md); [held-out results](../ml/reports/text/v1.md).

Absolutist-word frequency, first-person language and VADER sentiment remain inspectable measurements. Their small installed weights should not be read as universal psychological rules. The classifier carries most of the branch weight. VADER supplies a general sentiment method, not a depression diagnosis. [VADER implementation and paper reference](https://github.com/cjhutto/vaderSentiment).

The team's context checks expose errors with recovery stories, negation, other people's feelings and writing style. Optional Gemini ratings attempt to handle the writer's present context, but are disabled by default and have zero installed scoring weight. Their team-written evaluation is not independent clinical validation. [Context report](../ml/reports/text/context_check.md); [content-rating implementation](../backend/app/pipeline/llm.py).

## Wearable data

**Decision:** use available activity, sleep and recovery measurements as another personal-observation branch, while keeping engagement rewards separate from scoring weights.

Depresjon provides repeated actigraphy recordings from 23 participants in its condition group and 32 controls. The repository evaluates activity features with leave-one-person-out validation and uses those results to inspect directions and variability. Two computed activity features—night activity and relative amplitude—currently have zero drift weight. [Dataset paper](https://doi.org/10.1145/3204949.3208125); [repository health report](../ml/reports/health/summary.md).

Actiwatch activity counts and Fitbit steps are different measurements. Personal standardization removes a numerical scale factor; it does not establish that devices, missingness patterns or derived rhythms are equivalent. Depresjon also includes unipolar and bipolar presentations. Its classification result is not validation of the live wearable score.

Sleep, sleep midpoint, resting heart rate and HRV directions are literature-informed priors rather than features trained on sleep/HRV labels in Depresjon. Longitudinal wristband research motivates investigating sleep features, while HRV research also illustrates associations with depression and medication effects. Neither establishes the app's particular weights or directional rules. [Wearable sleep study, 2021](https://mhealth.jmir.org/2021/4/e24604/); [Kemp et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20138254/).

Medication, fitness, physical illness and sleep timing can confound those directions. The priors need separate validation, and simple monotonic rules cannot represent every pattern of change. Active minutes can earn food but have no drift weight.

The integration adds only the latest completed wearable day, retains its measurement date and source, and does not rewrite old scores or freeze a partial current day as a complete observation. Repeated syncs are deduplicated. These are data-integrity decisions, not findings about clinical accuracy.

**Implementation:** [health.py](../backend/app/pipeline/health.py), [google_health.py](../backend/app/pipeline/google_health.py), [health_sync.py](../backend/app/health_sync.py), [installed health weights](../backend/app/engine/weights/health.json).

## Personal reference and change detection

**Decision:** use transparent, baseline-relative arithmetic rather than present a single population-classifier probability as a person's clinical state.

The engine uses median/MAD summaries, available within-person variability priors, direction-adjusted deviations and correlation-aware weighted sums. These are statistical construction choices. Their application to this setting is assessed in the repository experiments; the exact reference requirements are not prescribed by a TMS trial.

An **anchored track** keeps the locked reference fixed. A **rolling track** compares with recent history. The anchor is intended to avoid absorbing gradual deterioration into the reference; the rolling track provides a second comparison with recent observations. Taking the maximum favours sensitivity to either pattern and has an alerting trade-off. This particular two-track rule is a project design choice.

EWMA gives recent days more influence while retaining earlier history. The general method comes from statistical process monitoring. It does not supply Steady's clinical thresholds. [Roberts, 1959](https://doi.org/10.1080/00401706.1959.10489860).

The **14-day windows**, **alpha 0.3**, **half-weight extra entries**, shrinkage strength and escalation rules are configurable prototype settings. The [engine experiments](../ml/reports/engine/design_checks.md) test their behaviour under generated timing, noise and adherence scenarios. Gaussian/unit-noise interpretations depend on modelling assumptions and estimated correlations.

The application omits missing observations. Experiments that added non-participation to the score produced undesirable trade-offs, so engagement is currently used for companion behaviour rather than a scored depression signal. A person being busy should not automatically become evidence of illness.

**Implementation:** [baseline.py](../backend/app/baseline.py), [state.py](../backend/app/engine/state.py), [check-in scoring](../backend/app/routers/checkins.py), [escalation.py](../backend/app/engine/escalation.py).

## Small actions and adaptive selection

Behavioural activation research motivates attention to manageable activities and participation. A list of walks, music, social contact or small tasks in an app is not itself a delivered, validated behavioural-activation treatment. [Ekers et al., 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0100100).

Adaptive-intervention research motivates using observations and feedback to tailor support over time. Steady first narrows eligible activities using the available signal, then samples from per-person Beta posteriors. Thompson sampling supplies that exploration/exploitation mechanism. [Nahum-Shani et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/); [Thompson, 1933](https://doi.org/10.1093/biomet/25.3-4.285).

The live update currently uses completion and a helpfulness rating. The reward function has a place for next-day state improvement, but the endpoint passes `None` for that term. Corrected feedback replaces the day's previous posterior update. The README therefore does not claim the app has learned which activities prevent relapse or improve a clinical outcome.

**Implementation:** [signals.fitting_arms](../backend/app/engine/signals.py), [bandit.py](../backend/app/engine/bandit.py), [intervention feedback](../backend/app/routers/interventions.py).

## Engagement and social design

Gamification research motivates exploring feedback, progression and rewards, with heterogeneous methods and evidence across applications. It does not validate Appie, pet food, streaks or this circle. Their effect on retention, burden and wellbeing remains a user-research question. [Sardi, Idri & Fernández-Alemán, 2017](https://pubmed.ncbi.nlm.nih.gov/28536062/).

The product rules reward qualifying participation equally on difficult and good days. A rest-day allowance softens streaks; lifetime earned XP survives feeding. Companion messages use timing and memo length rather than analysing the person's words or reading the clinical score. Those are deliberate experience and separation-of-responsibility choices.

Social consent is separate, sharing fields are opt-in, and reactions are preset. The friend feed is constructed from an allow-list rather than an entire clinical object. A support card has a patient-visible hold/cancellation mechanism. These choices reduce disclosure and pressure; they do not establish anonymity, clinical benefit or production-grade access control.

**Implementation:** [rewards.py](../backend/app/rewards.py), [appie.py](../backend/app/appie.py), [companion rules](../backend/app/engine/companion.py), [social endpoints](../backend/app/routers/social.py).

## Inspectable evidence and privacy

New observations save their reference values, feature shares, weights, artifact identifiers and state transition in the same transaction as the clinical entry. This supports checking the calculation that actually happened. Older entries without snapshots remain distinguishable from a saved ingestion-time decision.

Feature contributions explain the implemented arithmetic. They do not establish that a feature caused a change in a person's health. Likewise, a classifier's word attribution explains its language score rather than a clinician's diagnosis.

Raw recording deletion reduces local retention but prevents later re-extraction from that waveform. Saved measurements and snapshots preserve the score calculation, not the original sound. Configured cloud transcription still transfers audio to a provider. Local OAuth files, transcripts and measurements remain sensitive data; the prototype has not established production privacy or clinical governance.

**Implementation:** [clinical.capture_analysis](../backend/app/clinical.py), [explanation.py](../backend/app/explanation.py), [raw-audio lifecycle](../backend/app/routers/checkins.py).

## What still needs evaluation

- Prospective longitudinal performance after TMS: reference quality, false alerts, missed deterioration, lead time and clinician workload.
- Cross-language, demographic, recording-device and wearable-device robustness.
- The fixed-phrase scoring gate, content-rating reproducibility and appropriate handling of missing observations.
- Retention, perceived pressure, accessibility, social consent and the value of Appie in ordinary use.
- Production authentication, role authorization, data protection and a clinically reviewed escalation workflow.

The public-corpus reports establish component experiments. The append-only synthetic demo establishes inspectable software behaviour. Neither replaces these evaluations.
