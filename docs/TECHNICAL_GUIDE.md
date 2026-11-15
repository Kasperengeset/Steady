# Technical and function guide

[Project overview](../README.md) · [Research and decisions](RESEARCH.md) · [Clinical walkthrough](CLINICIAN.md)

This is a map of the application's main functions and how they work together. It covers the core processing and product flows; small rendering helpers and framework callbacks remain documented next to their source. FastAPI's generated `/docs` page supplies the current request/response schemas.

## System boundaries

The React/Vite progressive web app has patient, clinician and friend routes. FastAPI owns inference, SQLite persistence, rewards, health connection and push delivery. Training/evaluation code lives in `ml/`; the live app reads exported model artifacts and branch configuration files.

Three concepts are kept separate:

- **Observation:** something submitted or received, with its timestamp, source, origin and extracted measurements.
- **Clinical state:** a calculation against personal reference measurements and previous scored days.
- **Engagement:** qualifying participation, food, lifetime XP, habits and companion messages.

A low clinical score is not required to earn food. Appie's friendly message rules do not inspect the diary contents or the clinical drift score.

## One check-in, end to end

1. The patient records a memo or writes an entry. The recording flow converts browser audio to mono WAV and preserves a draft if saving fails. A request key is reused when retrying the same submission.
2. A check-in endpoint validates the patient and request, then runs the appropriate pipeline. Uploaded audio is held temporarily and removed in the endpoint's cleanup path.
3. `_record_checkin()` stores the observation and features, starts or advances a reference if needed, and computes whichever reference-track comparisons are available.
4. The day's entry-weighted branch scores and smoothed state are updated. Review rules and today's action use the available result; rewards are reconciled through the canonical ledger.
5. `capture_analysis()` records the reference, feature calculations, artifact identifiers and before/after state. The entry, evidence and response are committed together. A retry can return the saved response rather than repeat the observation.
6. The patient receives the result. The clinical entry inspector reads the saved snapshot; it does not rerun the original inference.

**Coordinator:** [routers/checkins.py](../backend/app/routers/checkins.py), especially `create_checkin()`, `create_text_checkin()`, `create_wearable_checkin()`, `_saved_response()`, `_score_tracks()` and `_record_checkin()`.

## Feature extraction functions

| Function | Job and result | Source |
|---|---|---|
| `audio.load()` | Decode supported audio, average channels, resample when needed and return mono float32 samples at 16 kHz. | [audio.py](../backend/app/pipeline/audio.py) |
| `audio.active_seconds()` / `speech_span()` | Estimate active-energy duration and the first-to-last speech span. These serve reward qualification and reading checks, not diagnosis. | [audio.py](../backend/app/pipeline/audio.py) |
| `prosody.extract_full_signal()` / `from_full()` | Run eGeMAPS extraction, then map the full output to the six live prosodic measurements. | [prosody.py](../backend/app/pipeline/prosody.py) |
| `emotion.run_signal()` | Run the pretrained emotion network and return its embedding and dimensional estimates. The live acoustic branch uses arousal and valence. | [emotion.py](../backend/app/pipeline/emotion.py) |
| `llm.transcribe()` / `analyze()` | Request an English verbatim transcript when configured. Failure or a missing key is marked unavailable; placeholder text is not passed into the diary classifier. | [llm.py](../backend/app/pipeline/llm.py) |
| `text.prepare()` / `mask_text()` | Clean text and mask forum/treatment vocabulary consistently with model training, including breaks that prevent bigrams across removed terms. | [text.py](../backend/app/pipeline/text.py) |
| `text.text_logit()` / `lexicon_features()` | Compute the exported TF-IDF/logistic-regression logit and auxiliary absolutist, first-person and sentiment measurements. | [text.py](../backend/app/pipeline/text.py) |
| `text.explain()` | Return words or word pairs contributing to the language classifier. This explains the classifier, not a clinical diagnosis. | [text.py](../backend/app/pipeline/text.py) |
| `text.crisis_hits()` / `analyze_text()` | Apply the deterministic text phrase check and assemble text features. | [text.py](../backend/app/pipeline/text.py) |
| `llm.rate_text()` | Optionally request present-self context ratings and supporting evidence. Disabled by default; currently unweighted by the state engine. | [llm.py](../backend/app/pipeline/llm.py) |
| `phrase.reading_check()` | Check the reading's duration/rate and available transcript against the fixed passage. | [phrase.py](../backend/app/pipeline/phrase.py) |
| `health.daily_activity_features()` | Turn 1,440 minute values into daily activity, inactivity, night activity, relative amplitude and most-active-window onset. | [health.py](../backend/app/pipeline/health.py) |
| `health.wearable_features()` | Validate the available wearable summaries and assemble their feature dictionary; omit missing values. Active minutes are stored for engagement. | [health.py](../backend/app/pipeline/health.py) |

The [pipeline coordinator](../backend/app/pipeline/__init__.py) exposes `extract_features()` for free speech, `extract_text_features()` for writing and `extract_wearable_features()` for wearable data. `add_phrase()` adds separately named `phrase_*` measurements only when its reading checks pass. The fixed passage never enters the free-text classifier.

Without a transcript, a spoken memo still has acoustic measurements, but there is no language analysis of its meaning. Acoustic arousal and valence do not identify the words spoken.

## Baseline lifecycle

| Function | Job | Source |
|---|---|---|
| `start()` | Create a versioned building reference. A superseded building reference is rejected; an existing active reference can continue scoring. | [baseline.py](../backend/app/baseline.py) |
| `progress()` | Report collection duration, observation counts, status and optional clinician confirmation. | [baseline.py](../backend/app/baseline.py) |
| `try_lock()` | Lock an eligible reference after its period and coverage conditions are met; save robust per-feature statistics and retire the previous active version. | [baseline.py](../backend/app/baseline.py) |
| `stats_for()` | Return reference centre, scale and count, with the configured pooled fallback. | [baseline.py](../backend/app/baseline.py) |
| `hour_slopes()` | Retrieve the baseline's fitted time-of-day adjustments. | [baseline.py](../backend/app/baseline.py) |
| `reject()` | Reject a reference and begin a replacement collection period. | [baseline.py](../backend/app/baseline.py) |

Defaults come from [config.py](../backend/app/config.py): 14 collection days, 10 observations overall, feature coverage normally at least 5, a 14-day rolling window and at least 7 rolling values for an eligible feature. Coverage can extend collection beyond the nominal period. These are prototype settings; a reference becoming active does not confirm clinical remission.

## Score functions and arithmetic

The following functions live in [engine/state.py](../backend/app/engine/state.py).

| Function | What it does |
|---|---|
| `load_branch()` | Load installed weights, feature directions and metadata; use the defined prior fallback when an artifact is absent. |
| `robust_stats()` | Compute a median and MAD-based scale, with available corpus variability shrinkage and a nonzero-scale fallback. |
| `prior_sd()` / `history_stats()` | Retrieve a corpus variability prior and construct eligible recent-history statistics. |
| `hour_slope()` | Fit a personal linear hour effect with shrinkage toward no hour effect. |
| `adjust_for_hour()` | Adjust a measurement to the reference hour; leave it unchanged if adjustment information is missing. |
| `zscores()` | Convert available measurements into direction-adjusted deviations from the chosen reference. |
| `feature_shares()` / `branch_deviation()` | Calculate normalized weighted feature contributions and their branch sum. |
| `checkin_weight()` | Give full weight to a prompted entry or the day's first entry in that branch; otherwise use the configured half weight. |
| `combine_checkins()` | Combine that day's repeated branch observations with weighted Stouffer-style normalization. |
| `daily_deviation()` | Average the branch deviations available for the day. |
| `ewma()` | Smooth a daily deviation against the previous state; initialize with the first available daily deviation. |
| `top_contributors()` | Summarize prominent feature shares for the app's explanations and action-selection context. |

For each reference track, the calculation is:

```text
adjusted_i = measured_i - hour_slope_i * (local_hour - reference_hour_i)
z_i        = direction_i * (adjusted_i - centre_i) / scale_i
D_entry,b  = sum(weight_i * z_i) / sqrt(weight^T * correlation * weight)
D_day,b    = sum(entry_weight_j * D_entry,b,j) / sqrt(sum(entry_weight_j^2))
D_day      = mean(available D_day,b)
S_day      = alpha * D_day + (1 - alpha) * S_previous_scored_day

decision state = max(available anchored S_day, rolling S_day)
```

The first state is initialized directly from its daily deviation. Without a supplied within-person correlation, the feature-normalization fallback assumes independence. Missing values are excluded rather than imputed as neutral measurements. A feature measured two personal scale units below its centre with direction `-1` has `z = +2`; this is a deviation, not a 200% illness probability.

The **anchored** comparison retains the locked reference. The **rolling** comparison uses recent observations adjusted with the active reference's hour slopes. An additional entry on the same scoring day updates that day's aggregate; the saved previous-day state makes the EWMA arithmetic inspectable.

## Clinical evidence functions

| Function | Job | Source |
|---|---|---|
| `capture_analysis()` | Build the ingestion snapshot: origin, references, raw measurements, adjusted values, z-scores, shares, entry weights, artifact hashes, settings and state before/after. | [clinical.py](../backend/app/clinical.py) |
| `model_catalog()` | Describe installed branch artifacts, report availability, evaluation scope and limitations. Configured status is not proof an extractor ran on a particular entry. | [clinical.py](../backend/app/clinical.py) |
| `entries()` / `entry_detail()` | Supply the paginated observation log and a selected entry's saved evidence. Legacy entries receive explicitly labelled reconstruction when possible. | [clinical.py](../backend/app/clinical.py) |
| `overview()` / `workspace()` | Assemble cohort triage and the selected patient's longitudinal clinical workspace. | [clinical.py](../backend/app/clinical.py) |
| `latest_explanation()` | Collect stored daily evidence for the selected reference track and request its longitudinal explanation. | [explanation.py](../backend/app/explanation.py) |
| `day_feature_contributions()` / `explain()` | Attribute the stored state to feature/day contributions and the retained older-history term. | [engine/explain.py](../backend/app/engine/explain.py) |
| `clinical_demo.seed()` | Append or reopen an isolated drift, stable or building demonstration. Generated diaries run the local text classifier; acoustic measurements are simulated and wearable inputs generated. | [clinical_demo.py](../backend/app/clinical_demo.py) |

An ingestion snapshot and a later reconstruction are different evidence types. A saved feature contribution explains a calculation, not a causal medical mechanism. Transcription availability and each entry's actual measurements should be inspected independently of the model catalog.

## Action selection and review rules

| Function | Job | Source |
|---|---|---|
| `signals.fitting_arms()` | Choose the eligible activity keys from the available signal context, or retain the available set when nothing suitable stands out. | [signals.py](../backend/app/engine/signals.py) |
| `bandit.choose_arm()` | Draw a Beta sample for each eligible activity and select the largest sample. | [bandit.py](../backend/app/engine/bandit.py) |
| `bandit.reward()` / `update()` | Turn completion, rating and optional next-day improvement into a fractional reward; update the activity's Beta parameters. | [bandit.py](../backend/app/engine/bandit.py) |
| `interventions.feedback()` | Persist feedback and replace a previous update when corrected, avoiding another trial on a retry. The live call passes no next-day improvement. | [interventions.py](../backend/app/routers/interventions.py) |
| `escalation.evaluate_level()` | Apply the prototype sustained-deviation or immediate crisis-flag rules. | [escalation.py](../backend/app/engine/escalation.py) |
| `escalation.elevated_flag()` | Identify the separately configured unusually negative anchored-state pattern for clinician review. | [escalation.py](../backend/app/engine/escalation.py) |

Current feedback arithmetic is `0.5 * completed * ((rating - 1) / 4)` plus an optional `0.5` improvement term. Since the live endpoint passes `state_improved=None`, that second term contributes zero. The action learner therefore adapts to recorded participation/helpfulness, not measured clinical improvement.

`evaluate_level()` uses the available scored-state sequence: `S > 1` for three observations gives level 1; `S > 2`, or seven values above 1, gives level 2; a crisis flag gives level 3. Missing calendar days should not be interpreted as measured states. These thresholds and the clinician-only elevation setting are prototype rules. Social disclosure remains subject to separate consent and support-card controls.

## Rewards, habits and Appie

| Function | Job | Source |
|---|---|---|
| `memo_counts()` | Check speech-duration or written-word qualification for the daily memo reward. | [rewards.py](../backend/app/rewards.py) |
| `sync()` | Reconcile eligible memo, prompt, action, follow-up and streak earnings in the canonical ledger. | [rewards.py](../backend/app/rewards.py) |
| `balance()` / `xp()` / `spend()` | Read available food, read positive lifetime earnings, and record a validated spend. | [rewards.py](../backend/app/rewards.py) |
| `streak_at()` / `current_streak()` / `week()` | Calculate activity history with the configured rest-day allowance and the person's local date. | [rewards.py](../backend/app/rewards.py) |
| `appie.view()` | Assemble the patient screen's companion, habits, wallet, level, prompt and message data. | [appie.py](../backend/app/appie.py) |
| `appie.feed()` | Spend food with retry protection; preserve lifetime earned XP and unlocked appearances. | [appie.py](../backend/app/appie.py) |
| `add_habit()` / `change_habit()` / `tick_habit()` | Create, edit and complete custom habits, applying duplicate protection and daily reward limits. | [appie.py](../backend/app/appie.py) |
| `on_wearable()` | Mark qualifying supported wearable habits and award their food without coupling it to the state score. | [appie.py](../backend/app/appie.py) |
| `migrate_legacy()` | Import earlier local Appie balances/habits into the canonical ledger without deleting clinical rows; safe to repeat. | [appie.py](../backend/app/appie.py) |
| `companion.decide()` / `evaluate()` | Choose and persist a reviewed engagement-based message, respecting timing, quiet hours and pause. | [engine/companion.py](../backend/app/engine/companion.py) |

The canonical daily memo reward is 10 food, with a 5-food prompt-window bonus. Qualification normally needs at least 10 seconds of active speech or 20 written words. Custom habits earn 2 food each, capped at three per local day. These amounts are product settings; they are not clinical targets. Appie's display uses lifetime earnings for progression and one ledger for the spendable balance.

## Prompt, social and health functions

| Function or module | Responsibility |
|---|---|
| [prompt.allowed_intervals() / draw_minute()](../backend/app/prompt.py) | Construct permitted local minutes after quiet-hour exclusions and draw uniformly from them. |
| [prompt.todays() / answer() / update_settings()](../backend/app/prompt.py) | Persist the day's prompt, associate an eligible answer, and manage timing, pause and enabled preferences. |
| [push.deliver_due() / run_loop()](../backend/app/push.py) | Deliver configured, due Web Push prompts and avoid stale or quiet-time delivery. |
| [social router](../backend/app/routers/social.py) | Handle identity presentation, separate social consent, membership, field-level sharing, reactions, leaving and support-card cancellation. |
| [health router](../backend/app/routers/health.py) | Start OAuth with PKCE and expiring one-use state, process the callback, and expose connection/sync/disconnect controls. |
| [GoogleHealth.fetch_days()](../backend/app/pipeline/google_health.py) | Fetch and parse the signed-in account's read-only wearable measurements into the shared feature-input format. |
| [health_sync.ingest()](../backend/app/health_sync.py) | Add only the latest completed measurement day at receipt time, retain the original date and source, and deduplicate repeated syncs. |
| [health_sync.sync() / sync_connected_patients()](../backend/app/health_sync.py) | Coordinate individual and periodic connected-account sync. |
| [health_sync.demo() / disconnect()](../backend/app/health_sync.py) | Add an explicitly generated sample or remove the local connection and saved token. |

Pseudonymous circle display does not anonymize the clinician's patient record. A friend token is a demo membership mechanism, not production clinical authentication. Health tokens are local ignored files, not database exports or circle content.

## API and screen map

All API paths below are under `/api`; see `/docs` for schemas.

| Area | Main endpoints / screens |
|---|---|
| Observations | `POST /checkins`, `/checkins/text`, `/checkins/wearable`; patient `/checkin`. |
| Clinical evidence | `GET /clinician/overview`, `/clinician/patients/{id}/workspace`, `/clinician/patients/{id}/entries`, `/clinician/entries/{id}`; desktop `/clinician`. |
| Demo history | `POST /clinician/demo` with `drift`, `stable` or `building`. |
| Reference management | `/patients/{id}/baseline` and its `start`, `confirm`, `reject` actions. |
| Companion and habits | `/patients/{id}/appie`, `/patients/{id}/appie/feed`, `/patients/{id}/appie/habits`; patient `/appie` and `/habits`. |
| Suggested-action feedback | `POST /interventions/feedback`; patient `/followup`. |
| Health connection | `/patients/{id}/health` plus `/patients/{id}/health/connect`, `/patients/{id}/health/sync`, `/patients/{id}/health/demo`, `/patients/{id}/health/connection`; OAuth `/google-health/callback`. |
| Social and prompts | Contracts in [social.py](../backend/app/routers/social.py), [prompts.py](../backend/app/routers/prompts.py) and [push.py](../backend/app/routers/push.py); patient `/circle`, friend `/join`. |

The React route definitions are in [App.tsx](../frontend/src/App.tsx). Typed client calls live in [frontend/src/lib](../frontend/src/lib). The API health endpoint reports capabilities so an older backend paired with a newer frontend is easier to diagnose.

## Persistence and reproducibility

[models.py](../backend/app/models.py) stores patients, observations, feature rows, versioned references, daily states, intervention logs and review records. Additional model modules define prompting, social and Appie state. [db.py](../backend/app/db.py) creates tables and applies the repository's additive SQLite compatibility updates.

New entries keep their ingestion-time `analysis_json` and saved response. Source dates and data-origin labels distinguish generated, live and unknown legacy observations. Patient-facing retries, feeding and wearable sync have duplicate protections. The clinical demonstration appends its own labelled profile rather than resetting the current database.

Model reports are linked in [the research guide](RESEARCH.md). Reproducing those experiments requires the relevant permitted datasets; running the app uses the installed artifacts. Raw recordings are deleted, so saved measurements reproduce the score arithmetic rather than allow a fresh extraction from the original audio.
