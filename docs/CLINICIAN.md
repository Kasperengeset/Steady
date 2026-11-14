# Clinical workspace and demo

Open `http://localhost:5173/clinician`. The desktop shows the cohort, review priority, baseline readiness, recent entries and coverage of the three model branches. Patient workspaces show the longitudinal state, individual observations, reference measurements and model evidence. The view refreshes every five seconds while visible, on return to the tab and after a submitted entry.

## Demonstrate the full pipeline

1. Press **Demo workspace** on the clinical overview. This creates one isolated demonstration profile, or opens the existing one. It does not replace or reset any patient.
2. Inspect the generated history: fourteen reference days followed by ten days of increasing deviation. Its synthetic origin is displayed beside the history and baseline. Acoustic measurements are simulated; wearable measurements are generated. The synthetic diaries run through the installed local text model, and all observations use the ordinary state engine.
3. Open an entry to see its input, raw model measurements, personal reference centre and scale, signed deviations, feature contributions, branch scores, observation weight and state before/after it arrived.
4. On the demo profile, submit an original diary through **Live analysis**. For example, describe an ordinary day in 50–100 English words. The ordinary check-in endpoint runs the installed text model and persists the actual result. The new entry is labelled **live**, independently of its synthetic reference. Retries use the same request key and cannot duplicate its score or rewards.
5. Inspect the new entry and its language evidence. The trend is smoothed over days, so a new entry need not produce a large visible jump. Its measurements and contributions remain visible regardless of the size of that change.
6. To exercise voice, submit an actual recording. The backend decodes it, extracts openSMILE prosody and audEERING emotion features, and requests transcription when configured. Available transcripts also run through the text branch. Missing transcription is shown explicitly; it is never replaced by invented text. Raw audio is deleted after extraction.

The append-only demo endpoint is `POST /api/clinician/demo` with optional JSON `{"scenario":"drift"}`. `stable` creates a separate steady history; `building` creates a profile still collecting its reference. The equivalent CLI is `python scripts/seed_clinical_demo.py --scenario drift` from the repository root. Each story has a persistent unique key. Repeating it preserves live entries and does not duplicate generated history.

## What the model results mean

The text branch uses the installed masked TF-IDF/logistic regression artifact plus lexical measurements. Its classifier output describes a forum-trained depressive-language pattern; it is not a clinical probability of depression. Word attribution explains that classifier, while baseline deviations explain the person's drift score.

The voice branch measures six prosodic features and two emotion dimensions. The live state engine combines their changes from a personal reference, using equal weights and correlation normalization. It does not emit a voice diagnosis probability. The health branch uses available wearable measurements; absent inputs are omitted. Sleep, resting heart rate and HRV directions are literature priors. Active minutes can earn habit food but have no drift weight.

The model cards identify the installed weights by SHA-256 and show the corresponding repository report, held-out evaluation design and limitations. The reported public-dataset classification results are evidence about the model components. The synthetic demo demonstrates software behavior; neither establishes clinical relapse-prediction performance. Fixed-phrase features stay separate and have no drift weight until validated.

The anchored track compares entries with the locked reference; the rolling track compares with recent observations. Hour-of-day adjustment and the existing weighted extra check-in logic remain in effect. The exact score explanation attributes the accumulated state to features and dates. Notification thresholds and the elevated marker are decision-support signals from the existing engine.

## Saved evidence and older entries

New check-ins atomically save their measurements and an `analysis_json` snapshot: artifact hashes, reference version and origin, exact centres/scales, time adjustment, signed z-scores, per-feature shares, branch deviations, observation weights, engine settings, previous-day state and state before/after. The previous-day state makes the same-day EWMA arithmetic reproducible. A failure to save that evidence fails the transaction. Reading an entry does not run inference again or change its stored result.

Rows created before this update retain their original measurements and transcripts. Their comparison can be reconstructed using available references and current artifacts, but their original per-entry decisions were not saved. The inspector labels that reconstruction and leaves historical before/after values unknown. Unmarked legacy data has unknown origin; it is never silently relabelled live.

Wearable imports preserve the measurement date and source separately from the time the completed day was received. Generated health samples are labelled synthetic. API tokens, raw recordings, private model responses and request keys are not included in clinical exports.

Clinical endpoints:

- `GET /api/clinician/overview`
- `GET /api/clinician/patients/{id}/workspace?days=30` (7–120 days)
- `GET /api/clinician/patients/{id}/entries?limit=50&offset=0`
- `GET /api/clinician/entries/{entry_id}`

This remains a local demonstration app with patient selection rather than production authentication. Keep the server bound to localhost for this workflow.

## Verification

Backend tests use their own temporary database and disable cloud calls and background sync. The clinical tests check exact snapshot arithmetic and reference immutability, missing sources, old-entry reconstruction, append-only seeding, actual text inference, idempotent retries and API error cases. Frontend checks include the production TypeScript/Vite build and recording/service-worker tests. `node --test scripts/test-clinical.mjs` from `frontend` also renders the inspector with controlled data to verify patient-context isolation, visible model evidence and missing transcription. A manual browser pass is still needed for layout, physical microphone permission, Google consent and device push delivery when browser control is unavailable.
