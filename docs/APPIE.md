# Steady and Appie integration

The patient app keeps Mahdan's sage, sand and cream design and generated Appie artwork. It uses Kasper's canonical prompt, rewards, social, phrase and scoring modules. There is one reward ledger. The Appie API is an additional read model at `/api/patients/{id}/appie`; the existing `/companion`, `/prompt`, `/profile`, `/circle` and `/rewards` contracts remain available.

## What to test

1. Open Today. Appie, your food, earned XP, streak, daily prompt and today's small action should load.
2. Open Check in. Read the fixed Rainbow Passage or skip that step, record your day, listen back, and save. Writing is also available. A failed save keeps the draft, and retries reuse a request key.
3. Open Habits. Complete a custom hobby, check the week, or change reminder timezone, quiet hours, pause and on/off preferences.
4. Open Appie. Give him a nickname and feed him. Food is spent; lifetime earned XP and unlocked looks remain. Headphones and a backpack unlock as XP grows.
5. Open Circle. Choose separate social consent and the individual details to share. Create a circle or join by code, send a preset reaction, and inspect the held support card and cancellation where applicable. A friend can join through `/join` without the health features.
6. Open the clinician portal. Inspect the cohort, patient trends, every logged entry, baseline measurements and model evidence. The entry inspector shows actual outputs and saved scoring decisions. Pages refresh every five seconds while visible and on return to the tab. Use **Demo workspace** to test real inference against a labelled synthetic reference; see [the clinician walkthrough](CLINICIAN.md).

## Shared rules

The canonical reward engine gives one qualifying daily memo reward, the prompt-window bonus, action and follow-up rewards, and gentle streak bonuses. A spoken memo needs ten seconds of speech; a written entry needs twenty words. Rewards depend on taking part, never the depression score. Custom habits earn two food each, capped at three per local day. Automatic wearable habits earn five food each.

Food is the canonical ledger balance and XP is total positive earnings. Feeding is a ledger spend with retry protection. It cannot lower levels. Older local Appie wallets and habits are imported into that ledger during startup without deleting the old tables or clinical rows. The migration is repeat-safe and preserves the existing food balance before normal historical reward reconciliation.

The fixed phrase, reading checks, separate phrase baselines, hour-of-day adjustment, weighted extra check-ins, clinical evidence and elevated flags remain from Kasper's engine. The phrase is never passed to the diary text model. Raw recordings are deleted after extraction. Optional context ratings retain their upstream default of disabled and zero scoring weight.

Circle responses follow the existing separate consent, pseudonym, individual sharing switches, token membership and support-card cancellation. Scores, diaries, transcripts, voice and Fitbit measurements are not included in the circle feed. Free-text chat is not implemented.

## Google Health and notifications

Habits has Google Health connect, sync and disconnect controls. Configure the OAuth client in the root `.env`; the default callback is `http://127.0.0.1:8000/api/google-health/callback`. A Web client must register that exact URL. The flow uses PKCE, expiring one-use state and an identity check before saving the patient-bound token in ignored `data/health_tokens/`.

Sync observes the latest completed Fitbit day at receipt time. It keeps the original measurement date and does not rewrite past scores or treat partial current-day steps as a complete day. Workout rewards use active duration, excluding pauses, with a step-cadence fallback. Active minutes earn food without adding an unvalidated scoring weight. Hobbies are tracked manually. The demo health button adds explicitly labelled generated data.

The existing Web Push service is retained. Set VAPID values with the team's setup before enabling notifications in the UI. The browser asks permission only after the person presses the button. A service worker handles delivery and opens the check-in flow. Reminder preferences use local time and support quiet hours, pausing and turning reminders off. No email, SMS or emergency-service integration is added.

## Run locally

Use the existing backend virtual environment and frontend dependencies. On Windows, `scripts/run_local.ps1` starts both services, records process IDs and logs in ignored `.runs/`. It accepts `-Python`, `-BackendPort`, `-FrontendPort` and `-NoWarmup`. Use `scripts/stop_local.ps1` for those recorded processes.

Or run `backend/.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000` from `backend`, then `npm.cmd run dev -- --host 127.0.0.1` from `frontend`. Open `http://localhost:5173/`.

If the screens update but Appie returns Not Found, an older backend process is still running. Restart both services from the same checkout. `/api/health` now reports `api_version: appie-integration-v1` so that mismatch is visible.

The synthetic seed script creates four demo profiles, a circle, custom hobbies and earned food through the real API. It refuses to replace an existing database without explicit `--reset`. Never reset real patient data for a demo.

`scripts/seed_demo.py` retains Kasper's prompt-aware demo. `scripts/seed_health_demo.py` additionally supplies the wearable history, health branch, custom hobbies and Appie treats. Both use separate synthetic data when no local Fitbit history is supplied.

## Validation

Run `python -m pytest -q -p no:cacheprovider` from `backend`. The test configuration clears the Gemini key, disables warm-up and live sync, creates a temporary database, and aborts if that isolation changes. Run `node --test scripts/test-recording.mjs scripts/test-push.mjs` and `npm.cmd run build` from `frontend`.

A migration check on a read-only backup preserves all clinical rows and the old wallet, survives a repeated upgrade, and exercises the patient APIs. Browser automation is unavailable in this session, so live microphone capture, visual browser review, actual Google consent and device push delivery still need a manual pass. The app continues to use demo patient selection; production sign-in and role authorization are separate work.
