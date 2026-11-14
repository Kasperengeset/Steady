# Beyond the Clinic – Junction 2026 (Aalto University challenge)

Transcranial magnetic stimulation (TMS) is used in hospitals and clinics to treat people with severe depression. The treatment usually requires multiple brief daily visits for brain stimulation. Even if the treatment is successful, there is a risk that depression returns. Therefore, we would like to have “maintenance” treatment at home.

**Our idea:** a ~20-second daily voice check-in (morning + evening) → measured against the patient's *own* baseline → picks today's intervention and learns what works → escalates to friends or the clinician when needed.

> LLM for meaning, signal processing for measurements, personal baseline for change.

One engine, three views: **Patient** (check-in + today's action) · **Clinician** (risk-sorted list + *why*) · **Friends** (a discreet nudge, never health data).

```
PWA (React) ──audio──▶ FastAPI
                         ├─ prosody (Parselmouth/eGeMAPS)
                         ├─ emotion (audEERING wav2vec2)
                         └─ content + crisis flag (Gemini)
                         ▼
                 state engine (z-scores vs 14-day baseline → EWMA)
                         ▼
          Thompson-sampling bandit ─▶ escalation ladder
```

Raw audio is deleted right after feature extraction. Full plan: [`docs/plan.md`](docs/plan.md). **Current status, model results and decisions: [`STATUS.md`](STATUS.md).**

## Run locally

The combined patient app keeps the sage-and-sand design and Appie artwork, with the shared backend's phrase checks, local reminders, push, social consent, rewards and scoring. See [Appie integration and testing](docs/APPIE.md). On Windows, `scripts/run_local.ps1` starts the frontend and backend together from the same checkout.

The [clinical workspace walkthrough](docs/CLINICIAN.md) covers the desktop at `/clinician`: cohort triage, longitudinal state, raw model measurements, exact saved entry analysis, personal baselines and held-out model evidence. **Demo workspace** adds a separate synthetic history without resetting patients. Newly submitted diaries and recordings run through the actual pipelines and remain labelled live.

```bash
# backend (Python 3.13)
cd backend
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload        # http://localhost:8000/docs

# frontend (Node 22) – in another terminal
cd frontend
npm install && npm run dev                     # http://localhost:5173
```

## Secrets and the Gemini key

Do these once per clone:

```bash
git config core.hooksPath scripts/git-hooks   # pre-commit hook: blocks .env, keys, data and audio
cp .env.example .env                          # then paste GEMINI_API_KEY into .env (gitignored)
backend/.venv/bin/python scripts/check_gemini.py   # tests the key without printing it
backend/.venv/bin/python scripts/check_transcribe.py my_recording.m4a   # transcribe your own voice + text score
backend/.venv/bin/python scripts/check_transcribe.py --diary            # TTS → transcribe → score the diary set
```

**Transcription** (`backend/app/pipeline/llm.py`): with a key in `.env`, every spoken check-in is transcribed by `gemini-3.5-transcribe` in verbatim mode (keeps "um" and repetitions), English, audio sent inline (no 48-hour Files API copy), `store=False`. The transcript feeds the text branch and the crisis phrase check. Without a key, or if the call fails, the check-in still works on the voice features alone.

- The key lives only in `.env`. Never in code, the frontend (Vite `VITE_*` variables are shipped to the browser), a commit, an issue or a chat. All Gemini calls go through the backend.
- Every Gemini call passes `store=False`: Gemini stores requests by default, and diaries are health data.
- Never send Androids Corpus audio to Gemini or any other API (licence, STATUS.md B2). Test with our own recordings.
- If a key ever reaches GitHub: delete it at aistudio.google.com/api-keys at once and create a new one, then clean the history. Deleting the file in a new commit is not enough. (The hackathon kills every key in the project if one leaks.)
- **Hackathon account** (goo.gle/hackathon-account): use the API key from the team's temporary project; each of us puts their own copy in their own `.env`. The account is deleted the day after the hackathon, so record the demo video before then. Without a key the app still runs (voice features + the local text model on written diaries); only transcription needs Gemini.
- Keep usage hackathon-sized: tests use a fake client, and the simulator must never call Gemini (50 patients × 70 days would be thousands of requests). 503 = model overloaded (transcription retries twice); 429 = spend cap, raise it at aistudio.google.com/spend.

## Repo layout

| Path | What |
|---|---|
| `backend/app/pipeline/` | feature extractors: prosody + emotion (acoustic), `text.py` (text branch), Gemini (stub) |
| `backend/app/engine/` | state engine, bandit, escalation; branch weights in `weights/` |
| `backend/app/routers/` | API – contract in `schemas.py`, mirrored in `frontend/src/lib/api.ts` |
| `frontend/src/views/` | Patient / Clinician / Friends |
| `ml/` | acoustic branch training on the Androids Corpus (data not in repo) |
| `ml/text/` | text branch training on the Reddit Mental Health Dataset – see [`ml/text/README.md`](ml/text/README.md) |
| `ml/download_data.py` | downloads Reddit (text) and Depresjon (health) into `data/` |

## License

MIT – except the Androids Corpus, which is not included and has its own non-commercial licence.
