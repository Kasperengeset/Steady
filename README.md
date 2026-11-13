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

Raw audio is deleted right after feature extraction. Full plan: [`docs/plan.md`](docs/plan.md).

## Run locally

```bash
# backend (Python 3.13)
cd backend
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload        # http://localhost:8000/docs

# frontend (Node 22) – in another terminal
cd frontend
npm install && npm run dev                     # http://localhost:5173
```

Copy `.env.example` to `.env` and add `GEMINI_API_KEY` when the real pipeline lands.

## Repo layout

| Path | What |
|---|---|
| `backend/app/pipeline/` | feature extractors (currently stubs) |
| `backend/app/engine/` | state engine, bandit, escalation |
| `backend/app/routers/` | API – contract in `schemas.py`, mirrored in `frontend/src/lib/api.ts` |
| `frontend/src/views/` | Patient / Clinician / Friends |
| `ml/` | weight training on the Androids Corpus (data not in repo) |

## License

MIT – except the Androids Corpus, which is not included and has its own non-commercial licence.
