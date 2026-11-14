# Beyond the Clinic – tentative plan (Junction 2026)

> **This is the original plan from Friday 13 Nov.** Several parts have since changed (three model branches, a fixed healthy baseline, equal weights, etc.). The current state, results and decisions are in [`STATUS.md`](../STATUS.md).

**Challenge:** Aalto University – help patients stay well at home after depression treatment (TMS – transcranial magnetic stimulation).
**Team:** 2 people — **A** = frontend/product/pitch, **B** = backend/ML.
**Deadlines:** Pre-submission Fri 13 Nov 23:59 · Final submission Sun 15 Nov 10:00 · Peer review Sun 10:30–14:00 (must be present).

---

## 1. Concept

A short daily voice conversation with an AI (morning + evening, ~20 s) → measures the state against the patient's own baseline → chooses today's intervention and learns what works → escalates to friends/clinician when needed.

**One engine, three views:**

1. **Patient** – check-in and today's intervention
2. **Clinician** – patient list sorted by risk, trend and *why*
3. **Friends/community** – a discreet hint when the patient may need contact

---

## 2. Architecture

```
Mobile PWA (patient) ──audio──▶ FastAPI backend
     │                            ├─▶ Gemini (content, perceived emotion, crisis flag → JSON)
     │                            ├─▶ Parselmouth/openSMILE (prosody measurements)
     │                            └─▶ audEERING wav2vec2 (arousal/valence/dominance)
     │                            ▼
     │                      State engine (personal baseline + weights from Androids)
     │                            ▼
     │                      Bandit (chooses intervention) ──▶ Escalation ladder
     ▼                            ▼                               ▼
 Today's intervention     Clinician dashboard              Friend notification
```

| Part | Choice |
|---|---|
| Frontend | PWA (progressive web app) in React + Vite, recording with `MediaRecorder` |
| Conversation | Gemini Live API (real-time speech in/out) |
| Backend | Python, FastAPI, SQLite |
| Wearable | Adapter: simulated data first, Google Health API (replaces the Fitbit Web API) if time |
| Privacy | Raw audio deleted after feature extraction; Vertex AI in an EU region |

---

## 3. Signals per check-in

| Source | Feature $x_i$ | Direction (high = worse) |
|---|---|---|
| Prosody (Parselmouth/eGeMAPS) | speech rate, F0 standard deviation (pitch variation), pause ratio, energy | −, −, +, − |
| audEERING model | arousal, valence | −, − |
| Gemini (content) | mood 0–10, energy, anhedonia, social contact, `risk` | −, −, +, −, crisis |
| Wearable | sleep duration, nightly HRV (RMSSD), resting heart rate, steps | −, −, +, − |

**Principle:** LLM (large language model) for *meaning*, signal processing for *measurements*. Measurements must be reproducible for the baseline to work.

---

## 4. State engine

For feature $i$ on day $t$, window $W = [t-14,\, t-1]$:

$$z_{i,t} = s_i \cdot \frac{x_{i,t} - \mu_{i,W}}{\sigma_{i,W}}, \quad s_i \in \{-1,+1\}$$

$$D_t = \sum_i w_i\, z_{i,t}, \qquad S_t = \alpha D_t + (1-\alpha) S_{t-1}, \quad \alpha \approx 0.3$$

- $S_t$ is an EWMA (exponentially weighted moving average).
- First week: population norms until the baseline is built.
- Weights $w_i$: from logistic regression on the Androids Corpus (see §6); fallback = equal weights.

**Escalation ladder**

| Level | Trigger | Recipient |
|---|---|---|
| 0 | Normal | Patient (today's intervention) |
| 1 | $S_t > 1$ for 3 days | Chosen friends – a hint, not data |
| 2 | $S_t > 2$, or level 1 for 7 days without improvement | Clinician – flag + top-3 contributing features |
| ! | `risk = true` | Clinician immediately + crisis contact shown |

---

## 5. Intervention loop (bandit)

- **Arms:** walk outside, call a friend, morning light, music session, breathing exercise, small activity task (behavioural activation).
- **Morning:** Thompson sampling — $\theta_k \sim \text{Beta}(a_k, b_k)$, choose $\arg\max_k \theta_k$.
- **Evening:** reward
  $$r_t = 0.5 \cdot \text{completed} \cdot \text{rating} + 0.5 \cdot \mathbb{1}[S_{t+1} < S_t] \in [0,1]$$
- **Update:** $a_k \leftarrow a_k + r_t,\; b_k \leftarrow b_k + (1-r_t)$.
- **If time:** contextual – separate posteriors for "low energy" vs "normal".

---

## 6. Data foundation for voice analysis

1. **Literature:** Cummins et al. (2015), *Speech Communication* – justifies the choice of prosody features; eGeMAPS as the standard set.
2. **Pre-trained model:** audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` (MSP-Podcast). Licence CC-BY-NC-SA 4.0.
3. **Clinical validation:** Androids Corpus (116 people, 64 depressed / 52 controls, Italian, direct download 3.7 GB).
   - Features per person → standardise → logistic regression with regularisation
   - Cross-validation with the included `fold-list.csv` (speaker-independent)
   - Report AUC (area under the ROC curve) → coefficients become $w_i$
   - ⚠️ Non-commercial research only, no redistribution → **the data must not go into the repo**

**Limitations to state openly:** a between-person classifier applied to within-person change (weights = prior, baseline = core) · small dataset · Italian (prosody transfers, content is handled by Gemini).

---

## 7. Simulator (demo engine)

Synthetic persona, 60 days:

- Latent mood as an AR(1) process; relapse starts ~day 40
- Each arm has a hidden true effect for the persona
- Features = function of latent mood + noise

**Should show:** (1) $S_t$ flags before self-reported mood drops, (2) the bandit converges to the right arms. Slider for fast-forward.

---

## 8. Timeline

| Time | A (frontend/pitch) | B (backend/ML) |
|---|---|---|
| **Fri 21:30–23:59** | Repo (public, MIT), PWA skeleton, record button. **Pre-submission** | FastAPI skeleton, data model, start Androids download |
| *Sleep* | | |
| **Sat 08–12** | Patient flow: morning/evening, today's intervention | Feature extraction + logistic regression on Androids → $w_i$, AUC |
| **Sat 12** | **Integration check:** one check-in end to end | |
| **Sat 12–13** | Mentor chat (MJ Suh): signals, datasets, feedback | |
| **Sat 13–18** | Clinician dashboard + friend view | Gemini call, state engine, bandit, escalation, simulator → UI |
| **Sat 18–22** | Pitch deck (PDF) + demo script | Google Health adapter *if time*, otherwise polish |
| **Sat 22–24** | Video (max 2:00). **Feature freeze** | |
| **Sun 08–09:30** | Buffer, set project to **FINAL** | |
| **Sun 10:30–14** | Peer review at the table | |

---

## 9. Pitch

**Video (2:00):** problem (0:15) → persona (0:15) → live check-in + intervention (0:30) → fast-forward: learning and early flag (0:25) → clinician and friend view (0:20) → why smart sensing (0:15).

**Table (30 s):** a judge talks for 20 s → sees their own score and intervention → the slider shows a detected relapse.

**Core message:** *LLM for meaning, signal processing for measurements, personal baseline for change – validated on a clinical corpus.*

---

## 10. Cut list (in order)

1. Real Google Health connection
2. Gemini Live (fall back to recording + TTS/text)
3. Contextual bandit
4. Login

## 11. Rules to remember

- All code is written during the event
- Public repo, MIT licence (but not the Androids data)
- Submission: name, punchline, description, technologies, video ≤ 2:00, demo and source-code link, pitch deck (PDF)

## 12. Open questions

- Final A/B role split
- Do we have a Fitbit/Pixel Watch available?
- Which Gemini Live model is newest in AI Studio, and does it support the language we demo in?
- Do the mentors have their own data or better datasets?
