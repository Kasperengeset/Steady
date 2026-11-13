# Beyond the Clinic – tentativ plan (Junction 2026)

**Utfordring:** Aalto University – hjelpe pasienter å holde seg friske hjemme etter depresjonsbehandling (TMS – transkraniell magnetstimulering).
**Lag:** 2 personer — **A** = frontend/produkt/pitch, **B** = backend/ML.
**Frister:** Pre-submission fre 13.11 kl. 23:59 · Endelig innlevering søn 15.11 kl. 10:00 · Peer review søn 10:30–14:00 (må være til stede).

---

## 1. Konsept

Daglig kort stemmesamtale med en AI (morgen + kveld, ~20 s) → måler tilstand mot pasientens egen baseline → velger dagens tiltak og lærer hva som virker → eskalerer til venner/behandler ved behov.

**Én motor, tre visninger:**

1. **Pasient** – innsjekk og dagens tiltak
2. **Behandler** – pasientliste sortert etter risiko, trend og *hvorfor*
3. **Venner/community** – diskret hint når pasienten kan trenge kontakt

---

## 2. Arkitektur

```
Mobil-PWA (pasient) ──lyd──▶ FastAPI-backend
     │                          ├─▶ Gemini (innhold, opplevd følelse, krise-flagg → JSON)
     │                          ├─▶ Parselmouth/openSMILE (prosodi-målinger)
     │                          └─▶ audEERING wav2vec2 (arousal/valens/dominans)
     │                          ▼
     │                    Tilstandsmotor (personlig baseline + vekter fra Androids)
     │                          ▼
     │                    Bandit (velger tiltak) ──▶ Eskaleringsstige
     ▼                          ▼                          ▼
 Dagens tiltak          Behandlerdashbord           Venne-varsel
```

| Del | Valg |
|---|---|
| Frontend | PWA (progressive web app) i React + Vite, opptak med `MediaRecorder` |
| Samtale | Gemini Live API (sanntids tale inn/ut) |
| Backend | Python, FastAPI, SQLite |
| Wearable | Adapter: simulerte data først, Google Health API (erstatter Fitbit Web API) hvis tid |
| Personvern | Rå lyd slettes etter trekkuttrekk; Vertex AI i EU-region |

---

## 3. Signaler per innsjekk

| Kilde | Trekk $x_i$ | Retning (høy = verre) |
|---|---|---|
| Prosodi (Parselmouth/eGeMAPS) | taletempo, F0-standardavvik (tonehøydevariasjon), pauseandel, energi | −, −, +, − |
| audEERING-modell | arousal, valens | −, − |
| Gemini (innhold) | humør 0–10, energi, anhedoni, sosial kontakt, `risk` | −, −, +, −, krise |
| Wearable | søvnlengde, nattlig HRV (RMSSD), hvilepuls, skritt | −, −, +, − |

**Prinsipp:** LLM (stor språkmodell) for *mening*, signalbehandling for *målinger*. Målinger må være reproduserbare for at baseline skal fungere.

---

## 4. Tilstandsmotor

For trekk $i$ på dag $t$, vindu $W = [t-14,\, t-1]$:

$$z_{i,t} = s_i \cdot \frac{x_{i,t} - \mu_{i,W}}{\sigma_{i,W}}, \quad s_i \in \{-1,+1\}$$

$$D_t = \sum_i w_i\, z_{i,t}, \qquad S_t = \alpha D_t + (1-\alpha) S_{t-1}, \quad \alpha \approx 0{,}3$$

- $S_t$ er EWMA (eksponentielt vektet glidende gjennomsnitt).
- Første uke: populasjonsnormer til baseline er bygget.
- Vekter $w_i$: fra logistisk regresjon på Androids Corpus (se §6); fallback = like vekter.

**Eskaleringsstige**

| Nivå | Utløser | Mottaker |
|---|---|---|
| 0 | Normalt | Pasient (dagens tiltak) |
| 1 | $S_t > 1$ i 3 dager | Valgte venner – hint, ikke data |
| 2 | $S_t > 2$, eller nivå 1 i 7 dager uten bedring | Behandler – flagg + topp-3 bidragende trekk |
| ! | `risk = true` | Behandler umiddelbart + krisekontakt vises |

---

## 5. Tiltaksloop (bandit)

- **Armer:** tur ute, ringe en venn, morgenlys, musikkøkt, pusteøvelse, liten aktivitetsoppgave (atferdsaktivering).
- **Morgen:** Thompson sampling — $\theta_k \sim \text{Beta}(a_k, b_k)$, velg $\arg\max_k \theta_k$.
- **Kveld:** belønning
  $$r_t = 0{,}5 \cdot \text{gjennomført} \cdot \text{vurdering} + 0{,}5 \cdot \mathbb{1}[S_{t+1} < S_t] \in [0,1]$$
- **Oppdatering:** $a_k \leftarrow a_k + r_t,\; b_k \leftarrow b_k + (1-r_t)$.
- **Hvis tid:** kontekstuell – egne posteriorer for «lav energi» vs. «normal».

---

## 6. Datagrunnlag for stemmeanalyse

1. **Litteratur:** Cummins et al. (2015), *Speech Communication* – begrunner valg av prosodi-trekk; eGeMAPS som standardsett.
2. **Forhåndstrent modell:** audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` (MSP-Podcast). Lisens CC-BY-NC-SA 4.0.
3. **Klinisk validering:** Androids Corpus (116 personer, 64 deprimerte / 52 kontroller, italiensk, direkte nedlasting 3,7 GB).
   - Trekk per person → standardiser → logistisk regresjon med regularisering
   - Kryssvalidering med medfølgende `fold-list.csv` (speaker-uavhengig)
   - Rapporter AUC (areal under ROC-kurven) → koeffisienter blir $w_i$
   - ⚠️ Kun ikke-kommersiell forskning, ingen videredistribusjon → **data skal ikke i repoet**

**Begrensninger å si høyt:** mellom-person-klassifiserer brukt på innen-person-endring (vekter = prior, baseline = kjerne) · lite datasett · italiensk (prosodi overføres, innhold tas av Gemini).

---

## 7. Simulator (demo-motor)

Syntetisk persona, 60 dager:

- Latent humør som AR(1)-prosess; tilbakefall starter ~dag 40
- Hver arm har skjult sann effekt for personaen
- Trekk = funksjon av latent humør + støy

**Skal vise:** (1) $S_t$ flagger før selvrapportert humør faller, (2) banditen konvergerer mot riktige armer. Slider for hurtigspoling.

---

## 8. Tidsplan

| Tid | A (frontend/pitch) | B (backend/ML) |
|---|---|---|
| **Fre 21:30–23:59** | Repo (offentlig, MIT), PWA-skjelett, opptaksknapp. **Pre-submission** | FastAPI-skjelett, datamodell, start nedlasting av Androids |
| *Søvn* | | |
| **Lør 08–12** | Pasientflyt: morgen/kveld, dagens tiltak | Trekkuttrekk + logistisk regresjon på Androids → $w_i$, AUC |
| **Lør 12** | **Integrasjonssjekk:** én innsjekk ende til ende | |
| **Lør 12–13** | Mentorprat (MJ Suh): signaler, datasett, feedback | |
| **Lør 13–18** | Behandlerdashbord + venne-visning | Gemini-kall, tilstandsmotor, bandit, eskalering, simulator → UI |
| **Lør 18–22** | Pitch deck (PDF) + demomanus | Google Health-adapter *hvis tid*, ellers polering |
| **Lør 22–24** | Video (maks 2:00). **Feature freeze** | |
| **Søn 08–09:30** | Buffer, sett prosjekt til **FINAL** | |
| **Søn 10:30–14** | Peer review ved bordet | |

---

## 9. Pitch

**Video (2:00):** problem (0:15) → persona (0:15) → live innsjekk + tiltak (0:30) → hurtigspoling: læring og tidlig flagg (0:25) → behandler- og vennevisning (0:20) → hvorfor smart sensing (0:15).

**Bord (30 s):** dommer snakker 20 s → ser egen score og tiltak → slider viser oppdaget tilbakefall.

**Kjernebudskap:** *LLM for mening, signalbehandling for målinger, personlig baseline for endring – validert på klinisk korpus.*

---

## 10. Kuttliste (i rekkefølge)

1. Ekte Google Health-kobling
2. Gemini Live (fall tilbake til opptak + TTS/tekst)
3. Kontekstuell bandit
4. Innlogging

## 11. Regler å huske

- All kode skrives under eventet
- Offentlig repo, MIT-lisens (men ikke Androids-data)
- Innlevering: navn, punchline, beskrivelse, teknologier, video ≤ 2:00, demo- og kildekodelenke, pitch deck (PDF)

## 12. Åpne spørsmål

- Endelig rollefordeling A/B
- Har vi en Fitbit/Pixel Watch tilgjengelig?
- Hvilken Gemini Live-modell er nyest i AI Studio, og støtter den språket vi demoer på?
- Har mentorene egne data eller bedre datasett?
