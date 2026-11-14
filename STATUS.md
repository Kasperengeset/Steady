# STATUS – Beyond the Clinic

> Levende dokument: hva som er bygget, hva vi har lært om modellen, hvilke valg vi har tatt og hvorfor, og hva som gjenstår.
> **Oppdater denne når noe endres.** Detaljert opprinnelig plan: [`docs/plan.md`](docs/plan.md). Siste oppdatering: lør 14.11.2026.

## Kjernebudskap

*Vi diagnostiserer ikke. Vi starter når pasienten er frisk etter TMS, og oppdager når hen glir tilbake – målt mot sin egen friske normal, ikke mot et populasjonssnitt.*

---

## 1. Arkitektur – tre grener, én motor

| Gren | Når | Datasett for vekter | Status |
|---|---|---|---|
| **Akustikk** | brukeren *snakker* innsjekken | Androids Corpus | ✅ trent, i bruk |
| **Tekst** | brukeren *skriver* dagbok | eget tekstdatasett (teamet) | ⏳ plassholder – **tilfeldige tall** |
| **Helse** | daglig, automatisk | Fitbit / Google Health (kandidat: Depresjon-datasettet) | ⏳ ikke startet |

Per innsjekk:
1. Trekk ut tall (features) per gren.
2. **z-score mot pasientens egen baseline** (siste 14 dager): `z = retning × (i dag − snitt) / sd`.
3. Delscore per gren: `D_gren = Σ vekt × z`.
4. Dagsscore `D_t` = snitt av grenene som har data i dag.
5. Glatting: `S_t = 0,3·D_t + 0,7·S_(t−1)`.
6. Eskalering: venner ved S > 1 i 3 dager · behandler ved S > 2 · umiddelbart ved krise-flagg.
7. Tiltak velges med Thompson-sampling (bandit) og lærer av tilbakemelding om kvelden.

Vekter per gren ligger i `backend/app/engine/weights/<gren>.json`. Mangler fila, brukes litteraturens retning og like vekter.

---

## 2. Akustikk-modellen

### Trekk (8)
| Trekk | Kilde | Metode | Hva det måler |
|---|---|---|---|
| `speech_rate` | openSMILE eGeMAPS | FFT-basert | stemte segmenter/s (≈ stavelser/s) |
| `f0_std` | openSMILE | FFT (grunntone i halvtoner) | tonevariasjon (monotoni) |
| `pause_ratio` | openSMILE | stemt/ustemt | andel ustemt tid (pauser + ustemte konsonanter) |
| `energy` | openSMILE | FFT, hørselsbånd | lydstyrke |
| `jitter` | openSMILE | tidsdomene, periode for periode | ujevn tonehøyde |
| `shimmer` | openSMILE | tidsdomene | ujevn amplitude |
| `arousal` | wav2vec2 (audEERING) | nevralt nett på rå lydbølge, **ingen FFT** | hvor energisk stemmen høres ut |
| `valence` | wav2vec2 | samme | hvor positiv stemmen høres ut |

openSMILE og wav2vec2 er **ferdige verktøy som ikke trenes**. Det eneste vi trener, er den logistiske regresjonen som vekter de 8 tallene.

### Trening
- **Data:** Androids Corpus, intervjuklipp. 866 klipp med merkelapp → median per person → **114 talere (62 deprimerte / 52 friske)**. 2 talere med ukjent tilstand er droppet.
- **Modell:** standardisering → L2-regularisert logistisk regresjon. C velges med indre kryssvalidering på **log-loss**.
- **Validering:** korpusets egen `fold-lists.csv` med 5 folds, der samme person aldri er i både trening og test.
- **Vekter:** `|β|` normalisert til sum 1. Retning = `sign(β)`.

### Resultater (AUC, snitt ± sd over 5 folds)
| Modell | AUC | Brukes i appen? |
|---|---|---|
| **Våre 8 trekk** | **0,77 ± 0,08** | ✅ |
| Alle 88 eGeMAPS | 0,88 ± 0,06 | referanse |
| wav2vec2-representasjon (1024 → PCA 20) | 0,91 ± 0,06 | referanse |
| Bare kjønn + alder | 0,48 | sjekk for juksesignal ✅ |

Første modellversjon (C valgt etter AUC):
- bare openSMILE-trekkene: 0,76
- bare wav2vec2-trekkene: 0,73
- begge: 0,76 (de overlapper mye)

### Gjeldende vekter (`acoustic.json`)
`arousal` 0,18 · `jitter` 0,18 · `valence` 0,17 · `energy` 0,14 · `f0_std` 0,13 · `shimmer` 0,10 · `speech_rate` 0,07 · `pause_ratio` 0,04

Alle retninger stemmer med litteraturen. `pause_ratio` snur fortegn i den samlede modellen (kollinearitet, se beslutning B6), så der beholdes litteraturens retning.

### Kjente svakheter og sjekker
- **Mellom- vs. innen-person:** klassifikatoren er trent på forskjeller *mellom* personer, men brukes til endring *innen* én person. Vektene er en prior, og baselinen er kjernen.
- **Opptaksforhold:** deprimerte har lengre klipp (median 18,6 s mot 10,5 s) og ~2 dB svakere lyd.
  - Lengde alene gir AUC 0,64, men hjelper nesten ikke når den legges til trekkene (0,76 → 0,77). Trekkene er altså ikke bare et mål på lengde.
  - Lydstyrke alene gir 0,68. Uten `energy` faller AUC bare til 0,74.
- **Mikrofon:** `energy` avhenger av mikrofon og avstand. Telefonopptak ligger rundt 98-persentilen i korpuset. Vurder lavere vekt i appen.
- **wav2vec2-referansen (0,91)** kan delvis fange mikrofon og rom. Bruk **0,77** som det ærlige tallet utad.
- **Språk:** korpuset er italiensk. Prosodi overføres rimelig mellom språk, og innholdet tas av tekst-grenen.
- **Lisens:** både Androids og audEERING-modellen er kun for ikke-kommersiell bruk.

---

## 3. Viktige konsepter (ting vi har diskutert)

### Vi måler endring, ikke diagnose
Appen sammenligner pasienten med seg selv. **Er noen allerede deprimert når vi begynner, blir det baselinen, og vi ser ingen avvik.** Det fungerer likevel for denne oppgaven:
- Pasientene starter etter **vellykket TMS**, altså i remisjon, og klinikken vet det fra depresjonsscorene under behandlingen.
- Baselinen er derfor «frisk normal», og vi leter etter **tilbakefall**.

### Svakhet: langsomt tilbakefall glir inn i baselinen ⚠️
Den glidende baselinen på 14 dager tilpasser seg et tilbakefall som kommer sakte over uker. Løsningen er planlagt (åpen oppgave T1):
- **fast «frisk baseline»** fra de første ~2 ukene etter behandling, som fanger langsom glidning
- **glidende baseline** for de siste 14 dagene, som fanger brå endring

### Hva gir absolutt nivå i stedet for bare endring?
- Tekst-grenen og krise-flagget (for eksempel «jeg føler meg håpløs»)
- Korpusmodellen (`ml/score_clip.py`), som er for svak til å stå alene (AUC 0,77, mikrofonavhengig)
- Et spørreskjema innimellom (for eksempel PHQ-9 annenhver uke) som klinisk anker

### Hvor mange klipp trengs?
Modellen er **allerede trent** (114 personer). Brukerens egne klipp brukes til å teste og til baseline, ikke til trening.
- **Test av modellen:** 2–5 klipp i ulike tilstander (engasjert vs. trøtt og flat), med samme mikrofon og avstand, 30–60 s fritt snakk. Sjekk at trekkene beveger seg i riktig retning (`ml/score_clip.py`).
- **Personlig baseline:** koden krever minst 3, men det er statistisk tynt (et SD-estimat fra 3 målinger er svært usikkert). **~7 er brukbart, ~14 (to uker) er stabilt.** Det passer med perioden for «frisk baseline» etter TMS.
- **Morgen og kveld** bør ha hver sin baseline, fordi stemmen er systematisk forskjellig gjennom døgnet (T10).
- **Bedre modell = flere *personer*, ikke flere klipp.** Mange klipp fra én person uten merkelapp hjelper ikke klassifikatoren. Personlige vekter ville krevd måneder med data sammen med depresjonsscore.
- **Demo:** to uker finnes ikke. Bruk noen opptak, populasjonsnormer (T5) og simulatoren.

### Tale vs. tekst
Tale går til akustikk-grenen, skrevet dagbok til tekst-grenen. Gemini brukes i tekst- og innholdsdelen.

---

## 4. Beslutningslogg

| # | Beslutning | Hvorfor |
|---|---|---|
| B1 | Tre grener med hver sin modell og sitt datasett | Ulike signaler og datasett; enklere å utvikle parallelt |
| B2 | **Ingen korpuslyd sendes til Gemini eller andre API-er** | Androids-lisensen forbyr videredistribusjon |
| B3 | Google-krav oppfylles med Gemini i tekst-grenen, ikke Vertex for akustikk | M1 med MPS (Apple-GPU) er raskt nok (~0,1 s per 3 s lyd) |
| B4 | 8 tolkbare trekk, ikke 88 eller wav2vec2-representasjonen | Kan forklares for behandler; mindre fare for å lære mikrofon og rom med 114 personer |
| B5 | C velges etter log-loss, ikke AUC | AUC bryr seg ikke om skala, så C havnet på 0,001 og alle sannsynligheter ble ~0,5 |
| B6 | Trekk der fortegnet snur mot litteraturen, beholder litteraturens retning | Snuingen skyldes kollinearitet; for endring innen person vil vi ha den marginale retningen |
| B7 | Nettleseren konverterer opptak til 16 kHz mono WAV | Backend trenger ikke ffmpeg |
| B8 | Rå lyd slettes rett etter uttrekk | Personvern |
| B9 | Samme uttrekkskode i trening (`ml/`) og app (`backend/app/pipeline/`) | Vektene blir gyldige for appens trekk |

---

## 5. Åpne oppgaver

- [ ] **T1** Fast «frisk baseline» i tillegg til den glidende (`engine/state.py`, `routers/checkins.py`)
- [ ] **T2** Ved talt innsjekk skal plassholderen for Gemini **ikke** legge tilfeldige tekst-trekk inn i scoren, bare transkripsjon og krise-flagg (`pipeline/__init__.py`)
- [ ] **T3** Tekst-grenen: trene på teamets tekstdatasett → `weights/text.json`
- [ ] **T4** Helse-grenen: Fitbit/Google Health-adapter, vekter fra Depresjon-datasettet (aktivitet minutt for minutt, MADRS)
- [ ] **T5** Populasjonsnormer den første uka, før baselinen er bygget (`state.zscores`)
- [ ] **T6** Vurder lavere vekt på `energy` på grunn av mikrofonavhengighet
- [ ] **T7** Kjør leseoppgaven som ekstra kontroll (`--task reading --no-write`)
- [ ] **T8** Belønningen til banditen: «ble tilstanden bedre?» (`S_{t+1} < S_t`) er ikke koblet inn (`routers/interventions.py`)
- [ ] **T9** Test med Kaspers egne opptak (`ml/score_clip.py`)
- [ ] **T10** Baseline: høyere minimum (~5–7 innsjekker før z-score) og egen baseline for morgen og kveld (`state.zscores`, `_baseline_history`)
- [ ] Simulator for demo-slider (`sim/simulator.py` er en plassholder)

---

## 6. Kjøre

```bash
# app
cd backend && .venv/bin/uvicorn app.main:app --reload     # :8000/docs
cd frontend && npm run dev                                # :5173

# akustikk-modellen
bash ml/download_androids.sh                              # 3,7 GB → data/androids/ (godtar lisensen)
backend/.venv/bin/python ml/extract_features.py           # ~15 min på M1, kan avbrytes og fortsette
backend/.venv/bin/python ml/train_weights.py              # → weights/acoustic.json + ml/reports/acoustic/
backend/.venv/bin/python ml/score_clip.py <fil.m4a>       # test et eget opptak mot korpuset
```

| Hva | Hvor |
|---|---|
| Rapporter og figurer | `ml/reports/acoustic/` (`roc.png`, `coefficients.png`, `summary.md`) |
| Data | `data/` er gitignored og må **aldri** committes |
