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
3. Delscore per gren: `D_gren = Σ vekt × z / √(Σ vekt²)` (SD 1 når ingenting endres, se B10).
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

### Hva modellen er – og ikke er
**Ett enkelt klipp gir nesten ingenting brukbart alene.**
- **I appen** gir ett klipp bare 8 trekk-verdier. Score (`S`) kommer først når det finnes nok tidligere innsjekker å sammenligne med.
- **`ml/score_clip.py`** gir `P(depresjonslignende stemme)` mot korpuset. AUC 0,77 betyr riktig rangering av et tilfeldig par deprimert/frisk 77 % av gangene, men for én person er feilmarginen stor. Den er trent på italiensk tale, er mikrofonavhengig og er **ingen diagnose**.

Treningen gir to ting, ingen av dem en prediksjon:
1. **Validering av måleinstrumentet:** trekkene bærer depresjonssignal på et klinisk korpus (AUC 0,77, alle retninger i tråd med litteraturen).
2. **Vekter:** hvilke trekk som teller mest når vi måler *endring*.

Det nyttige resultatet er **tidsserien S_t over dager**, ikke det enkelte klippet.

**Ikke vist:** at trente vekter oppdager tilbakefall bedre enn like vekter. Det krever data der samme personer følges gjennom et tilbakefall, og Androids har ett opptak per person.
**Pitch:** si «trekkene er validert på klinisk korpus», **ikke** «modellen oppdager depresjon med 77 % nøyaktighet».

### Vi måler endring, ikke diagnose
Appen sammenligner pasienten med seg selv. **Er noen allerede deprimert når vi begynner, blir det baselinen, og vi ser ingen avvik.** Det fungerer likevel for denne oppgaven:
- Pasientene starter etter **vellykket TMS**, altså i remisjon, og klinikken vet det fra depresjonsscorene under behandlingen.
- Baselinen er derfor «frisk normal», og vi leter etter **tilbakefall**.

### Simulatoren (`backend/app/sim/simulator.py`)
Kjører den **ekte motoren** (`state.py`, `bandit.py`, `escalation.py`, trent `acoustic.json`) på en syntetisk pasient i 70 dager, med én talt innsjekk per dag.
- Latent depresjon `d_t` i [0, 1] følger en AR(1)-prosess. Et langsomt tilbakefall starter dag 40 og bygger seg opp over ~15 dager.
- Trekk = personlig forskyvning + `d_t` × (pasientsnitt − kontrollsnitt fra korpuset) + støy. Effektstørrelsene er realistiske, fra Androids (Cohens d 0,2–0,9), og lagres som `feature_stats` i `acoustic.json`.
- Banditen velger tiltak. Hvert tiltak har en skjult «sann» nytte.
- API: `POST /api/sim/run` gir én pasient. `GET /api/sim/robustness?seeds=50` gir tall over mange pasienter, og **det er de tallene som skal brukes utad, ikke ett heldig frø.**

**Resultater over 50 syntetiske pasienter:**

| | Glidende baseline (dagens app) | Fast «frisk baseline» (T1) |
|---|---|---|
| Tilbakefall som ikke oppdages | **21/50** | **0/50** |
| Falske alarmer før tilbakefall (snitt, dager) | 1,56 | 0,38 |
| Oppdaget før selvrapport (median) | ~1 dag (bare de oppdagede) | ~0,5 dag |
| Banditen finner beste tiltak | 19/50 (tilfeldig ≈ 8/50) | |

Tolkning:
- **Den faste baselinen er nødvendig.** Den glidende «spiser» tilbakefallet. I frø 0 stiger S til 1,35 og synker til −0,2 mens pasienten fortsatt er syk.
- **Vi oppdager ikke tilbakefall tydelig tidligere enn selvrapport**, selv med en antatt forsinkelse på 4 dager i selvrapporten. Gevinsten er at vi oppdager det *passivt og pålitelig* (0/50 bommer) uten at pasienten må fylle ut skjema. Ikke påstå tidlig varsling i pitchen uten å ha vist det.
- **Banditen lærer svakt på 70 dager.** Belønningen er maks 0,5, fordi «ble tilstanden bedre?» ikke er koblet inn (T8).
- **Antakelser** (følger med i API-svaret): dag-til-dag-støy = 0,6 × variasjonen mellom personer, selvrapport 4 dager forsinket, og uavhengig støy i trekkene (ekte trekk henger sammen, så signalet er noe overvurdert).

### Svakhet: langsomt tilbakefall glir inn i baselinen ⚠️ (bekreftet i simulatoren)
Den glidende baselinen på 14 dager tilpasser seg et tilbakefall som kommer sakte over uker. **Simulatoren viser at 21/50 tilbakefall går tapt.** Løsningen (T1, høyeste prioritet) er:
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

### Vekter for endring ≠ vekter for klassifisering (målt 14.11)
Klassifikatoren lærer å skille *personer*. Appen skal oppdage *endring innen person*. Da bør et trekk telle etter hvor mye det endrer seg ved depresjon, sammenlignet med hvor mye det svinger hos samme person: **d innen person = (pasientsnitt − kontrollsnitt) / SD innen person**. Androids har ~5 klipp per person, så dette kan måles:

| Trekk | d innen person | ICC (stabilitet hos samme person) | Vekt i dag (klassifikator) |
|---|---|---|---|
| `energy` | **2,13** | 0,89 | 0,14 |
| `arousal` | **1,88** | 0,85 | 0,18 |
| `valence` | 1,15 | 0,63 | 0,17 |
| `jitter` | 0,71 | 0,68 | 0,18 |
| `pause_ratio` | 0,56 | 0,66 | 0,04 |
| `speech_rate` | 0,39 | 0,58 | 0,07 |
| `shimmer` | 0,36 | 0,74 | 0,10 |
| `f0_std` | 0,30 | 0,64 | 0,13 |

- Medianforholdet mellom variasjon innen og mellom personer er **0,70** (simulatoren antar 0,6).
- **Forbehold:** klippene er fra samme økt og samme mikrofon, så variasjonen fra dag til dag er større, særlig for `energy`, som avhenger av avstanden til telefonen. Tallene for `energy` og `arousal` er derfor optimistiske.

### Røyktest på egne opptak (14.11, 8 klipp: Kasper og Mahdan, 2 bra og 2 dårlige hver)
Samme DJI-mikrofon og samme dag. Lyd og trekk fra opptakene er **ikke** lagt i repoet.

**Mot korpuset** (`score_clip.py`, P(depresjonslignende)):
| | Bra 1 | Bra 2 | Dårlig 1 | Dårlig 2 |
|---|---|---|---|---|
| Kasper | 0,31 | 0,14 | 0,33 | 0,34 |
| Mahdan | 0,29 | 0,18 | 0,55 | 0,59 |

Mahdan skilles, Kasper ikke.

**Mot egen baseline** (snitt av de 2 bra klippene, SD innen person fra korpuset). Bra klipp er målt mot det andre bra klippet:
| | Bra (normal svingning) | Dårlig 1 / 2, dagens vekter | Dårlig 1 / 2, vekter for endring (T13) |
|---|---|---|---|
| Kasper | ±2,28 (dagens) / ±1,70 (T13) | 1,85 / 1,91 | **3,34 / 2,96** |
| Mahdan | ±1,29 / ±0,84 | 3,71 / 4,30 | **4,27 / 4,77** |

Funn:
- **Med dagens vekter skilles ikke Kaspers dårlige klipp fra hans normale svingning**, fordi jitter svinger mye og har stor vekt.
- **Med T13-vekter skilles begge**: de dårlige klippene ligger ~1,9× (Kasper) og ~5,4× (Mahdan) lenger ute enn den normale svingningen.
- **Arousal og valens (wav2vec2) bærer signalet** hos begge.
- Det bekrefter kjernebudskapet: korpusmodellen bommer på Kasper, men egen baseline fanger ham.

Forbehold: n = 2 personer; de dårlige klippene er muligens spilt og dermed overdrevne; samme økt (variasjon over dager er større); SD hentet fra korpuset.

### Forutsetninger for at det skal virke
En god «frisk baseline» er **nødvendig, men ikke tilstrekkelig**.
1. **Baselinen trenger 1–2 uker**, ikke én dag (~7–14 innsjekker med hverdager, helg, morgen og kveld).
2. **Klinikken bekrefter utgangspunktet** (for eksempel MADRS ved avsluttet TMS), fordi remisjon ofte har restsymptomer.
3. **Stabile opptaksforhold.** Ny telefon, annen avstand eller et støyende rom ser ut som endring. Dette er den største praktiske risikoen (T14).
4. **Stemmen er ikke spesifikk for depresjon.** Forkjølelse, søvn, alkohol, stress og medisiner gir falske alarmer, så vi trenger flere signalkilder (T3, T4).
5. **Baselinen eldes.** Den bør fornyes jevnlig, men **bare når frisk tilstand er bekreftet** (for eksempel med PHQ-9). Ellers kommer frosken i gryta tilbake.
6. **Ikke alle viser depresjon i stemmen.** Effekten er moderat i snitt.
7. **Pasienten må sjekke inn regelmessig.**
8. **Simulatortallene bygger på antakelser.** Virkelig ytelse krever en pilot over tid.

### Forbedringsplan (prioritert)
1. **T1** fast «frisk baseline». Simulatoren: 21/50 → 0/50 bom.
2. **T13** vekter for endring ∝ d innen person (tabellen over). Klassifikatoren brukes bare til retning og validering, og simulatoren får målt støy i stedet for 0,6.
3. **T14** opptaksprotokoll og kvalitetssjekk:
   - samme spørsmål, minst 20 s, fast avstand
   - avvis opptak med lite tale, klipping eller lav signal/støy
   - lydstyrke relativt til bakgrunnsstøy
4. **T10** mer robust baseline: median/MAD, morgen og kveld hver for seg, minst ~7 innsjekker.
5. **T15** ta hensyn til samvariasjon mellom trekk (Mahalanobis), så samme signal ikke telles dobbelt og gir falske alarmer.
6. **T12** kalibrere grensene i simulatoren mot et akseptabelt antall falske alarmer per pasient-måned.
7. **T3/T4** tekst- og helsegrenen: uavhengige kilder gir den største reelle gevinsten.
8. Høyere AUC (flere eGeMAPS-trekk, kjønnsnormalisering, lese- og intervjuoppgaven samlet). Minst viktig for produktet.
9. **Etter hackathon:** pilot over tid med PHQ-9. Det er eneste måte å validere tidlig varsling og lære personlige vekter på.

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
| B10 | Delscore per gren skaleres: `D_b = Σ w·z / √(Σ w²)` | Med vekter som summerer til 1 var D et vektet snitt med SD ≈ 0,36, så eskaleringsgrensene (S > 1, S > 2) ble nesten aldri nådd. Nå er D ~ N(0, 1) når ingenting endrer seg |
| B11 | Simulatoren bruker den ekte motoren og effektstørrelser fra korpuset | Demoen viser det systemet faktisk gjør; robusthet måles over 50 frø |

---

## 5. Åpne oppgaver

- [ ] **T1 (høyest prioritet)** Fast «frisk baseline» i tillegg til den glidende (`engine/state.py`, `routers/checkins.py`). Simulatoren viser 21/50 bom uten den, 0/50 med.
- [ ] **T2** Ved talt innsjekk skal plassholderen for Gemini **ikke** legge tilfeldige tekst-trekk inn i scoren, bare transkripsjon og krise-flagg (`pipeline/__init__.py`)
- [ ] **T3** Tekst-grenen: trene på teamets tekstdatasett → `weights/text.json`
- [ ] **T4** Helse-grenen: Fitbit/Google Health-adapter, vekter fra Depresjon-datasettet (aktivitet minutt for minutt, MADRS)
- [ ] **T5** Populasjonsnormer den første uka, før baselinen er bygget (`state.zscores`)
- [ ] **T6** Vurder lavere vekt på `energy` på grunn av mikrofonavhengighet
- [ ] **T7** Kjør leseoppgaven som ekstra kontroll (`--task reading --no-write`)
- [ ] **T8** Belønningen til banditen: «ble tilstanden bedre?» (`S_{t+1} < S_t`) er ikke koblet inn (`routers/interventions.py`)
- [ ] **T9** Test med Kaspers egne opptak (`ml/score_clip.py`)
- [ ] **T10** Baseline: høyere minimum (~5–7 innsjekker før z-score) og egen baseline for morgen og kveld (`state.zscores`, `_baseline_history`)
- [x] Simulator med den ekte motoren (`sim/simulator.py`, `/api/sim/run`, `/api/sim/robustness`)
- [ ] **T11** Demo-visning i frontend: slider over dagene, S (glidende vs. fast) mot selvrapport, og hva banditen har lært (typer finnes i `frontend/src/lib/api.ts`)
- [ ] **T13** Vekter for endring ∝ d innen person, målt på Androids. Simulatoren skal bruke målt støy (se «Vekter for endring»)
- [ ] **T14** Opptaksprotokoll og kvalitetssjekk: lite tale, klipping, signal/støy; lydstyrke relativt til bakgrunnsstøy
- [ ] **T15** Ta hensyn til samvariasjon mellom trekk (Mahalanobis-avstand) for færre falske alarmer
- [ ] **T12** Kalibrere eskaleringsgrensene: avveining mellom hvor raskt vi oppdager og antall falske alarmer, over mange frø i simulatoren

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
