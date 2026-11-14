# Transcript check (14 Nov)

Does going through speech and transcription change the text score? `scripts/check_transcribe.py --diary`: each diary sanity entry is read aloud by `gemini-3.8-flash-tts`, transcribed by the app's own code (`gemini-3.5-transcribe`, verbatim, English, inline audio, `store=False`) and scored by the text model. Synthetic voices only, no Androids audio.

| Entry | Written | Spoken → transcribed | Word match |
|---|---|---|---|
| good_1 | 0.11 | 0.13 | 99% |
| good_2 | 0.13 | 0.13 | 100% |
| good_3 | 0.37 | 0.37 | 100% |
| good_4 | 0.30 | 0.24 | 99% |
| good_5 | 0.07 | 0.07 | 100% |
| low_1 | 0.92 | 0.92 | 100% |
| low_2 | 0.64 | 0.60 | 99% |
| low_3 | 0.58 | 0.55 | 98% |
| low_4 | 0.64 | 0.64 | 100% |
| low_5 | 0.29 | 0.27 | 98% |

Mean score shift 0.016, largest 0.058. No entry changed label.

**Limitation:** TTS is clean read speech. Real check-ins have fillers, restarts and pauses, which verbatim mode keeps, so the shift on real speech may be larger.

## Our own recordings (14 Nov)

The same 8 clips as the acoustic smoke test (STATUS.md): Kasper and Mahdan each read 2 good-day and 2 bad-day scripts, 24 to 39 s, 48 kHz WAV, DJI mic. Transcribed by the app's own code, scored by the text model. Audio and transcripts are not in the repo.

| Speaker | Good 1 | Good 2 | Bad 1 | Bad 2 |
|---|---|---|---|---|
| Kasper | 0.02 | 0.07 | 0.53 | 0.69 |
| Mahdan | 0.02 | 0.06 | 0.62 | 0.73 |

- All 8 on the right side of 0.5. Within each speaker, both bad days scored above both good days.
- Verbatim mode kept real disfluencies ("um", "yeah", "picking or biking, I mean", "the the blinds").
- The same script read by the two speakers differed by at most 0.09, against a good/bad gap of about 0.5, so transcription noise is small.

**Caveats.** Both speakers read the same scripts, so this tests transcription on real speech and the wording of the scripts, not the speakers' delivery (that is the acoustic branch). The good scripts drop "I" (0 to 3 first-person words per 100, bad 15 to 19), the same style gap as the diary set. Bad 1 ("I'm doing all right, I suppose") is the most masked script and only just crossed 0.5 for Kasper (0.53).
