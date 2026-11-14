"""Fixed reading phrase at the start of every check-in (issue #21, STATUS.md B22).

The person reads the same two sentences every day, then talks freely for 30 s. The words stay the
same, so a day-to-day change in the phrase is in the voice (rate, pauses, pitch, energy), not in what
was said. Speech science compares voices with standard reading passages for the same reason.

- The phrase is recorded as its own file, so we never have to find it inside the free talk.
- Its acoustic features are stored with a `phrase_` prefix: their own baseline, never mixed with free
  talk. They are not in engine/state.py DIRECTION, so they do not move the score until the phrase
  check (ml/phrase_check.py) shows they are steadier than free talk (#27).
- Reading check: with a Gemini key, the phrase is transcribed and compared with the known text (word
  error rate). Without one, only a timing check runs. A failed read skips the phrase features that day.
- The phrase transcript never reaches the text branch. Only the crisis phrase list runs on it.

Never change PHRASE: a new phrase means a new baseline for every `phrase_` feature. If it must change,
bump PHRASE_ID so old and new values can be told apart.
"""

import re

# Rainbow Passage (Fairbanks 1960), first two sentences – public domain, about 9 s read aloud
PHRASE_ID = "rainbow-v1"
PHRASE = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors."
)
# Counted by hand: when the sun·light strikes rain·drops in the air (10) | they act as a pri·sm and form
# a rain·bow (11) | the rain·bow is a di·vi·sion of white light in·to ma·ny beau·ti·ful co·lors (20)
PHRASE_SYLLABLES = 41

MAX_WER = 0.5  # issue #21: above this the read failed and the phrase features are skipped that day
# Plausible reading rate for adults: ~3–6 syllables/s; wide margins so slow or fast days still pass
MIN_SYLL_RATE, MAX_SYLL_RATE = 1.5, 8.0
MIN_SPAN_S = 2.0

FILLERS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "eh"}
SPELLING = {"colours": "colors", "colour": "color"}
# Transcribers sometimes split compounds: "sun light", "rain drops". Join them back before scoring.
JOINS = {("sun", "light"): "sunlight", ("rain", "drops"): "raindrops", ("rain", "bow"): "rainbow"}


def words(text: str) -> list[str]:
    """Lower-case words without punctuation or filler sounds ("um" is kept by verbatim transcription)."""
    raw = [SPELLING.get(w, w) for w in re.findall(r"[a-z']+", text.lower().replace("’", "'"))]
    raw = [w for w in raw if w not in FILLERS]
    out: list[str] = []
    for w in raw:
        if out and (out[-1], w) in JOINS:
            out[-1] = JOINS[(out[-1], w)]
        else:
            out.append(w)
    return out


REFERENCE = words(PHRASE)


def wer(hypothesis: str, reference: list[str] | None = None) -> float:
    """Word error rate: (substitutions + deletions + insertions) / words in the reference."""
    ref = REFERENCE if reference is None else reference
    hyp = words(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def reading_check(span_s: float, transcript: str | None) -> dict:
    """Did the person read the phrase? Returns the check stored with the check-in (llm_json["phrase"])."""
    rate = PHRASE_SYLLABLES / span_s if span_s > 0 else None
    check: dict = {
        "phrase_id": PHRASE_ID,
        "span_s": round(span_s, 2),
        "syllables_per_s": round(rate, 3) if rate else None,
        "wer": None,
        "checked_by": "timing",
    }
    if transcript is not None:
        check["wer"] = round(wer(transcript), 3)
        check["checked_by"] = "transcript"
    if span_s < MIN_SPAN_S or rate is None:
        reason = "too short or silent"
    elif not MIN_SYLL_RATE <= rate <= MAX_SYLL_RATE:
        reason = f"reading rate {rate:.1f} syllables/s is implausible"
    elif check["wer"] is not None and check["wer"] > MAX_WER:
        reason = f"words did not match the phrase (WER {check['wer']:.2f})"
    else:
        reason = None
    check["ok"] = reason is None
    check["reason"] = reason
    return check
