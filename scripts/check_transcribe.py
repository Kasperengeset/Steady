"""Test Gemini transcription and the text branch on it. Needs GEMINI_API_KEY in .env.

    backend/.venv/bin/python scripts/check_transcribe.py my_recording.m4a    # your own voice
    backend/.venv/bin/python scripts/check_transcribe.py --diary             # transcript check

--diary reads the 10 diary sanity entries aloud with Gemini TTS, transcribes them with the
app's own code (pipeline/llm.py) and compares the text-model score of each transcript with the
written version (GitHub issue #3). Synthetic voices only – never Androids audio. Temporary audio
is deleted straight away. All calls use store=False.
"""

import argparse
import base64
import csv
import difflib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.pipeline import llm  # noqa: E402
from app.pipeline import text as tp  # noqa: E402

TTS_MODEL = "gemini-3.8-flash-tts"
DIARY = ROOT / "ml" / "text" / "diary_sanity.csv"


def score(t: str) -> str:
    r = tp.analyze_text(t)
    flag = "  CRISIS" if r["crisis"] else ""
    return f"{r['p_depressive']:.2f} ({r['label']}){flag}"


def tts_wav(text: str) -> Path:
    r = llm._client().interactions.create(model=TTS_MODEL, input=text, store=False)
    audio = r.output_audio
    if audio is None:
        raise RuntimeError("TTS returned no audio")
    fd, name = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(audio.data) if isinstance(audio.data, str) else audio.data)
    return Path(name)


def words(t: str) -> list[str]:
    return tp._WORD.findall(t.lower().replace("’", "'"))


def run_files(paths: list[str]) -> None:
    for p in paths:
        transcript = llm.transcribe(p)
        print(f"\n{p}\n  transcript: {transcript}\n  score: {score(transcript)}")


def run_diary() -> None:
    rows = [r for r in csv.DictReader(open(DIARY, encoding="utf-8")) if r["variant"] == "as written"]
    print(f"{'id':7} {'written':>8} {'spoken':>8} {'word match':>11}")
    shifts = []
    for r in rows:
        wav = tts_wav(r["text"])
        try:
            transcript = llm.transcribe(str(wav))
        finally:
            wav.unlink(missing_ok=True)
        pw = tp.analyze_text(r["text"])["p_depressive"]
        ps = tp.analyze_text(transcript)["p_depressive"]
        match = difflib.SequenceMatcher(None, words(r["text"]), words(transcript)).ratio()
        shifts.append(abs(ps - pw))
        same = (pw >= 0.5) == (ps >= 0.5)
        print(f"{r['id']:7} {pw:8.2f} {ps:8.2f} {match:10.0%}{'' if same else '  label flipped'}")
    print(f"\nmean score shift {sum(shifts) / len(shifts):.3f}, largest {max(shifts):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="audio files of your own voice (wav, m4a, mp3, webm, ...)")
    ap.add_argument("--diary", action="store_true", help="TTS → transcribe → score the 10 diary entries")
    args = ap.parse_args()
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is empty in .env – see README, 'Secrets and the Gemini key'.")
    if not args.files and not args.diary:
        ap.error("give an audio file or --diary")
    if args.files:
        run_files(args.files)
    if args.diary:
        run_diary()


if __name__ == "__main__":
    main()
