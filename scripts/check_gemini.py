"""Check that the Gemini API key in .env works. Never prints the key.

    backend/.venv/bin/python scripts/check_gemini.py          # Mac/Linux
    backend\\.venv\\Scripts\\python scripts\\check_gemini.py      # Windows

Sends one tiny prompt with store=False (Gemini stores requests by default; we opt out
everywhere because diaries are health data).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import settings  # noqa: E402  (reads <repo>/.env)

TEXT_MODEL = "gemini-4-hydrogen"
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"


def main() -> None:
    key = settings.gemini_api_key.strip()
    if not key:
        sys.exit("GEMINI_API_KEY is empty. Copy .env.example to .env in the repo root and paste the key there.")
    print(f"key found in .env ({len(key)} characters, not shown)")

    from google import genai

    client = genai.Client(api_key=key)
    try:
        r = client.interactions.create(model=TEXT_MODEL, input="Reply with exactly: OK", store=False)
    except Exception as e:  # noqa: BLE001 – show the reason, never the key
        sys.exit(f"call failed: {type(e).__name__}: {str(e).replace(key, '<key>')[:300]}")
    print(f"{TEXT_MODEL} replied: {(r.output_text or '').strip()!r}")

    try:
        # gemini-3.5-transcribe is only listed on v1beta (the transcription docs use v1beta too)
        beta = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
        names = {m.name.removeprefix("models/") for m in beta.models.list()}
        for m in (TEXT_MODEL, TRANSCRIBE_MODEL):
            print(f"{m}: {'available' if m in names else 'NOT in model list'}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not list models: {type(e).__name__})")


if __name__ == "__main__":
    main()
