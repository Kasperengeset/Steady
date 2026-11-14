"""Score diary text from the command line with the app's text model.

    backend/.venv/bin/python ml/text/predict.py "I feel tired and nothing seems worth doing"
    backend/.venv/bin/python ml/text/predict.py          # type entries, empty line to quit

The score is the probability that the text reads like r/depression rather than everyday or
relationship forums. A language signal, not a diagnosis. In the app it is compared with the
person's own baseline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import text as tp  # noqa: E402


def show(text: str) -> None:
    r = tp.analyze_text(text, explain_top=5)
    f = r["features"]
    print(f"\nprediction: {r['label']}")
    print(f"  score {r['p_depressive']:.2f} (cut-off {tp.THRESHOLD}, 0 = everyday, 1 = depressive)"
          f"   log-odds {f['text_logit']:+.2f}")
    print(f"  absolutist words {f['absolutist']:.1f} per 100   first-person words {f['first_person']:.1f} per 100"
          f"   sentiment {f['sentiment']:+.2f} (-1 negative, +1 positive)")
    if r["words_up"]:
        print("  pushed up:   " + ", ".join(f"{w} ({c:+.2f})" for w, c in r["words_up"]))
    if r["words_down"]:
        print("  pushed down: " + ", ".join(f"{w} ({c:+.2f})" for w, c in r["words_down"]))
    if r["crisis"]:
        print(f"  CRISIS PHRASE: {r['crisis_hits']} -> clinician immediately")
    if r["short_input"]:
        print(f"  note: {r['n_words']} words. The model was trained on 50 to 100-word entries, "
              "so a short sentence gives a rough score.")


def main() -> None:
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Type a diary entry and press Enter. Empty line to quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        show(line)


if __name__ == "__main__":
    main()
