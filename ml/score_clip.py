"""Score your own audio clip(s) against the Androids-trained acoustic model.

  backend/.venv/bin/python ml/score_clip.py ~/Desktop/meg.m4a [more files ...]

Uses the same feature code as the app, refits the speaker-level logistic regression on
data/features/interview_clips.csv (seconds) and prints, per clip:
  - P(depressed-like voice) according to the between-person model
  - each feature's value, where it falls among corpus controls / patients, and its push on the score

NOT a diagnosis: the model is trained on 114 Italian speakers with different microphones, and in the
app it is only used to weight within-person *change*, never to classify a person.
"""

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from app.pipeline import audio, emotion, prosody  # noqa: E402
from corpus import FEATURES  # noqa: E402
from train_weights import APP_FEATURES, model, speaker_table  # noqa: E402


def to_wav(path: Path) -> Path:
    """soundfile reads wav/flac/ogg; anything else (m4a, mp3, aac …) is converted with macOS afconvert."""
    try:
        import soundfile as sf

        sf.info(str(path))
        return path
    except Exception:  # noqa: BLE001
        out = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(path), str(out)], check=True)
        return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spk = speaker_table(pd.read_csv(FEATURES / "interview_clips.csv"))
    cols = [f"app_{f}" for f in APP_FEATURES]
    X, y = spk[cols].to_numpy(), spk["label"].to_numpy()
    clf = model().fit(X, y)
    scaler, lr = clf[0], clf[-1]
    ctrl, pat = spk[y == 0], spk[y == 1]

    for arg in sys.argv[1:]:
        path = Path(arg).expanduser()
        if not path.is_file():
            print(f"\n{path}: file not found – drag the audio file into the terminal to paste its exact path")
            continue
        x = audio.load(str(to_wav(path)))
        feats = prosody.from_full(prosody.extract_full_signal(x))
        feats.update(emotion.run_signal(x)[1])
        v = np.array([[feats[f] for f in APP_FEATURES]])
        p = clf.predict_proba(v)[0, 1]
        push = scaler.transform(v)[0] * lr.coef_[0]

        print(f"\n{path.name}  ({len(x) / audio.SR:.1f} s)")
        print(f"  P(depressed-like voice) = {p:.2f}   (corpus base rate {y.mean():.2f})")
        print(f"  {'feature':12s} {'value':>9s}  {'pct ctrl':>8s} {'pct pat':>8s}  push")
        for i, f in sorted(enumerate(APP_FEATURES), key=lambda t: -abs(push[t[0]])):
            pc = (ctrl[f"app_{f}"] < v[0, i]).mean() * 100
            pp = (pat[f"app_{f}"] < v[0, i]).mean() * 100
            arrow = "▲ toward depressed" if push[i] > 0 else "▼ toward control"
            print(f"  {f:12s} {v[0, i]:9.3f}  {pc:7.0f}% {pp:7.0f}%  {arrow} ({push[i]:+.3f})")
        if len(x) < 8 * audio.SR:
            print("  ⚠️  Short clip – corpus clips are ~13 s median; 15–60 s of free speech gives steadier features.")
    print("\nNot a diagnosis: between-person model, Italian corpus, different microphones.")


if __name__ == "__main__":
    main()
