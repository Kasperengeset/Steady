"""Shared audio loading: mono float32 at 16 kHz (what both openSMILE and wav2vec2 expect).

The PWA uploads 16 kHz mono WAV (converted in the browser), so no ffmpeg is needed.
"""

import numpy as np
import soundfile as sf

SR = 16_000


def load(path: str) -> np.ndarray:
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError("Unsupported audio format. Upload WAV/FLAC (16 kHz mono preferred).") from e
    x = x.mean(axis=1)
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)
