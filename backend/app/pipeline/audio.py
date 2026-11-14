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


def _frame_db(x: np.ndarray, sr: int, frame_s: float = 0.025, hop_s: float = 0.010) -> tuple[np.ndarray, int, int]:
    """Frame loudness in dB (25 ms frames, 10 ms hop)."""
    frame, hop = int(frame_s * sr), int(hop_s * sr)
    if len(x) < frame:
        return np.array([]), frame, hop
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10), frame, hop


def _active(db: np.ndarray, drop_db: float) -> np.ndarray:
    """Indices of frames within drop_db of the loud frames (95th percentile); none for silence."""
    if db.size == 0 or np.percentile(db, 95) < -60:
        return np.array([], dtype=int)
    return np.flatnonzero(db > np.percentile(db, 95) - drop_db)


def speech_span(x: np.ndarray, sr: int = SR, drop_db: float = 30.0) -> float:
    """Seconds from the first to the last frame with speech: trims silence around it, keeps the pauses
    inside it. For a fixed text (pipeline/phrase.py) this gives the reading rate directly."""
    db, frame, hop = _frame_db(x, sr)
    active = _active(db, drop_db)
    return float((active[-1] - active[0]) * hop + frame) / sr if active.size else 0.0


def active_seconds(x: np.ndarray, sr: int = SR, drop_db: float = 30.0) -> float:
    """Seconds with voice energy (pauses do not count). Used to check a memo really has speech."""
    db, _, hop = _frame_db(x, sr)
    return float(_active(db, drop_db).size * hop) / sr
