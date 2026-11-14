"""Prosody via openSMILE eGeMAPSv02 functionals (Eyben et al. 2016; Cummins et al. 2015).

extract()      -> the ~6 interpretable features the state engine uses
extract_full() -> all 88 eGeMAPS functionals (reference model in ml/train_weights.py)
"""

import threading
from functools import lru_cache

import numpy as np

from . import audio

# app feature name -> eGeMAPS column
MAP = {
    "speech_rate": "VoicedSegmentsPerSec",
    "f0_std": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",  # semitones: less gender-dependent than Hz
    "energy": "loudness_sma3_amean",
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
}


_LOCK = threading.Lock()


def _smile():
    """Load once, thread-safe: the startup warm-up thread and the first request may call this at the
    same time, and concurrent first imports of transformers fail (ImportError)."""
    with _LOCK:
        return _load_smile()


@lru_cache(maxsize=1)
def _load_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_full_signal(x: np.ndarray) -> dict[str, float]:
    df = _smile().process_signal(x, audio.SR)
    return {k: float(v) for k, v in df.iloc[0].items()}


def from_full(full: dict[str, float]) -> dict[str, float]:
    out = {name: full[col] for name, col in MAP.items()}
    voiced, unvoiced = full["MeanVoicedSegmentLengthSec"], full["MeanUnvoicedSegmentLength"]
    out["pause_ratio"] = unvoiced / (voiced + unvoiced) if voiced + unvoiced > 0 else 0.0
    return out


def extract_full(audio_path: str) -> dict[str, float]:
    return extract_full_signal(audio.load(audio_path))


def extract(audio_path: str) -> dict[str, float]:
    return from_full(extract_full(audio_path))
