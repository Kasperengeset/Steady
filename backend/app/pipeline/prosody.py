"""Prosody via Parselmouth/openSMILE (eGeMAPS). STUB – returns plausible random values."""

import random


def extract(audio_path: str) -> dict[str, float]:
    # TODO(B, lør): parselmouth – speech rate, F0 std, pause ratio, energy
    return {
        "speech_rate": random.gauss(4.0, 0.4),  # syllables/s
        "f0_std": random.gauss(30.0, 5.0),  # Hz
        "pause_ratio": random.gauss(0.25, 0.05),
        "energy": random.gauss(60.0, 4.0),  # dB
    }
