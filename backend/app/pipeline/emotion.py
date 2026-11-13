"""audEERING wav2vec2 (arousal/valence/dominance). STUB."""

import random


def extract(audio_path: str) -> dict[str, float]:
    # TODO(B, lør): audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    return {
        "arousal": random.uniform(0.3, 0.7),
        "valence": random.uniform(0.3, 0.7),
    }
