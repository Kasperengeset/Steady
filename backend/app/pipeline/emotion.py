"""Dimensional emotion with audEERING wav2vec2-large-robust-12-ft-emotion-msp-dim.

Model card: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
Licence CC-BY-NC-SA 4.0. Outputs arousal, dominance, valence in ~[0, 1].
"""

from functools import lru_cache

import numpy as np

from . import audio

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CHUNK_S = 30


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _model():
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    dev = _device()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = EmotionModel.from_pretrained(MODEL_ID).to(dev).eval()
    return processor, model, dev


def run_signal(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Returns (mean 1024-d embedding, {arousal, dominance, valence}) averaged over 30 s chunks."""
    import torch

    processor, model, dev = _model()
    x = x / (np.abs(x).max() + 1e-8)  # peak-normalise loudness
    n = CHUNK_S * audio.SR
    chunks = [x[i : i + n] for i in range(0, len(x), n) if len(x[i : i + n]) >= audio.SR // 2] or [x]
    embs, preds = [], []
    with torch.no_grad():
        for c in chunks:
            inp = processor(c, sampling_rate=audio.SR, return_tensors="pt").input_values.to(dev)
            h, y = model(inp)
            embs.append(h[0].cpu().numpy())
            preds.append(y[0].cpu().numpy())
    weights = np.array([len(c) for c in chunks], dtype=float)
    emb = np.average(np.stack(embs), axis=0, weights=weights)
    a, d, v = np.average(np.stack(preds), axis=0, weights=weights)
    return emb, {"arousal": float(a), "dominance": float(d), "valence": float(v)}


def extract(audio_path: str) -> dict[str, float]:
    _, scores = run_signal(audio.load(audio_path))
    return {"arousal": scores["arousal"], "valence": scores["valence"]}


def embed(audio_path: str) -> np.ndarray:
    return run_signal(audio.load(audio_path))[0]
