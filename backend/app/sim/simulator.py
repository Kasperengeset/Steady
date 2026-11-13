"""Synthetic 60-day persona for the demo slider – docs/plan.md §7. STUB."""

import math
import random


def simulate(days: int = 60, seed: int = 0) -> list[dict]:
    # TODO(B, lør): AR(1) latent mood, relapse ~day 40, hidden arm effects, bandit run
    rng = random.Random(seed)
    out = []
    for t in range(days):
        relapse = max(0.0, (t - 40) / 10)
        S = 0.3 * math.sin(t / 5) + relapse + rng.gauss(0, 0.2)
        out.append({"day": t, "S": round(S, 3), "self_mood": round(7 - relapse * 1.5 + rng.gauss(0, 0.5), 2)})
    return out
