"""Thompson sampling over Beta posteriors – docs/plan.md §5."""

import random
from typing import Protocol


class ArmLike(Protocol):
    key: str
    a: float
    b: float


def choose_arm[T: ArmLike](arms: list[T], rng: random.Random | None = None) -> T:
    """Thompson sampling. Pass a seeded rng for reproducible simulations."""
    r = rng or random
    return max(arms, key=lambda arm: r.betavariate(arm.a, arm.b))


def reward(completed: bool, rating: int | None, state_improved: bool | None) -> float:
    """r = 0.5 * completed * rating_norm + 0.5 * 1[S_{t+1} < S_t]. Unknown improvement → 0."""
    rating_norm = ((rating or 1) - 1) / 4
    return 0.5 * float(completed) * rating_norm + 0.5 * float(bool(state_improved))


def update(arm: ArmLike, r: float) -> None:
    arm.a += r
    arm.b += 1 - r
