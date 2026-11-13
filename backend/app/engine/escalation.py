"""Escalation ladder – docs/plan.md §4.

0 normal · 1 friends (S>1 for 3 days) · 2 clinician (S>2, or level 1 for 7 days) · 3 crisis.
"""

CRISIS = 3


def evaluate_level(recent_S: list[float], crisis: bool) -> int:
    """recent_S: newest last."""
    if crisis:
        return CRISIS
    if recent_S and recent_S[-1] > 2:
        return 2
    if len(recent_S) >= 7 and all(s > 1 for s in recent_S[-7:]):
        return 2
    if len(recent_S) >= 3 and all(s > 1 for s in recent_S[-3:]):
        return 1
    return 0
