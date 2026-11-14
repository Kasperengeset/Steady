"""Health branch: one day of wearable data -> features for the engine (STATUS.md T4).

Same code in training (ml/health/train.py, Depresjon actigraphy) and in the app (Fitbit / Google
Health daily data, or the simulator), so validated features mean the same thing in both (B9).

Activity features come from minute-level activity: 1440 values, index 0 = 00:00. Depresjon gives
Actiwatch counts per minute, Fitbit gives steps per minute. The units differ, but the engine only
uses z-scores against the person's own baseline, so the scale cancels out.

  steps               total activity of the day                          lower = worse
  inactive_share      share of minutes with zero activity                higher = worse
  night_activity      mean activity 00:00-06:00 (restless nights)         higher = worse
  relative_amplitude  (M10 - L5) / (M10 + L5): how pronounced the daily   lower = worse
                      rhythm is (M10 = most active 10 h, L5 = least active 5 h)
  m10_onset           clock hour the most active 10 h start               later = worse

From the wearable's own summaries (no training data; directions from the literature, RADAR-MDD):

  sleep_hours         total sleep                                        lower = worse
  sleep_midpoint      hours after midnight of the middle of sleep        later = worse
  resting_hr          resting heart rate                                 higher = worse
  hrv_rmssd           heart-rate variability (RMSSD, ms)                 lower = worse
"""

from collections.abc import Sequence
from math import isfinite

MINUTES_PER_DAY = 1440
ACTIVITY_FEATURES = ("steps", "inactive_share", "night_activity", "relative_amplitude", "m10_onset")
SUMMARY_FEATURES = ("sleep_hours", "sleep_midpoint", "resting_hr", "hrv_rmssd")
FEATURES = ACTIVITY_FEATURES + SUMMARY_FEATURES

# +1 = higher is worse, -1 = higher is better (engine/state.py reads this for the health branch)
DIRECTION = {
    "steps": -1, "inactive_share": +1, "night_activity": +1, "relative_amplitude": -1, "m10_onset": +1,
    "sleep_hours": -1, "sleep_midpoint": +1, "resting_hr": +1, "hrv_rmssd": -1,
}


def _window_means(hourly: list[float], width: int) -> list[float]:
    """Mean of each run of `width` consecutive hours within the day (no wrap past midnight)."""
    return [sum(hourly[s : s + width]) / width for s in range(24 - width + 1)]


def daily_activity_features(minutes: Sequence[float]) -> dict[str, float]:
    """Five activity features from one full day of minute-level activity (1440 values from 00:00)."""
    if len(minutes) != MINUTES_PER_DAY:
        raise ValueError(f"need {MINUTES_PER_DAY} minute values (00:00 to 23:59), got {len(minutes)}")
    x = [float(v) for v in minutes]
    if any(not isfinite(v) or v < 0 for v in x):
        raise ValueError("minute steps must be finite and nonnegative")
    hourly = [sum(x[h * 60 : (h + 1) * 60]) / 60 for h in range(24)]
    m10 = _window_means(hourly, 10)
    l5 = min(_window_means(hourly, 5))
    best = max(range(len(m10)), key=m10.__getitem__)
    ra = (m10[best] - l5) / (m10[best] + l5) if m10[best] + l5 > 0 else 0.0
    return {
        "steps": sum(x),
        "inactive_share": sum(v == 0 for v in x) / MINUTES_PER_DAY,
        "night_activity": sum(x[: 6 * 60]) / (6 * 60),
        "relative_amplitude": ra,
        "m10_onset": float(best),
    }


def wearable_features(
    minute_steps: Sequence[float] | None = None,
    steps: float | None = None,
    sleep_hours: float | None = None,
    sleep_midpoint: float | None = None,
    resting_hr: float | None = None,
    hrv_rmssd: float | None = None,
    active_minutes: float | None = None,
) -> dict[str, float]:
    """Engine features for one day. Minute-level steps give all five activity features; with only a
    daily total, `steps` alone. Anything missing is simply left out (the engine scores what it has)."""
    out: dict[str, float] = {}
    if minute_steps is not None:
        out.update(daily_activity_features(minute_steps))
    elif steps is not None:
        out["steps"] = float(steps)
    for name, v in (("sleep_hours", sleep_hours), ("sleep_midpoint", sleep_midpoint),
                    ("resting_hr", resting_hr), ("hrv_rmssd", hrv_rmssd),
                    ("active_minutes", active_minutes)):
        if v is not None:
            out[name] = float(v)
    # Training and sync call this function directly as well as through Pydantic.
    for name, value in out.items():
        if not isfinite(value) or (value < 0 and name != "sleep_midpoint"):
            raise ValueError(f"{name} must be finite and nonnegative")
    if out.get("sleep_hours", 0) > 24 or abs(out.get("sleep_midpoint", 0)) > 24:
        raise ValueError("sleep hours must be 0 to 24 and midpoint must be -24 to 24")
    if "resting_hr" in out and out["resting_hr"] <= 0:
        raise ValueError("resting heart rate must be positive")
    if out.get("active_minutes", 0) > 1440:
        raise ValueError("active minutes must be between 0 and 1440")
    # active_minutes is stored only for engagement. It intentionally has no
    # direction or trained weight in the health score.
    return out
