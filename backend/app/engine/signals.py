"""Plain-language names for every signal, and which small actions fit when a signal is off.

Used by the patient app (the reason under today's action), the action choice (only actions that fit
what is lacking go to the bandit) and the clinician's "Why is this changing?" panel.
"""

# feature -> (source, plain name, technical name, format)
# format: "int", "pct", "clock" (hours after midnight), "h" (hours), "bpm", "ms", "num"
SIGNALS: dict[str, tuple[str, str, str, str]] = {
    # health (Fitbit / Google Health)
    "steps": ("health", "Steps per day", "steps", "int"),
    "inactive_share": ("health", "Time inactive during the day", "share of minutes with zero steps", "pct"),
    "m10_onset": ("health", "Start of the most active hours", "M10 onset", "clock"),
    "night_activity": ("health", "Night-time activity", "mean activity 00:00-06:00", "num"),
    "relative_amplitude": ("health", "Day-night rhythm", "relative amplitude (M10 - L5)/(M10 + L5)", "num"),
    "sleep_hours": ("health", "Hours of sleep", "minutes asleep / 60", "h"),
    "sleep_midpoint": ("health", "Sleep timing (midpoint)", "midpoint of the main sleep", "clock"),
    "resting_hr": ("health", "Resting heart rate", "daily resting heart rate", "bpm"),
    "hrv_rmssd": ("health", "Heart-rate variability", "HRV (RMSSD)", "ms"),
    # voice (spoken check-ins)
    "energy": ("acoustic", "Energy (loudness)", "eGeMAPS loudness", "num"),
    "f0_std": ("acoustic", "Pitch variation", "F0 standard deviation (semitones)", "num"),
    "speech_rate": ("acoustic", "Speaking speed", "voiced segments per second", "num"),
    "pause_ratio": ("acoustic", "Pauses", "share of unvoiced time", "pct"),
    "jitter": ("acoustic", "Voice steadiness (pitch)", "jitter", "num"),
    "shimmer": ("acoustic", "Voice steadiness (loudness)", "shimmer", "num"),
    "arousal": ("acoustic", "Tone: calm to energetic", "wav2vec2 arousal", "num"),
    "valence": ("acoustic", "Tone: negative to positive", "wav2vec2 valence", "num"),
    # words (diary or transcript)
    "text_logit": ("text", "Language closer to low-mood writing", "TF-IDF + logistic regression log-odds", "num"),
    "absolutist": ("text", "All-or-nothing words", "absolutist words per 100", "num"),
    "first_person": ("text", "Self-focus (I, me, my)", "first-person singular per 100 words", "num"),
    "sentiment": ("text", "Overall tone of the text", "VADER compound", "num"),
}

SOURCE_LABEL = {"health": "Activity & sleep", "acoustic": "Voice", "text": "Words"}

# Short phrase when a signal moves in the worse direction (clinician notifications, summaries)
WORSE = {
    "steps": "steps down", "inactive_share": "less active", "m10_onset": "days start later",
    "night_activity": "restless nights", "relative_amplitude": "flatter day-night rhythm",
    "sleep_hours": "less sleep", "sleep_midpoint": "sleep later", "resting_hr": "resting heart rate up",
    "hrv_rmssd": "HRV down", "energy": "quieter voice", "f0_std": "flatter voice",
    "speech_rate": "slower speech", "pause_ratio": "more pauses", "jitter": "less steady voice",
    "shimmer": "less steady voice", "arousal": "lower energy in the voice", "valence": "more negative tone of voice",
    "text_logit": "heavier language", "absolutist": "more all-or-nothing words",
    "first_person": "more inward", "sentiment": "more negative words",
}

# What the patient reads under today's action when this signal is the main thing that is off
REASON = {
    "steps": "Your steps have been lower than usual.",
    "inactive_share": "You've been less active during the day than usual.",
    "m10_onset": "Your days have been starting later than usual.",
    "sleep_hours": "You've slept less than usual.",
    "sleep_midpoint": "Your sleep has shifted later than usual.",
    "resting_hr": "Your body seems more tired than usual.",
    "hrv_rmssd": "Your body seems more tired than usual.",
    "text_logit": "Your check-ins sound a bit heavier than usual.",
    "first_person": "Your check-ins have turned more inward.",
    "sentiment": "Your check-ins sound a bit heavier than usual.",
    "absolutist": "Your check-ins sound a bit heavier than usual.",
}
REASON_SOURCE = {"acoustic": "Your voice sounds a bit quieter than usual."}
NEUTRAL_REASON = "A small thing that has helped you before."

# Which actions fit when a signal is off (arm keys from seed.ARMS)
FITS = {
    "steps": ["walk_outside", "small_task"],
    "inactive_share": ["walk_outside", "small_task"],
    "m10_onset": ["morning_light", "walk_outside"],
    "sleep_hours": ["morning_light", "breathing"],
    "sleep_midpoint": ["morning_light", "walk_outside"],
    "resting_hr": ["breathing", "walk_outside"],
    "hrv_rmssd": ["breathing", "music"],
    "text_logit": ["call_friend", "small_task"],
    "first_person": ["call_friend", "walk_outside"],
    "sentiment": ["call_friend", "music"],
    "absolutist": ["call_friend", "small_task"],
}
FITS_SOURCE = {"acoustic": ["call_friend", "music", "walk_outside"]}


def source_of(feature: str) -> str | None:
    s = SIGNALS.get(feature)
    return s[0] if s else None


def main_off_signal(top: list[dict] | None) -> str | None:
    """The feature contributing most to 'worse than usual' (top_contributors output), if any."""
    for t in top or []:
        if t.get("contribution", 0) > 0.3:
            return t["feature"]
    return None


def reason_for(top: list[dict] | None) -> str:
    f = main_off_signal(top)
    if f is None:
        return NEUTRAL_REASON
    return REASON.get(f) or REASON_SOURCE.get(source_of(f) or "", NEUTRAL_REASON)


def fitting_arms(top: list[dict] | None, arm_keys: list[str]) -> list[str]:
    """Arm keys that fit what is lacking; all arms when nothing stands out."""
    f = main_off_signal(top)
    if f is None:
        return arm_keys
    fit = FITS.get(f) or FITS_SOURCE.get(source_of(f) or "", [])
    chosen = [k for k in arm_keys if k in fit]
    return chosen or arm_keys
