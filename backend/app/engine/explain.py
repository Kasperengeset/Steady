"""Explanation of the smoothed score S (STATUS.md B20): exact split over days and features.

S follows S_t = α·D_t + (1 − α)·S_(t−1), starting with S = D on the first scored day. Unrolled:

    S_t = Σ_j  weight_j · D_j,   weight_newest = α,  then α(1−α), α(1−α)², …  (oldest day: (1−α)^m)

and every day's D is itself a sum of feature shares (state.feature_shares, averaged over branches and
combined over the day's check-ins). So S splits exactly into "which days" and "which features" –
nothing is hidden. Days older than the window are reported together as `older`.
"""

import math

PHRASES = {
    "arousal": "less energetic-sounding voice",
    "energy": "quieter voice",
    "valence": "more negative-sounding voice",
    "speech_rate": "slower speech",
    "f0_std": "flatter pitch",
    "pause_ratio": "more pauses",
    "jitter": "less steady pitch",
    "shimmer": "less steady loudness",
    "text_logit": "diary language closer to depression",
    "sentiment": "more negative wording",
    "absolutist": "more absolute words (always, never)",
    "first_person": "more self-focused wording",
}


def day_feature_contributions(sums_by_branch: dict[str, dict[str, float]], n_by_branch: dict[str, int]) -> dict[str, float]:
    """Per feature, its part of the day's D_t (= mean over branches of Σ_k D_bk / √n_b)."""
    out: dict[str, float] = {}
    branches = [b for b in sums_by_branch if n_by_branch.get(b)]
    for b in branches:
        for f, v in sums_by_branch[b].items():
            out[f] = out.get(f, 0.0) + v / math.sqrt(n_by_branch[b]) / len(branches)
    return out


def explain(days: list[dict], S: float, alpha: float, window: int = 14) -> dict:
    """days: oldest→newest scored days of one track, each {"date", "D", "features": {f: part of D}}.
    Returns the split of S over the last `window` days and over features, plus a sentence."""
    m = len(days) - 1
    by_day, by_feature = [], {}
    for j, d in enumerate(days):
        if j < m + 1 - window:
            continue
        w = alpha * (1 - alpha) ** (m - j) if j > 0 else (1 - alpha) ** m
        by_day.append({"date": str(d["date"]), "D": round(d["D"], 3), "weight": round(w, 4), "contribution": round(w * d["D"], 4)})
        for f, v in d.get("features", {}).items():
            by_feature[f] = by_feature.get(f, 0.0) + w * v
    older = S - sum(x["contribution"] for x in by_day)
    by_feature = {f: round(v, 4) for f, v in sorted(by_feature.items(), key=lambda t: -t[1])}
    return {"S": round(S, 4), "by_day": by_day, "by_feature": by_feature, "older": round(older, 4), "sentence": sentence(S, by_feature)}


def sentence(S: float, by_feature: dict[str, float]) -> str:
    worse = [f for f, v in by_feature.items() if v > 0.1 and f in PHRASES][:2]
    if S > 0.5 and worse:
        return "Compared with the healthy normal over the last days: " + " and ".join(PHRASES[f] for f in worse) + "."
    if S < -1:
        return "Above the healthy normal over the last days (more energy and positive tone than usual)."
    return "Close to the healthy normal over the last days."
