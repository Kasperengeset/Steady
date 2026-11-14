"""Builds the clinician explanation of a patient's current S from stored daily scores (STATUS.md B20)."""

import json

from sqlmodel import Session, col, select

from .config import settings
from .engine import explain
from .models import StateScore


def _day_features(row: StateScore, track: str) -> dict[str, float]:
    ev = json.loads(row.evidence_json or "{}").get(track, {})
    raw = json.loads(row.branch_scores_json or "{}")
    lists, weights = raw.get(track, {}), raw.get("w", {}).get(track, {})
    # The day's D per branch is Σ w·D / √(Σ w²) (B31); evidence holds Σ w·share, so divide by √(Σ w²).
    # Rows without weights are B16 (all 1): Σ w² = n.
    n = {b: sum(w * w for w in weights[b]) if b in weights else (len(v) if isinstance(v, list) else 1)
         for b, v in lists.items()}
    return explain.day_feature_contributions(ev, n)


def latest_explanation(s: Session, patient_id: int, track: str | None = None, window: int = 14) -> dict | None:
    rows = list(s.exec(select(StateScore).where(StateScore.patient_id == patient_id).order_by(col(StateScore.date))).all())
    if not rows:
        return None
    track = track or rows[-1].driver or "anchored"
    D_field, S_field = f"D_{track}", f"S_{track}"
    days = [{"date": r.date, "D": getattr(r, D_field), "features": _day_features(r, track)}
            for r in rows if getattr(r, D_field) is not None]
    S = getattr(rows[-1], S_field)
    if not days or S is None:
        return None
    out = explain.explain(days, S, settings.ewma_alpha, window)
    out["track"] = track
    return out
