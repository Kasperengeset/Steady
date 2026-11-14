"""Clinician read models and immutable per-entry analysis. No diagnosis is inferred here.

Snapshots freeze the measurements, references, artifacts and arithmetic used when
an entry arrived. Old rows may be compared retrospectively, but their missing
historical decisions are never presented as exact ingestion-time evidence.
"""

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, col, select

from . import baseline, explanation
from .config import settings
from .engine import state
from .models import Arm, Baseline, BaselineStat, BaselineStatus, CheckIn, Feature, FeatureSource, InterventionLog, Patient, StateScore
from .pipeline import text as text_model

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"acoustic": "Voice", "text": "Language", "health": "Wearable health"}
LIMITATIONS = [
    "These models measure patterns and change from a person's reference. They are not validated to diagnose depression or predict relapse.",
    "Synthetic reference data demonstrates the software. It is not a clinically confirmed healthy baseline for a real person.",
    "Missing sources are omitted. A missing measurement is not a reassuring measurement.",
]


def _json(value, default=None):
    try:
        return json.loads(value) if value else ({} if default is None else default)
    except (TypeError, ValueError):
        return {} if default is None else default


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _aware(value).isoformat() if value else None


def origin(value: str | None) -> str:
    return value if value in ("live", "synthetic") else "unknown"


def _data_source(row: CheckIn) -> dict:
    """Only provenance fields, never arbitrary provider JSON, secrets or request bodies."""
    payload = _json(row.llm_json)
    wearable = payload.get("wearable", {})
    if not isinstance(wearable, dict):
        wearable = {}
    source = wearable.get("source")
    source = source if source in ("demo", "google", "google_health", "fitbit", "manual", "csv") else None
    return {"source": source or ("simulation" if origin(row.data_origin) == "synthetic" else "live" if origin(row.data_origin) == "live" else "unknown"),
            "measurement_date": str(wearable["measurement_date"])[:10] if wearable.get("measurement_date") else None,
            "received_at": str(wearable["received_at"])[:40] if wearable.get("received_at") else _iso(row.ts),
            "synthetic": origin(row.data_origin) == "synthetic"}


def _hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@lru_cache(maxsize=1)
def model_catalog() -> list[dict]:
    """Versioned local artifacts plus the results reported in this checkout."""
    specs = {
        "acoustic": ("ml/reports/acoustic/summary.md", {"roc_auc": 0.770, "roc_auc_sd": 0.076}, 114,
                     "Five corpus folds, held-out speakers", ["Italian Androids interviews; small sample and recording-condition sensitivity.",
                     "The app uses eight features with equal weights and correlation normalization, not a calibrated voice diagnosis probability."]),
        "text": ("ml/reports/text/v1.md", {"roc_auc": 0.970, "f1": 0.908}, 9198,
                 "70/15/15 split by Reddit author; balanced test chunks", ["Forum language differs from daily diaries and can retain topic and writing-style effects.",
                 "Gemini content ratings have zero drift weight until separately validated."]),
        "health": ("ml/reports/health/summary.md", {"roc_auc": 0.818, "f1": 0.706, "mcc": 0.464}, 55,
                   "Leave-one-patient-out on Depresjon", ["Actiwatch activity counts differ from Fitbit steps.",
                   "Sleep, resting heart rate and HRV directions are literature priors; exercise duration earns habit food but has no drift weight."]),
    }
    out = []
    for key, (report, metrics, sample_size, split, limits) in specs.items():
        model = state.load_branch(key)
        path = state.WEIGHTS_DIR / f"{key}.json"
        report_present = (ROOT / report).is_file()
        item = {
            "key": key, "label": LABELS[key], "status": "configured" if model.trained else "literature-fallback",
            "description": {
                "acoustic": "openSMILE prosody and audEERING wav2vec2 emotion dimensions feed eight baseline-relative measurements, with equal weights and correlation normalization.",
                "text": "Masked TF-IDF unigrams/bigrams and logistic regression, plus lexical features. Actual diary or transcript outputs are compared with the personal reference.",
                "health": "Wearable activity, sleep and recovery features are compared with the personal reference. Activity directions have Depresjon evidence; recovery directions are literature priors.",
            }[key],
            "dataset": {"acoustic": "Androids interviews", "text": "Reddit Mental Health Dataset", "health": "Depresjon"}[key],
            "artifact_path": str(path.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": _hash(path),
            "features": [{"name": name, "weight": model.weights.get(name, 0), "direction": direction}
                         for name, direction in model.direction.items()],
            "validation": {"report": report, "metrics": metrics if report_present else {},
                           "sample_size": sample_size if report_present else None, "split": split,
                           "sample_unit": {"acoustic": "speakers across five held-out folds", "text": "held-out test chunks, balanced 1:1", "health": "people, held out one at a time"}[key],
                           "scope": "Public-dataset classification; not longitudinal clinical relapse validation.",
                           "report_sha256": _hash(ROOT / report)},
            "limitations": limits,
        }
        if key == "text":
            item["classifier_sha256"] = _hash(text_model.MODEL_PATH)
        out.append(item)
    return out


def score_point(row: StateScore | None) -> dict | None:
    if row is None:
        return None
    raw = _json(row.branch_scores_json)
    tracks = {}
    for track in ("anchored", "rolling"):
        values = {}
        for branch, entries in raw.get(track, {}).items():
            seq = entries if isinstance(entries, list) else [entries]
            if not seq:
                continue
            weights = raw.get("w", {}).get(track, {}).get(branch)
            values[branch] = round(state.combine_checkins(seq, weights), 4)
        tracks[track] = values
    return {"date": row.date.isoformat(), "D": row.D, "S": row.S, "D_anchored": row.D_anchored,
            "S_anchored": row.S_anchored, "D_rolling": row.D_rolling, "S_rolling": row.S_rolling,
            "driver": row.driver, "level": row.level, "elevated": row.elevated,
            "baseline_version": row.baseline_version, "branches": tracks.get(row.driver or "anchored", {}),
            "tracks": tracks, "day_weights": raw.get("w", {})}


def baseline_origin(s: Session, reference: Baseline | None) -> str:
    if reference is None:
        return "unknown"
    q = select(CheckIn.data_origin).where(CheckIn.patient_id == reference.patient_id, CheckIn.ts >= reference.started_at)
    if reference.locked_at is not None:
        q = q.where(CheckIn.ts <= reference.locked_at)
    values = {origin(x) for x in s.exec(q).all()}
    return next(iter(values)) if len(values) == 1 else "mixed" if values else "unknown"


def _raw_outputs(features: list[tuple[str, float, FeatureSource]]) -> dict:
    branches = {key: {} for key in LABELS}
    for name, value, src in features:
        if not name.startswith("phrase_"):
            branches[state.BRANCH_OF_SOURCE[src]][name] = value
    text = branches["text"]
    if "text_logit" in text:
        text["logit"] = text["text_logit"]
        text["p_depressive"] = 1 / (1 + math.exp(-max(-700, min(700, text["text_logit"]))))
        text["interpretation"] = "Reddit-trained depressive-language classifier output, not a clinical depression probability."
    return {"text": text or None, "acoustic": branches["acoustic"], "health": branches["health"]}


def _word_evidence(transcript: str | None, features, provenance: str) -> dict:
    values = {n: v for n, v, _ in features}
    if not transcript or transcript.startswith("(stub)") or "text_logit" not in values:
        return {"available": False, "words_up": [], "words_down": [], "provenance": provenance,
                "reason": "No actual transcript and text-model output are available together."}
    try:
        prepared = text_model.prepare(transcript)
        matches = math.isclose(text_model.text_logit(prepared), values["text_logit"], abs_tol=1e-5)
        if not matches:
            return {"available": False, "words_up": [], "words_down": [], "provenance": provenance,
                    "reason": "The current text artifact does not reproduce this stored output."}
        return {"available": True, **text_model.explain(prepared, 8), "provenance": provenance,
                "note": "Token contributions explain the local language classifier, not personal drift or a diagnosis."}
    except (OSError, ValueError, KeyError):
        return {"available": False, "words_up": [], "words_down": [], "provenance": provenance,
                "reason": "Local text artifact unavailable for attribution."}


def capture_analysis(s: Session, checkin: CheckIn, features, reference: Baseline | None, tr: dict,
                     before: dict | None, after: dict | None, entry_weights: dict,
                     phrase=None, prompt=None, provenance: str = "snapshot") -> dict:
    previous_day = s.exec(select(StateScore).where(
        StateScore.patient_id == checkin.patient_id, StateScore.date < checkin.ts.date())
        .order_by(col(StateScore.date).desc())).first() if provenance == "snapshot" else None
    tracks = {}
    for track in ("anchored", "rolling"):
        stats = {b: {name: {"center": v[0], "scale": v[1], "n": v[2]} for name, v in values.items()}
                 for b, values in tr.get("stats", {}).get(track, {}).items()}
        tracks[track] = {"D": tr["D"][track], "Z": tr["Z"][track], "shares": tr["shares"][track], "stats": stats}
    return {
        "schema_version": 1, "provenance": provenance, "snapshot_kind": provenance,
        "captured_at": _iso(baseline.clock()), "observed_at": _iso(checkin.ts), "data_origin": origin(checkin.data_origin),
        "baseline_id": reference.id if reference else None, "baseline_version": reference.version if reference else None,
        "baseline_origin": baseline_origin(s, reference), "baseline_status": reference.status.value if reference else "building",
        "tracks": tracks, "adjusted_values": tr.get("adjusted", {}), "hour_slopes": tr.get("hour_slopes", {}),
        "local_hour": checkin.local_hour, "time_of_day_mode": settings.time_of_day,
        "engine": {"ewma_alpha": settings.ewma_alpha, "extra_checkin_weight": state.EXTRA_WEIGHT,
                   "rolling_window_days": settings.baseline_days, "rolling_min_n": settings.rolling_min_n,
                   "branch_aggregation": "available branch mean", "state_selection": "maximum reference track"},
        "previous_day_state": score_point(previous_day),
        "entry_weights": entry_weights, "before": before, "after": after,
        "day_weights": (after or {}).get("day_weights", {}),
        "score_updated": provenance == "snapshot" and any(tr["D"][t] for t in ("anchored", "rolling")),
        "models": [{"key": m["key"], "artifact_sha256": m["artifact_sha256"],
                    "classifier_sha256": m.get("classifier_sha256"), "weights": {f["name"]: f["weight"] for f in m["features"]},
                    "directions": {f["name"]: f["direction"] for f in m["features"]}} for m in model_catalog()],
        "raw_outputs": _raw_outputs(features),
        "data_source": _data_source(checkin),
        "word_evidence": _word_evidence(checkin.transcript, features, "at_ingestion" if provenance == "snapshot" else "recomputed_current_artifact"),
        "phrase": phrase, "prompt": prompt,
        "scoring_status": "scored" if any(tr["D"][t] for t in ("anchored", "rolling")) else
                          "collecting_baseline" if reference is None else "observed_without_scored_reference",
        "limitations": [] if provenance == "snapshot" else ["Retrospective comparison with current artifacts. Original per-entry weighting and before/after scores were not stored."],
    }


def _features(s: Session, ids: list[int]) -> dict[int, list[Feature]]:
    out = defaultdict(list)
    if ids:
        for feature in s.exec(select(Feature).where(col(Feature.checkin_id).in_(ids))).all():
            out[feature.checkin_id].append(feature)
    return out


def entry_kind(row: CheckIn, features: list[Feature]) -> str:
    sources = {f.source for f in features if not f.name.startswith("phrase_")}
    if sources & {FeatureSource.prosody, FeatureSource.emotion}:
        return "voice"
    if sources & {FeatureSource.text, FeatureSource.llm} or (row.transcript and not row.transcript.startswith("(stub)")):
        return "text"
    return "wearable" if FeatureSource.wearable in sources else "other"


def entry_summary(row: CheckIn, features: list[Feature]) -> dict:
    analysis = _json(row.analysis_json)
    scored = sorted({b for track in analysis.get("tracks", {}).values() for b in track.get("D", {})})
    transcript = row.transcript if row.transcript and not row.transcript.startswith("(stub)") else None
    return {"id": row.id, "patient_id": row.patient_id, "ts": _iso(row.ts), "slot": row.slot.value,
            "kind": entry_kind(row, features), "data_origin": origin(row.data_origin),
            "transcript_preview": transcript[:180] if transcript else None, "crisis": row.risk_flag,
            "analysis_available": bool(features), "has_snapshot": bool(analysis),
            "analysis_provenance": "snapshot" if analysis else "reconstructed", "snapshot_kind": "snapshot" if analysis else "reconstructed",
            "feature_count": len(features), "scored_sources": scored,
            "S": (analysis.get("after") or {}).get("S"), "baseline_origin": analysis.get("baseline_origin", "unknown")}


def entries(s: Session, pid: int, limit: int = 50, offset: int = 0) -> dict:
    all_rows = s.exec(select(CheckIn).where(CheckIn.patient_id == pid)
                      .order_by(col(CheckIn.ts).desc(), col(CheckIn.id).desc())).all()
    selected = all_rows[offset:offset + limit]
    feats = _features(s, [row.id for row in selected])
    return {"total": len(all_rows), "limit": limit, "offset": offset,
            "entries": [entry_summary(row, feats[row.id]) for row in selected]}


def _reconstruct(s: Session, row: CheckIn, features: list[Feature]) -> dict:
    from .routers.checkins import _score_tracks
    daily = s.exec(select(StateScore).where(StateScore.patient_id == row.patient_id, StateScore.date == row.ts.date())).first()
    refs = s.exec(select(Baseline).where(Baseline.patient_id == row.patient_id, col(Baseline.locked_at).is_not(None))
                  .order_by(col(Baseline.version).desc())).all()
    reference = next((b for b in refs if daily and b.version == daily.baseline_version), None)
    reference = reference or next((b for b in refs if _aware(b.locked_at) <= _aware(row.ts)), None)
    values = defaultdict(dict)
    for feature in features:
        if not feature.name.startswith("phrase_") and feature.name != "active_minutes":
            values[state.BRANCH_OF_SOURCE[feature.source]][feature.name] = feature.value
    tr = _score_tracks(s, row.patient_id, _aware(row.ts), row.slot, values, reference, row.local_hour) if reference else {
        "D": {"anchored": {}, "rolling": {}}, "Z": {"anchored": {}, "rolling": {}}, "shares": {"anchored": {}, "rolling": {}}}
    triples = [(f.name, f.value, f.source) for f in features]
    result = capture_analysis(s, row, triples, reference, tr, None, None, {},
                              phrase=_json(row.llm_json).get("phrase"), provenance="reconstructed")
    result["daily_final"] = score_point(daily)
    return result


def entry_detail(s: Session, row: CheckIn) -> dict:
    features = _features(s, [row.id])[row.id]
    analysis = _json(row.analysis_json) or _reconstruct(s, row, features)
    driver = (analysis.get("after") or {}).get("driver") or "anchored"
    track = analysis.get("tracks", {}).get(driver, {})
    models = {m["key"]: m for m in analysis.get("models", [])}
    measurements = []
    for feature in features:
        branch = state.BRANCH_OF_SOURCE[feature.source]
        stats = track.get("stats", {}).get(branch, {}).get(feature.name, {})
        z = track.get("Z", {}).get(branch, {}).get(feature.name)
        weight = models.get(branch, {}).get("weights", {}).get(feature.name, 0)
        measurements.append({"name": feature.name, "value": feature.value, "source": feature.source.value,
            "branch": branch, "center": stats.get("center"), "scale": stats.get("scale"), "n": stats.get("n"),
            "z": z, "weight": weight, "direction": models.get(branch, {}).get("directions", {}).get(feature.name),
            "share": track.get("shares", {}).get(branch, {}).get(feature.name),
            "used": z is not None and weight > 0, "track": driver,
            "adjusted_value": analysis.get("adjusted_values", {}).get(branch, {}).get(feature.name),
            "entry_weight": analysis.get("entry_weights", {}).get(branch),
            "usage_known": analysis["provenance"] == "snapshot"})
    transcript = row.transcript if row.transcript and not row.transcript.startswith("(stub)") else None
    return {"entry": entry_summary(row, features), "transcript": transcript, "features": measurements,
            "analysis": analysis, "before": analysis.get("before"), "after": analysis.get("after"),
            "data_source": analysis.get("data_source") or _data_source(row),
            "word_evidence": analysis.get("word_evidence"), "raw_outputs": analysis.get("raw_outputs"),
            "phrase": analysis.get("phrase"), "speech_s": row.speech_s,
            "transcription_status": "available" if transcript else "not-applicable" if entry_kind(row, features) == "wearable" else "unavailable",
            "limitations": [*LIMITATIONS[:1], *analysis.get("limitations", [])]}


def _source_coverage(rows: list[CheckIn], features: dict[int, list[Feature]]) -> list[dict]:
    result = []
    for branch, label in LABELS.items():
        selected = [c for c in rows if any(state.BRANCH_OF_SOURCE[f.source] == branch and not f.name.startswith("phrase_") for f in features[c.id])]
        live = [c for c in selected if origin(c.data_origin) == "live"]
        scored = [c for c in selected if any(branch in t.get("D", {}) for t in _json(c.analysis_json).get("tracks", {}).values())]
        result.append({"key": branch, "label": label, "entries": len(selected), "live_entries": len(live),
                       "synthetic_entries": sum(origin(c.data_origin) == "synthetic" for c in selected),
                       "scored_entries": len(scored), "scored": bool(scored),
                       "last_at": _iso(max((c.ts for c in selected), default=None)),
                       "last_live_at": _iso(max((c.ts for c in live), default=None))})
    return result


def _baseline_view(s: Session, pid: int, now: datetime) -> dict:
    active = baseline.latest(s, pid, BaselineStatus.active)
    building = baseline.latest(s, pid, BaselineStatus.building)
    def show(ref):
        return {**baseline.progress(s, ref, now), "id": ref.id, "note": ref.note,
                "origin": baseline_origin(s, ref)} if ref else None
    stats = s.exec(select(BaselineStat).where(BaselineStat.baseline_id == active.id)).all() if active else []
    return {"active": show(active), "building": show(building), "origin": baseline_origin(s, active or building),
            "stats": [{"slot": x.slot, "feature": x.feature, "center": x.center, "scale": x.scale, "n": x.n,
                       "branch": baseline.FEATURE_BRANCH.get(x.feature)} for x in stats]}


def _patient_summary(s: Session, patient: Patient, rows: list[CheckIn], feats, scores: list[StateScore], now: datetime) -> dict:
    current = _current_state(patient, scores, now)
    counts = {key: sum(entry_kind(c, feats[c.id]) == key for c in rows) for key in ("voice", "text", "wearable")}
    live = [c for c in rows if origin(c.data_origin) == "live"]
    ref = baseline.latest(s, patient.id, BaselineStatus.active) or baseline.latest(s, patient.id, BaselineStatus.building)
    return {"id": patient.id, "name": patient.name, "demo_key": patient.demo_key,
            "data_origin": "demo-profile" if patient.demo_key else "unmarked", "created_at": _iso(patient.created_at),
            "current": current, "baseline": {**baseline.progress(s, ref, now), "origin": baseline_origin(s, ref)} if ref else None,
            "baseline_origin": baseline_origin(s, ref), "last_entry_at": _iso(max((c.ts for c in rows), default=None)),
            "last_live_entry_at": _iso(max((c.ts for c in live), default=None)), "entry_counts": counts,
            "sources": _source_coverage(rows, feats),
            "needs_attention": patient.risk_level > 0 or bool(current and current["elevated"]),
            "level": patient.risk_level}


def _current_state(patient: Patient, scores: list[StateScore], now: datetime) -> dict:
    point = score_point(scores[-1]) if scores else {
        "date": now.date().isoformat(), "D": None, "S": None,
        "D_anchored": None, "S_anchored": None, "D_rolling": None, "S_rolling": None,
        "driver": None, "baseline_version": None, "branches": {}, "tracks": {}, "elevated": False,
    }
    point["level"] = patient.risk_level
    point["has_score"] = bool(scores)
    return point


def overview(s: Session) -> dict:
    now = baseline.clock()
    people = s.exec(select(Patient).order_by(Patient.id)).all()
    rows = s.exec(select(CheckIn).where(CheckIn.ts <= now)).all()
    feats = _features(s, [c.id for c in rows])
    scores = s.exec(select(StateScore).where(StateScore.date <= now.date()).order_by(StateScore.date)).all()
    by_patient, score_by_patient = defaultdict(list), defaultdict(list)
    for row in rows:
        by_patient[row.patient_id].append(row)
    for row in scores:
        score_by_patient[row.patient_id].append(row)
    cohort = [_patient_summary(s, p, by_patient[p.id], feats, score_by_patient[p.id], now) for p in people]
    cohort.sort(key=lambda p: (-p["level"], -bool(p["current"] and p["current"]["elevated"]), p["name"]))
    today = [c for c in rows if c.ts.date() == now.date()]
    return {"generated_at": _iso(now), "patients": cohort,
            "summary": {"patients": len(people), "attention": sum(p["needs_attention"] for p in cohort),
                        "baseline_building": sum(p["baseline"] is None or p["baseline"]["status"] == "building" for p in cohort),
                        "entries_today": len(today), "live_entries_today": sum(origin(c.data_origin) == "live" for c in today)},
            "models": model_catalog(), "limitations": LIMITATIONS}


def workspace(s: Session, patient: Patient, days: int = 30) -> dict:
    now = baseline.clock()
    since = now.date() - timedelta(days=days - 1)
    start = datetime.combine(since, datetime.min.time(), tzinfo=timezone.utc)
    rows = s.exec(select(CheckIn).where(CheckIn.patient_id == patient.id, CheckIn.ts >= start, CheckIn.ts <= now)
                  .order_by(CheckIn.ts, CheckIn.id)).all()
    feats = _features(s, [c.id for c in rows])
    all_scores = s.exec(select(StateScore).where(StateScore.patient_id == patient.id, StateScore.date <= now.date())
                        .order_by(StateScore.date)).all()
    by_day, scores_by_day = defaultdict(list), {x.date: x for x in all_scores}
    for row in rows:
        by_day[row.ts.date()].append(row)
    series = []
    for i in range(days):
        day = since + timedelta(days=i)
        point = score_point(scores_by_day.get(day)) or {"date": day.isoformat(), "D": None, "S": None, "branches": {}, "tracks": {}, "level": 0, "elevated": False}
        point["coverage"] = {b: sum(any(state.BRANCH_OF_SOURCE[f.source] == b and not f.name.startswith("phrase_") for f in feats[c.id]) for c in by_day[day]) for b in LABELS}
        point["origins"] = sorted({origin(c.data_origin) for c in by_day[day]})
        series.append(point)
    labels = {a.key: a.label for a in s.exec(select(Arm).where(Arm.patient_id == patient.id)).all()}
    actions = s.exec(select(InterventionLog).where(InterventionLog.patient_id == patient.id, InterventionLog.date >= since)
                     .order_by(col(InterventionLog.date).desc())).all()
    snapshots = [_json(c.analysis_json) for c in reversed(rows) if c.analysis_json]
    return {"generated_at": _iso(now), "patient": {"id": patient.id, "name": patient.name, "demo_key": patient.demo_key,
                    "data_origin": "demo-profile" if patient.demo_key else "unmarked", "created_at": _iso(patient.created_at),
                    "level": patient.risk_level, "needs_attention": patient.risk_level > 0 or bool(all_scores and all_scores[-1].elevated)},
            "current": _current_state(patient, all_scores, now),
            "baseline": _baseline_view(s, patient.id, now), "series": series,
            "baseline_origin": baseline_origin(s, baseline.latest(s, patient.id, BaselineStatus.active)),
            "coverage": {"window_days": days, "total_entries": len(rows), "live_entries": sum(origin(c.data_origin) == "live" for c in rows),
                         "synthetic_entries": sum(origin(c.data_origin) == "synthetic" for c in rows),
                         "unknown_entries": sum(origin(c.data_origin) == "unknown" for c in rows),
                         "days_with_entries": len(by_day), "scored_days": sum(x.date >= since for x in all_scores)},
            "sources": _source_coverage(rows, feats), "explanation": explanation.latest_explanation(s, patient.id),
            "actions": [{"date": x.date.isoformat(), "arm_key": x.arm_key, "label": labels.get(x.arm_key, x.arm_key),
                         "completed": x.completed, "rating": x.rating, "reward": x.reward} for x in actions],
            "models": model_catalog(), "analysis": snapshots[0] if snapshots else None,
            "origins": sorted({origin(c.data_origin) for c in rows}),
            "recent_entries": [entry_summary(c, feats[c.id]) for c in reversed(rows[-8:])], "limitations": LIMITATIONS}
