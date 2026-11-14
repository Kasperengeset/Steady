import json
import os
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlmodel import Session, col, select

from .. import baseline, explanation
from ..config import settings
from ..db import get_session
from ..engine import bandit, escalation, state
from ..models import (
    Arm, BaselineStatus, CheckIn, ContactKind, Escalation, Feature, InterventionLog, Patient, Slot, StateScore,
)
from ..pipeline import PipelineResult, extract_features, extract_text_features
from ..schemas import BaselineOut, CheckInResponse, FeatureOut, InterventionOut, StateOut, TextCheckInIn

router = APIRouter(tags=["checkins"])


def _now() -> datetime:
    """Current time from the shared clock in baseline.py (tests replace it to advance days)."""
    return baseline.clock()


def _rolling_history(s: Session, patient_id: int, before: datetime, slot: Slot) -> dict[str, list[float]]:
    """Feature values from the same slot over the last `baseline_days` days (morning vs morning, T10)."""
    since = before - timedelta(days=settings.baseline_days)
    rows = s.exec(
        select(Feature.name, Feature.value)
        .join(CheckIn, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == patient_id, CheckIn.slot == slot, CheckIn.ts >= since, CheckIn.ts < before)
    ).all()
    hist: dict[str, list[float]] = defaultdict(list)
    for name, v in rows:
        hist[name].append(v)
    return hist


def _day_lists(raw: str | None) -> dict[str, dict[str, list[float]]]:
    """Today's per-check-in deviations {track: {branch: [D, ...]}}; old rows stored one float per branch."""
    out: dict[str, dict[str, list[float]]] = {"anchored": {}, "rolling": {}}
    for t, branches in (json.loads(raw) if raw else {}).items():
        if t in out and isinstance(branches, dict):
            out[t] = {b: (v if isinstance(v, list) else [v]) for b, v in branches.items()}
    return out


def _score_tracks(
    s: Session, patient_id: int, now: datetime, slot: Slot, today_by_branch: dict[str, dict[str, float]],
    active,
) -> dict:
    """Both tracks: vs the fixed healthy baseline (slow drift) and vs the last 14 days (abrupt change)."""
    anchor_stats = baseline.stats_for(s, active, slot.value)
    hist = _rolling_history(s, patient_id, now, slot)
    k = settings.sd_prior_strength
    D = {"anchored": {}, "rolling": {}}
    Z = {"anchored": {}, "rolling": {}}
    shares = {"anchored": {}, "rolling": {}}
    for branch, vals in today_by_branch.items():
        model = state.load_branch(branch)
        tracks = {
            "anchored": {f: anchor_stats[f] for f in model.direction if f in anchor_stats},
            "rolling": state.history_stats(hist, model, k, settings.rolling_min_n),
        }
        for track, stats in tracks.items():
            z = state.zscores(vals, stats, model.direction)
            if z:
                Z[track][branch] = z
                shares[track][branch] = state.feature_shares(z, model)
                D[track][branch] = round(sum(shares[track][branch].values()), 4)
    return {"D": D, "Z": Z, "shares": shares}


def _todays_intervention(s: Session, patient_id: int, today: date) -> InterventionOut:
    arms = s.exec(select(Arm).where(Arm.patient_id == patient_id)).all()
    labels = {a.key: a.label for a in arms}
    log = s.exec(
        select(InterventionLog).where(InterventionLog.patient_id == patient_id, InterventionLog.date == today)
    ).first()
    if log is None:
        arm = bandit.choose_arm(arms)
        log = InterventionLog(patient_id=patient_id, date=today, arm_key=arm.key)
        s.add(log)
    return InterventionOut(arm_key=log.arm_key, label=labels.get(log.arm_key, log.arm_key))


@router.post("/checkins", response_model=CheckInResponse)
async def create_checkin(
    audio: UploadFile,
    patient_id: int = Form(...),
    slot: Slot = Form(...),
    s: Session = Depends(get_session),
) -> CheckInResponse:
    patient = s.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(404, "patient not found")

    # Raw audio lives only in a temp file and is deleted right after feature extraction.
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        result = extract_features(tmp_path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        os.remove(tmp_path)
    return _record_checkin(s, patient, slot, result)


@router.post("/checkins/text", response_model=CheckInResponse)
def create_text_checkin(body: TextCheckInIn, s: Session = Depends(get_session)) -> CheckInResponse:
    """Written diary entry -> text branch only (no audio). English; 50 to 100 words works best."""
    patient = s.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(404, "patient not found")
    if not body.text.strip():
        raise HTTPException(400, "empty diary entry")
    return _record_checkin(s, patient, body.slot, extract_text_features(body.text))


def _record_checkin(s: Session, patient: Patient, slot: Slot, result: PipelineResult) -> CheckInResponse:
    """Store features, run the state engine, escalate, pick today's intervention."""
    patient_id = patient.id
    now = _now()
    checkin = CheckIn(
        patient_id=patient_id, ts=now, slot=slot,
        transcript=result.transcript, llm_json=result.llm_json, risk_flag=result.crisis,
    )
    s.add(checkin)
    s.flush()
    for name, value, source in result.features:
        s.add(Feature(checkin_id=checkin.id, name=name, value=value, source=source))

    # Healthy-baseline lifecycle (STATUS.md T1): no score until a baseline is locked
    active = baseline.latest(s, patient_id, BaselineStatus.active)
    building = baseline.latest(s, patient_id, BaselineStatus.building)
    if active is None and building is None:
        building = baseline.start(s, patient_id, now=now)
    if building is not None and baseline.try_lock(s, building, now):
        active, building = building, None

    today_by_branch: dict[str, dict[str, float]] = defaultdict(dict)
    for name, v, source in result.features:
        today_by_branch[state.BRANCH_OF_SOURCE[source]][name] = v

    S_val: float | None = None
    elevated = False
    S_track: dict[str, float | None] = {"anchored": None, "rolling": None}
    driver: str | None = None
    branch_D: dict[str, float] = {}
    if active is not None:
        tr = _score_tracks(s, patient_id, now, slot, today_by_branch, active)
        prev = s.exec(
            select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date < now.date())
            .order_by(col(StateScore.date).desc())
        ).first()
        score = s.exec(
            select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date == now.date())
        ).first()
        # All check-ins of the day count (STATUS.md B16): keep each check-in's D per track and branch,
        # combine per branch with sum/sqrt(n), then average over branches. A later check-in no longer
        # overwrites an earlier one (e.g. a negative diary followed by a neutral spoken check-in).
        day = _day_lists(score.branch_scores_json if score else None)
        evidence = json.loads(score.evidence_json) if score and score.evidence_json else {}
        for t in day:
            for branch, d in tr["D"][t].items():
                day[t].setdefault(branch, []).append(d)
                sums = evidence.setdefault(t, {}).setdefault(branch, {})
                for f, v in tr["shares"][t][branch].items():
                    sums[f] = round(sums.get(f, 0.0) + v, 5)
        day_branch = {t: {b: round(state.combine_checkins(v), 4) for b, v in day[t].items()} for t in day}
        D_track = {t: state.daily_deviation(day_branch[t]) for t in S_track}
        prev_S = {"anchored": prev.S_anchored if prev else None, "rolling": prev.S_rolling if prev else None}
        for t, d in D_track.items():
            if d is not None:
                S_track[t] = round(state.ewma(d, prev_S[t], settings.ewma_alpha), 5)
        present = {t: v for t, v in S_track.items() if v is not None}
        if present:
            # Escalate on whichever track is worse; tell the clinician which one (drift vs abrupt change)
            driver = max(present, key=present.get)
            S_val = present[driver]
            branch_D = day_branch[driver]
            score = score or StateScore(patient_id=patient_id, date=now.date(), D=D_track[driver], S=S_val)
            score.D, score.S, score.driver, score.baseline_version = D_track[driver], S_val, driver, active.version
            score.D_anchored, score.S_anchored = D_track["anchored"], S_track["anchored"]
            score.D_rolling, score.S_rolling = D_track["rolling"], S_track["rolling"]
            if tr["Z"][driver]:
                score.top_features_json = json.dumps(state.top_contributors(tr["Z"][driver]))
            score.branch_scores_json = json.dumps(day)
            score.evidence_json = json.dumps(evidence)
            recent_anch = [x.S_anchored for x in s.exec(
                select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date < now.date())
                .order_by(col(StateScore.date))
            ).all() if x.S_anchored is not None][-(settings.elevated_days - 1):]
            if S_track["anchored"] is not None:
                recent_anch.append(S_track["anchored"])
            score.elevated = escalation.elevated_flag(recent_anch, settings.elevated_S, settings.elevated_days)
            elevated = score.elevated
            s.add(score)

    recent = [x.S for x in s.exec(
        select(StateScore).where(StateScore.patient_id == patient_id).order_by(col(StateScore.date))
    ).all()]
    if S_val is not None and (not recent or recent[-1] != S_val):
        recent.append(S_val)
    level = escalation.evaluate_level(recent[-7:], result.crisis)
    if level > 0 and level != patient.risk_level:
        recipient = ContactKind.friend if level == 1 else ContactKind.clinician
        s.add(Escalation(patient_id=patient_id, level=level, recipient=recipient,
                         reason="crisis flag" if result.crisis else f"S={S_val} ({driver})"))
    patient.risk_level = level
    s.add(patient)

    intervention = _todays_intervention(s, patient_id, now.date())
    s.commit()
    why = explanation.latest_explanation(s, patient_id) if S_val is not None else None

    return CheckInResponse(
        checkin_id=checkin.id,
        features=[FeatureOut(name=n, value=round(v, 3), source=src.value) for n, v, src in result.features],
        state=StateOut(
            S=S_val, level=level, branches=branch_D,
            S_anchored=S_track["anchored"], S_rolling=S_track["rolling"], driver=driver,
            elevated=elevated, explanation=why,
        ),
        baseline=BaselineOut(**baseline.progress(s, active or building, now)),
        baseline_building=BaselineOut(**baseline.progress(s, building, now)) if building else None,
        intervention=intervention,
        crisis=result.crisis,
    )
