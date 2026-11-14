import json
import os
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlmodel import Session, col, select

from ..config import settings
from ..db import get_session
from ..engine import bandit, escalation, state
from ..models import (
    Arm, CheckIn, Contact, ContactKind, Escalation, Feature, InterventionLog, Patient, Slot, StateScore,
)
from ..pipeline import PipelineResult, extract_features, extract_text_features
from ..schemas import CheckInResponse, FeatureOut, InterventionOut, StateOut, TextCheckInIn

router = APIRouter(tags=["checkins"])


def _baseline_history(s: Session, patient_id: int, before: datetime) -> dict[str, list[float]]:
    since = before - timedelta(days=settings.baseline_days)
    rows = s.exec(
        select(Feature.name, Feature.value)
        .join(CheckIn, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == patient_id, CheckIn.ts >= since, CheckIn.ts < before)
    ).all()
    hist: dict[str, list[float]] = defaultdict(list)
    for name, v in rows:
        hist[name].append(v)
    return hist


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
    now = datetime.now(timezone.utc)
    checkin = CheckIn(
        patient_id=patient_id, ts=now, slot=slot,
        transcript=result.transcript, llm_json=result.llm_json, risk_flag=result.crisis,
    )
    s.add(checkin)
    s.flush()
    for name, value, source in result.features:
        s.add(Feature(checkin_id=checkin.id, name=name, value=value, source=source))

    # State engine – one deviation per branch, then averaged over branches with data
    history = _baseline_history(s, patient_id, now)
    today_by_branch: dict[str, dict[str, float]] = defaultdict(dict)
    for name, v, source in result.features:
        today_by_branch[state.BRANCH_OF_SOURCE[source]][name] = v
    z_by_branch: dict[str, dict[str, float]] = {}
    branch_D: dict[str, float] = {}
    for branch, vals in today_by_branch.items():
        model = state.load_branch(branch)
        z = state.zscores(vals, history, model.direction)
        if z:
            z_by_branch[branch] = z
            branch_D[branch] = round(state.branch_deviation(z, model), 4)
    prev = s.exec(
        select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date < now.date())
        .order_by(col(StateScore.date).desc())
    ).first()
    S_val: float | None = None
    D = state.daily_deviation(branch_D)
    if D is not None:
        S_val = state.ewma(D, prev.S if prev else None, settings.ewma_alpha)
        score = s.exec(
            select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date == now.date())
        ).first() or StateScore(patient_id=patient_id, date=now.date(), D=D, S=S_val)
        score.D, score.S = D, S_val
        score.top_features_json = json.dumps(state.top_contributors(z_by_branch))
        score.branch_scores_json = json.dumps(branch_D)
        s.add(score)

    recent = [x.S for x in s.exec(
        select(StateScore).where(StateScore.patient_id == patient_id).order_by(col(StateScore.date))
    ).all()]
    if S_val is not None and (not recent or recent[-1] != S_val):
        recent.append(S_val)
    level = escalation.evaluate_level(recent[-7:], result.crisis)
    if level > 0 and level != patient.risk_level:
        recipient = ContactKind.friend if level == 1 else ContactKind.clinician
        s.add(Escalation(patient_id=patient_id, level=level, recipient=recipient, reason=f"S={S_val}"))
    patient.risk_level = level
    s.add(patient)

    intervention = _todays_intervention(s, patient_id, now.date())
    s.commit()

    return CheckInResponse(
        checkin_id=checkin.id,
        features=[FeatureOut(name=n, value=round(v, 3), source=src.value) for n, v, src in result.features],
        state=StateOut(S=S_val, level=level, branches=branch_D),
        intervention=intervention,
        crisis=result.crisis,
    )
