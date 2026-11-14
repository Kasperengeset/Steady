import json
import logging
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Form, Header, HTTPException, UploadFile
from sqlalchemy import update
from sqlmodel import Session, col, select

from .. import baseline, explanation, prompt, rewards
from ..config import settings
from ..db import get_session
from ..engine import bandit, escalation, signals, state
from ..models import (
    Arm, BaselineStatus, CheckIn, ContactKind, Escalation, Feature, FeatureSource, InterventionLog, Patient, Slot, StateScore,
)
from ..pipeline import PipelineResult, add_phrase, extract_features, extract_text_features, extract_wearable_features, transcribe_phrase
from ..schemas import BaselineOut, CheckInResponse, FeatureOut, InterventionOut, StateOut, TextCheckInIn, WearableDayIn

router = APIRouter(tags=["checkins"])
log = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def _saved_response(s: Session, pid: int, key: str | None) -> CheckInResponse | None:
    if key is None:
        return None
    if not key.strip() or len(key) > 100:
        raise HTTPException(400, "invalid Idempotency-Key")
    row = s.exec(select(CheckIn).where(CheckIn.patient_id == pid, CheckIn.request_key == key)).first()
    if row is None:
        return None
    if row.response_json:
        return CheckInResponse.model_validate_json(row.response_json)
    raise HTTPException(409, "check-in is still being saved; try again shortly")


def _now() -> datetime:
    """Current time from the shared clock in baseline.py (tests replace it to advance days)."""
    return baseline.clock()


def _rolling_history(
    s: Session, patient_id: int, before: datetime, slot: Slot, slopes: dict[str, tuple[float, float]] | None = None,
) -> dict[str, list[float]]:
    """Feature values over the last `baseline_days` days. B30 (time_of_day = "hour"): all check-ins,
    each adjusted for its hour with the active baseline's slopes. Old "slots" mode: same slot only (T10)."""
    since = before - timedelta(days=settings.baseline_days)
    q = (select(Feature.name, Feature.value, CheckIn.local_hour)
         .join(CheckIn, col(Feature.checkin_id) == CheckIn.id)
         .where(CheckIn.patient_id == patient_id, CheckIn.ts >= since, CheckIn.ts < before))
    if settings.time_of_day != "hour":
        q = q.where(CheckIn.slot == slot)
    hist: dict[str, list[float]] = defaultdict(list)
    for name, v, hour in s.exec(q).all():
        hist[name].append(state.adjust_for_hour({name: v}, hour, slopes or {})[name])
    return hist


def _day_lists(raw: str | None) -> dict[str, dict[str, list[float]]]:
    """Today's per-check-in deviations {track: {branch: [D, ...]}}; old rows stored one float per branch."""
    out: dict[str, dict[str, list[float]]] = {"anchored": {}, "rolling": {}}
    for t, branches in (json.loads(raw) if raw else {}).items():
        if t in out and isinstance(branches, dict):
            out[t] = {b: (v if isinstance(v, list) else [v]) for b, v in branches.items()}
    return out


def _day_weights(raw: str | None, day: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    """Per-check-in weights stored next to the day lists under "w" (B31); older rows: all 1 (B16)."""
    stored = (json.loads(raw) if raw else {}).get("w", {})
    return {t: {b: list(stored.get(t, {}).get(b, [1.0] * len(v))) for b, v in day[t].items()} for t in day}


def _score_tracks(
    s: Session, patient_id: int, now: datetime, slot: Slot, today_by_branch: dict[str, dict[str, float]],
    active, hour: float | None = None,
) -> dict:
    """Both tracks: vs the fixed healthy baseline (slow drift) and vs the last 14 days (abrupt change).
    Today's values are adjusted for the hour of the check-in first (B30)."""
    anchor_stats = baseline.stats_for(s, active, slot.value)
    slopes = baseline.hour_slopes(s, active)
    hist = _rolling_history(s, patient_id, now, slot, slopes)
    today_by_branch = {b: state.adjust_for_hour(v, hour, slopes) for b, v in today_by_branch.items()}
    k = settings.sd_prior_strength
    D = {"anchored": {}, "rolling": {}}
    Z = {"anchored": {}, "rolling": {}}
    shares = {"anchored": {}, "rolling": {}}
    used_stats = {"anchored": {}, "rolling": {}}
    for branch, vals in today_by_branch.items():
        model = state.load_branch(branch)
        tracks = {
            "anchored": {f: anchor_stats[f] for f in model.direction if f in anchor_stats},
            "rolling": state.history_stats(hist, model, k, settings.rolling_min_n),
        }
        for track, stats in tracks.items():
            used_stats[track][branch] = stats
            z = state.zscores(vals, stats, model.direction, model.meta.get("z_cap"))
            if z:
                Z[track][branch] = z
                shares[track][branch] = state.feature_shares(z, model)
                D[track][branch] = round(sum(shares[track][branch].values()), 4)
    return {"D": D, "Z": Z, "shares": shares, "stats": used_stats,
            "adjusted": today_by_branch, "hour_slopes": slopes}


def _todays_intervention(s: Session, patient_id: int, today: date, top: list[dict] | None = None) -> InterventionOut:
    arms = s.exec(select(Arm).where(Arm.patient_id == patient_id)).all()
    labels = {a.key: a.label for a in arms}
    log = s.exec(
        select(InterventionLog).where(InterventionLog.patient_id == patient_id, InterventionLog.date == today)
    ).first()
    if log is None:
        fit = signals.fitting_arms(top, [a.key for a in arms])
        arm = bandit.choose_arm([a for a in arms if a.key in fit])
        log = InterventionLog(patient_id=patient_id, date=today, arm_key=arm.key)
        s.add(log)
    return InterventionOut(arm_key=log.arm_key, label=labels.get(log.arm_key, log.arm_key))


async def _to_temp(upload: UploadFile, limit: int = MAX_AUDIO_BYTES) -> str:
    content_type = (upload.content_type or "application/octet-stream").split(";")[0]
    if not (content_type.startswith("audio/") or content_type in ("application/octet-stream", "video/webm")):
        raise HTTPException(415, "Upload an audio recording.")
    payload = await upload.read(limit + 1)
    if not payload:
        raise HTTPException(400, "The recording is empty. Try recording it again.")
    if len(payload) > limit:
        raise HTTPException(413, "The recording is too large. Use a shorter recording.")
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload)
        return tmp.name


@router.post("/checkins", response_model=CheckInResponse)
async def create_checkin(
    audio: UploadFile,
    patient_id: int = Form(...),
    slot: Slot = Form(...),
    phrase: UploadFile | None = None,
    request_key: str | None = Header(default=None, alias="Idempotency-Key"),
    s: Session = Depends(get_session),
) -> CheckInResponse:
    """Spoken check-in. `audio` = the free-talk memo (30 s). `phrase` (optional, #21) = the fixed
    reading phrase, recorded as its own file. Old clients send only `audio` and keep working."""
    patient = s.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(404, "patient not found")
    saved = _saved_response(s, patient_id, request_key)
    if saved is not None:
        return saved

    # Raw audio lives only in temp files and is deleted right after feature extraction.
    paths = [await _to_temp(audio)]
    try:
        if phrase is not None:
            try:
                paths.append(await _to_temp(phrase, limit=4 * 1024 * 1024))
            except HTTPException:
                log.info("Optional phrase upload was skipped; keeping the free memo.")
        with ThreadPoolExecutor(max_workers=1) as pool:
            # The phrase transcription (network) runs while the memo is analysed
            phrase_text = pool.submit(transcribe_phrase, paths[1]) if len(paths) > 1 else None
            result = extract_features(paths[0])
            if phrase_text is not None:
                try:
                    phrase_transcript = phrase_text.result()
                except Exception:
                    phrase_transcript = None
                    log.info("Phrase transcription unavailable; using the timing check.")
                try:
                    add_phrase(result, paths[1], phrase_transcript)
                except Exception:
                    log.exception("Optional phrase analysis failed; keeping the free memo.")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        for p in paths:
            os.remove(p)
    return _record_checkin(s, patient, slot, result, request_key=request_key)


@router.post("/checkins/text", response_model=CheckInResponse)
def create_text_checkin(body: TextCheckInIn, s: Session = Depends(get_session),
                       request_key: str | None = Header(default=None, alias="Idempotency-Key")) -> CheckInResponse:
    """Written diary entry -> text branch only (no audio). English; 50 to 100 words works best."""
    patient = s.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(404, "patient not found")
    if not body.text.strip():
        raise HTTPException(400, "empty diary entry")
    saved = _saved_response(s, body.patient_id, request_key)
    if saved is not None:
        return saved
    return _record_checkin(s, patient, body.slot, extract_text_features(body.text), request_key=request_key)


@router.post("/checkins/wearable", response_model=CheckInResponse)
def create_wearable_checkin(body: WearableDayIn, s: Session = Depends(get_session)) -> CheckInResponse:
    """One completed wearable day; no diary or daily prompt is answered by an automatic sync."""
    patient = s.get(Patient, body.patient_id)
    if patient is None:
        raise HTTPException(404, "patient not found")
    try:
        result = extract_wearable_features(**body.model_dump(exclude={"patient_id", "slot"}))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not result.features:
        raise HTTPException(400, "no wearable values in the request")
    extras = ({"active_minutes": body.active_minutes} if body.active_minutes is not None else
              {"active_minutes": sum(1 for m in body.minute_steps if m >= 100)} if body.minute_steps else {})
    return _record_checkin(s, patient, body.slot, result, extras)


def _record_checkin(s: Session, patient: Patient, slot: Slot, result: PipelineResult,
                    extras: dict[str, float] | None = None,
                    recorded_at: datetime | None = None,
                    request_key: str | None = None,
                    data_origin: str = "live") -> CheckInResponse:
    """Store features, run the state engine, escalate, pick today's intervention."""
    patient_id = patient.id
    # Serialize state updates for a person. A retried request cannot append another
    # B31 contribution or pay another reward after its original HTTP response was lost.
    s.execute(update(Patient).where(Patient.id == patient_id).values(name=Patient.name))
    s.refresh(patient)
    saved = _saved_response(s, patient_id, request_key)
    if saved is not None:
        s.commit()
        return saved
    now = recorded_at or _now()
    if data_origin not in ("live", "synthetic"):
        raise ValueError("data_origin must be live or synthetic")
    from .. import clinical
    previous_score = s.exec(select(StateScore).where(
        StateScore.patient_id == patient_id, StateScore.date <= now.date())
        .order_by(col(StateScore.date).desc())).first()
    before = clinical.score_point(previous_score)
    checkin = CheckIn(
        patient_id=patient_id, ts=now, slot=slot,
        transcript=result.transcript, llm_json=result.llm_json, risk_flag=result.crisis,
        speech_s=result.speech_s, request_key=request_key, data_origin=data_origin,
    )
    s.add(checkin)
    s.flush()
    for name, value, source in result.features:
        s.add(Feature(checkin_id=checkin.id, name=name, value=value, source=source))
    # Daily prompt (#22): local time of every check-in; the first one after the prompt fires answers it
    memo_features = [(n, v, src) for n, v, src in result.features
                     if src != FeatureSource.wearable and not n.startswith("phrase_")]
    prompt_out = prompt.answer(s, patient_id, checkin, now) if memo_features else None

    # Healthy-baseline lifecycle (STATUS.md T1): no score until a baseline is locked
    active = baseline.latest(s, patient_id, BaselineStatus.active)
    building = baseline.latest(s, patient_id, BaselineStatus.building)
    if active is None and building is None:
        building = baseline.start(s, patient_id, now=now)
    if building is not None and baseline.try_lock(s, building, now):
        active, building = building, None

    today_by_branch: dict[str, dict[str, float]] = defaultdict(dict)
    for name, v, source in result.features:
        if not name.startswith("phrase_") and name != "active_minutes":
            today_by_branch[state.BRANCH_OF_SOURCE[source]][name] = v

    S_val: float | None = None
    elevated = False
    S_track: dict[str, float | None] = {"anchored": None, "rolling": None}
    driver: str | None = None
    branch_D: dict[str, float] = {}
    top_now: list[dict] | None = None
    tr = {"D": {"anchored": {}, "rolling": {}}, "Z": {"anchored": {}, "rolling": {}},
          "shares": {"anchored": {}, "rolling": {}}, "stats": {"anchored": {}, "rolling": {}},
          "adjusted": {}, "hour_slopes": {}}
    entry_weights: dict[str, float] = {}
    score = None
    if active is not None:
        tr = _score_tracks(s, patient_id, now, slot, today_by_branch, active, checkin.local_hour)
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
        # B31: the check-in that answers today's prompt, or the day's first one in a branch, counts fully;
        # extras count at half weight (same-day noise is partly shared). Shares are weighted the same way,
        # so the explanation still sums exactly to S (B20).
        raw = score.branch_scores_json if score else None
        day = _day_lists(raw)
        wts = _day_weights(raw, day)
        evidence = json.loads(score.evidence_json) if score and score.evidence_json else {}
        prompted = checkin.prompt_at is not None
        for branch in {b for t in day for b in tr["D"][t]}:
            w = state.checkin_weight(prompted, max(len(day[t].get(branch, [])) for t in day))
            entry_weights[branch] = w
            for t in day:
                if branch not in tr["D"][t]:
                    continue
                day[t].setdefault(branch, []).append(tr["D"][t][branch])
                wts[t].setdefault(branch, []).append(w)
                sums = evidence.setdefault(t, {}).setdefault(branch, {})
                for f, v in tr["shares"][t][branch].items():
                    sums[f] = round(sums.get(f, 0.0) + w * v, 5)
        day_branch = {t: {b: round(state.combine_checkins(v, wts[t][b]), 4) for b, v in day[t].items()} for t in day}
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
            score.branch_scores_json = json.dumps({**day, "w": wts})
            score.evidence_json = json.dumps(evidence)
            recent_anch = [x.S_anchored for x in s.exec(
                select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date < now.date())
                .order_by(col(StateScore.date))
            ).all() if x.S_anchored is not None][-(settings.elevated_days - 1):]
            if S_track["anchored"] is not None:
                recent_anch.append(S_track["anchored"])
            score.elevated = escalation.elevated_flag(recent_anch, settings.elevated_S, settings.elevated_days)
            elevated = score.elevated
            top_now = json.loads(score.top_features_json or "[]")
            s.add(score)

    recent = [x.S for x in s.exec(
        select(StateScore).where(StateScore.patient_id == patient_id, StateScore.date <= now.date())
        .order_by(col(StateScore.date))
    ).all()]
    if S_val is not None and (not recent or recent[-1] != S_val):
        recent.append(S_val)
    level = escalation.evaluate_level(recent[-7:], result.crisis)
    if level > 0 and level != patient.risk_level:
        recipient = ContactKind.friend if level == 1 else ContactKind.clinician
        s.add(Escalation(patient_id=patient_id, ts=now, level=level, recipient=recipient,
                         reason="crisis flag" if result.crisis else f"S={S_val} ({driver})"))
    patient.risk_level = level
    s.add(patient)
    if score is not None and S_val is not None:
        score.level = level
        s.add(score)

    intervention = _todays_intervention(s, patient_id, now.date(), top_now)
    earned = [{"reason": r.reason, "label": rewards.LABELS.get(r.reason, r.reason), "amount": r.amount}
              for r in rewards.sync(s, patient_id, now)]
    if any(src == FeatureSource.wearable for _, _, src in result.features):
        from .. import appie
        values = {**{n: v for n, v, _ in result.features}, **(extras or {})}
        earned.extend(appie.on_wearable(s, patient_id, now, values))
    s.flush()
    why = explanation.latest_explanation(s, patient_id) if S_val is not None else None
    final_score = s.exec(select(StateScore).where(
        StateScore.patient_id == patient_id, StateScore.date <= now.date())
        .order_by(col(StateScore.date).desc())).first()
    analysis = clinical.capture_analysis(s, checkin, result.features, active, tr, before,
                                         clinical.score_point(final_score), entry_weights,
                                         phrase=result.phrase_check, prompt=prompt_out)
    checkin.analysis_json = json.dumps(analysis, allow_nan=False)
    s.add(checkin)

    response = CheckInResponse(
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
        prompt=prompt_out,
        phrase=result.phrase_check,
        rewards=earned,
        analysis=analysis,
    )
    if request_key is not None:
        checkin.response_json = response.model_dump_json()
        s.add(checkin)
    s.commit()
    return response
