"""Read-side views for the two apps, built from what the engine stores. No scoring happens here.

Patient app:       today(), insights(), contacts()
Clinician portal:  notifications(), timeline(), explain()

Everything is relative to the patient's own healthy baseline (baseline.py). The clinician sees which
signals changed; the patient sees short, calm sentences and never a number like a score.
"""

import json
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta

from sqlmodel import Session, col, select

from . import baseline, explanation
from .engine import signals, state
from .models import (
    Arm, BaselineStatus, CheckIn, Contact, Escalation, Feature, FeatureSource, InterventionLog, Patient, StateScore,
)
from .pipeline import text as tp

PERSONAL = {FeatureSource.prosody, FeatureSource.emotion, FeatureSource.text, FeatureSource.llm}
STATUS_TEXT = {
    "building": "Learning your usual.",
    "close": "You're close to your usual rhythm.",
    "different": "This week is a bit different from your usual.",
    "care": "Things look harder than usual. Your care team may reach out.",
}


def status_of(S: float | None) -> str:
    if S is None:
        return "building"
    return "close" if S < 1 else "different" if S < 2 else "care"


def _scores(s: Session, pid: int) -> list[StateScore]:
    return list(s.exec(select(StateScore).where(StateScore.patient_id == pid).order_by(col(StateScore.date))).all())


def _top(score: StateScore | None) -> list[dict]:
    return json.loads(score.top_features_json or "[]") if score else []


def _baseline(s: Session, pid: int):
    return baseline.latest(s, pid, BaselineStatus.active), baseline.latest(s, pid, BaselineStatus.building)


def _personal_checkins(s: Session, pid: int, since: datetime | None = None, until: datetime | None = None):
    """Voice and written check-ins (not wearable syncs), oldest first, with their features."""
    q = select(CheckIn).where(CheckIn.patient_id == pid)
    if since is not None:
        q = q.where(CheckIn.ts >= since)
    if until is not None:
        q = q.where(CheckIn.ts <= until)
    out = []
    for c in s.exec(q.order_by(col(CheckIn.ts))).all():
        feats = s.exec(select(Feature).where(Feature.checkin_id == c.id)).all()
        if any(f.source in PERSONAL for f in feats):
            out.append((c, {f.name: f.value for f in feats}))
    return out


def _real_text(c: CheckIn) -> str | None:
    t = (c.transcript or "").strip()
    return None if not t or t.startswith("(stub)") else t


def _tone(text_logit: float | None, stats: dict) -> str | None:
    """Steady / Quieter / Lifted for one check-in, from its language score against the baseline."""
    if text_logit is None or "text_logit" not in stats:
        return None
    center, scale, _ = stats["text_logit"]
    z = (text_logit - center) / scale if scale else 0.0
    return "Quieter" if z >= 1 else "Lifted" if z <= -1 else "Steady"


def _phrases(top: list[dict], n: int = 2) -> list[str]:
    out: list[str] = []
    for t in top:
        p = signals.WORSE.get(t["feature"])
        if t.get("contribution", 0) > 0.3 and p and p not in out:
            out.append(p)
        if len(out) == n:
            break
    return out


def _sentence(phrases: list[str]) -> str:
    if not phrases:
        return ""
    s = ", ".join(phrases)
    return s[0].upper() + s[1:] + "."


# ------------------------------------------------------------------------------------ patient app

def today(s: Session, pid: int) -> dict:
    from .routers.checkins import _todays_intervention  # local import: routers import this module's siblings

    p = s.get(Patient, pid)
    now = baseline.clock()
    day = now.date()
    scores = _scores(s, pid)
    active, building = _baseline(s, pid)
    b = active or building
    last = scores[-1] if scores else None
    S = last.S if (last and active) else None

    log = s.exec(select(InterventionLog).where(InterventionLog.patient_id == pid, InterventionLog.date == day)).first()
    if log is None and active is not None:
        _todays_intervention(s, pid, day, _top(last))
        s.commit()
        log = s.exec(select(InterventionLog).where(InterventionLog.patient_id == pid, InterventionLog.date == day)).first()
    labels = {a.key: a.label for a in s.exec(select(Arm).where(Arm.patient_id == pid)).all()}
    action = None
    if log is not None:
        score_then = next((x for x in reversed(scores) if x.date <= log.date), None)
        action = {"date": log.date.isoformat(), "arm_key": log.arm_key, "label": labels.get(log.arm_key, log.arm_key),
                  "reason": signals.reason_for(_top(score_then)), "completed": log.completed, "rating": log.rating}

    pending = s.exec(
        select(InterventionLog).where(InterventionLog.patient_id == pid, col(InterventionLog.completed).is_(None),
                                      InterventionLog.date >= day - timedelta(days=1))
        .order_by(col(InterventionLog.date).desc())
    ).first()
    last_sync = s.exec(
        select(CheckIn.ts).join(Feature, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == pid, Feature.source == FeatureSource.wearable)
        .order_by(col(CheckIn.ts).desc())
    ).first()
    status = status_of(S)
    return {
        "patient": {"id": pid, "name": p.name if p else ""},
        "date": day.isoformat(),
        "status": status,
        "status_text": STATUS_TEXT[status],
        "baseline": baseline.progress(s, b, now) if b else None,
        "series": [{"date": x.date.isoformat(), "S": x.S} for x in scores if x.date > day - timedelta(days=14)],
        "action": action,
        "pending_followup": None if pending is None else {
            "date": pending.date.isoformat(), "arm_key": pending.arm_key, "label": labels.get(pending.arm_key, pending.arm_key)},
        "last_sync": last_sync.isoformat() if last_sync else None,
    }


def insights(s: Session, pid: int) -> list[dict]:
    """What helps: every action with how often it was done and helped, ranked by what the bandit learned."""
    arms = s.exec(select(Arm).where(Arm.patient_id == pid)).all()
    logs = s.exec(select(InterventionLog).where(InterventionLog.patient_id == pid).order_by(col(InterventionLog.date))).all()
    fits_when = defaultdict(list)
    for f, keys in signals.FITS.items():
        for k in keys:
            w = signals.WORSE.get(f)
            if w and w not in fits_when[k]:
                fits_when[k].append(w)
    out = []
    for a in arms:
        mine = [l for l in logs if l.arm_key == a.key]
        done = [l for l in mine if l.completed]
        ratings = [l.rating for l in done if l.rating]
        out.append({
            "key": a.key, "label": a.label,
            "suggested": len(mine), "done": len(done), "skipped": sum(l.completed is False for l in mine),
            "tried": sum(l.completed is not None for l in mine),  # answered: done or skipped (what the bandit counts)
            "helped": sum(r >= 4 for r in ratings), "mean_rating": round(statistics.fmean(ratings), 1) if ratings else None,
            "learned": round(a.a / (a.a + a.b), 3),
            "fits_when": fits_when.get(a.key, []),
            "history": [{"date": l.date.isoformat(), "completed": l.completed, "rating": l.rating} for l in mine[-10:]],
        })
    return sorted(out, key=lambda x: -x["learned"])


def contacts(s: Session, pid: int) -> list[dict]:
    return [{"id": c.id, "name": c.name, "kind": c.kind.value}
            for c in s.exec(select(Contact).where(Contact.patient_id == pid)).all()]


# ------------------------------------------------------------------------------------ clinician portal

def notifications(s: Session, status: str = "open") -> list[dict]:
    """Inbox: escalations (friends nudged, clinic flag, crisis) and baselines waiting for confirmation."""
    names = {p.id: p.name for p in s.exec(select(Patient)).all()}
    items = []
    for e in s.exec(select(Escalation).order_by(col(Escalation.ts).desc())).all():
        day = e.ts.date()
        score = s.exec(select(StateScore).where(StateScore.patient_id == e.patient_id, StateScore.date <= day)
                       .order_by(col(StateScore.date).desc())).first()
        why = _sentence(_phrases(_top(score)))
        if e.level >= 3:
            kind, msg = "crisis", "A check-in mentioned self-harm."
            c = s.exec(select(CheckIn).where(CheckIn.patient_id == e.patient_id, CheckIn.risk_flag == True)  # noqa: E712
                       .order_by(col(CheckIn.ts).desc())).first()
            quote = _real_text(c) if c else None
        elif e.level == 2:
            kind, msg, quote = "clinic", f"Harder days than usual. {why}".strip(), None
        else:
            kind, msg, quote = "friends", f"A bit different from usual for 3 days, friends were nudged. {why}".strip(), None
        items.append({"id": str(e.id), "patient_id": e.patient_id, "patient": names.get(e.patient_id, "?"), "kind": kind,
                      "level": e.level, "message": msg, "quote": quote, "ts": e.ts.isoformat(), "handled": e.handled})
    for pid, name in names.items():
        b = baseline.latest(s, pid, BaselineStatus.active)
        if b and not b.confirmed_by:
            items.append({"id": f"baseline-{b.id}", "patient_id": pid, "patient": name, "kind": "baseline", "level": 0,
                          "message": "Healthy baseline is ready to confirm.", "quote": None,
                          "ts": (b.locked_at or b.started_at).isoformat(), "handled": False})
    # A level can go up and down (2 -> 1 -> 2), which logs the same alert twice: keep the newest per
    # patient, kind and handled-state. Crisis flags are never collapsed.
    seen, unique = set(), []
    for i in sorted(items, key=lambda i: i["ts"], reverse=True):
        key = (i["patient_id"], i["kind"], i["handled"])
        if i["kind"] != "crisis" and key in seen:
            continue
        seen.add(key)
        unique.append(i)
    items = unique
    if status == "open":
        items = [i for i in items if not i["handled"]]
    elif status == "handled":
        items = [i for i in items if i["handled"]]
    rank = {"crisis": 0, "clinic": 1, "friends": 2, "baseline": 3}
    items.sort(key=lambda i: i["ts"], reverse=True)
    items.sort(key=lambda i: (i["handled"], rank[i["kind"]] if i["kind"] == "crisis" else 9))
    return items


def mark_handled(s: Session, notification_id: str) -> bool:
    if not notification_id.isdigit():
        return False
    e = s.get(Escalation, int(notification_id))
    if e is None:
        return False
    e.handled, e.handled_at = True, baseline.clock()
    s.add(e)
    s.commit()
    return True


def timeline(s: Session, pid: int, days: int) -> dict:
    """Day-by-day view for the clinician: score, check-ins, actions, and the patient's own words."""
    p = s.get(Patient, pid)
    now = baseline.clock()
    start = (now - timedelta(days=days - 1)).date()
    active, building = _baseline(s, pid)
    stats = baseline.stats_for(s, active, "all") if active else {}
    scores = {x.date: x for x in _scores(s, pid) if x.date >= start}
    esc = defaultdict(int)
    for e in s.exec(select(Escalation).where(Escalation.patient_id == pid)).all():
        esc[e.ts.date()] = max(esc[e.ts.date()], e.level)
    logs = {l.date: l for l in s.exec(select(InterventionLog).where(InterventionLog.patient_id == pid)).all()}
    since = datetime(start.year, start.month, start.day, tzinfo=now.tzinfo)
    personal = _personal_checkins(s, pid, since=since)
    checkin_days = {c.ts.date() for c, _ in personal}
    rows = []
    d = start
    while d <= now.date():
        sc, lg = scores.get(d), logs.get(d)
        rows.append({"date": d.isoformat(), "S": sc.S if sc else None, "S_anchored": sc.S_anchored if sc else None,
                     "S_rolling": sc.S_rolling if sc else None, "level": esc.get(d, 0), "checkin": d in checkin_days,
                     "action_done": bool(lg and lg.completed), "rating": lg.rating if lg else None})
        d += timedelta(days=1)
    feelings = []
    for c, f in reversed(personal):
        t = _real_text(c)
        if t:
            feelings.append({"ts": c.ts.isoformat(), "slot": c.slot.value, "text": t, "tone": _tone(f.get("text_logit"), stats),
                             "crisis": c.risk_flag})
        if len(feelings) == 6:
            break
    first = baseline.latest(s, pid, BaselineStatus.retired) or active or building
    b = active or building
    last_score = _scores(s, pid)[-1:] or [None]
    return {
        "patient": {"id": pid, "name": p.name if p else "", "level": p.risk_level if p else 0,
                    "status": status_of(last_score[0].S if (last_score[0] and active) else None),
                    "days_since_start": (now - baseline._aware(first.started_at)).days if first else None},
        "days": rows,
        "feelings": feelings,
        "baseline": baseline.progress(s, b, now) if b else None,
        "what_helps": insights(s, pid)[:4],
    }


def _period_values(s: Session, pid: int, since: datetime) -> dict[str, list[tuple[date, float]]]:
    rows = s.exec(
        select(CheckIn.ts, Feature.name, Feature.value)
        .join(Feature, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == pid, CheckIn.ts >= since)
    ).all()
    out: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for ts, name, v in rows:
        out[name].append((ts.date(), v))
    return out


def explain(s: Session, pid: int, days: int) -> dict:
    """'Why is this changing?': every signal in the period against the healthy baseline, grouped by source."""
    active, building = _baseline(s, pid)
    if active is None:
        return {"available": False, "message": "The healthy baseline is still being built."}
    now = baseline.clock()
    since = now - timedelta(days=days)
    stats = baseline.stats_for(s, active, "all")
    values = _period_values(s, pid, since)

    # 1. Each signal: usual (baseline median) vs now (period median), direction-adjusted z
    groups: dict[str, list[dict]] = {"health": [], "acoustic": [], "text": []}
    for f, (src, name, tech, fmt) in signals.SIGNALS.items():
        if src not in groups:
            continue
        model = state.load_branch(src)
        if model.weights.get(f, 0) == 0 and f in model.weights:
            continue  # zero-weight features (e.g. night_activity) are not part of the score
        vals = values.get(f, [])
        row = {"feature": f, "name": name, "technical": tech, "format": fmt}
        if f not in stats or len(vals) < 2:
            row.update({"status": "no_data"})
        else:
            center, scale, _ = stats[f]
            now_v = statistics.median(v for _, v in vals)
            z = model.direction.get(f, 1) * (now_v - center) / scale if scale else 0.0
            by_day = defaultdict(list)
            for d, v in vals:
                by_day[d].append(v)
            row.update({"usual": center, "now": now_v, "z": round(z, 2),
                        "status": "worse" if z > 0.5 else "better" if z < -0.5 else "same",
                        "series": [round(statistics.median(by_day[d]), 4) for d in sorted(by_day)]})
        groups[src].append(row)

    # 2. How much each source moved the score: mean daily branch deviation on the anchored track
    per_branch = defaultdict(list)
    for sc in _scores(s, pid):
        if sc.date < since.date():
            continue
        day = json.loads(sc.branch_scores_json or "{}").get("anchored", {})
        for br, v in day.items():
            per_branch[br].append(state.combine_checkins(v if isinstance(v, list) else [v]))
    mean_d = {br: statistics.fmean(v) for br, v in per_branch.items() if v}
    total = sum(abs(v) for v in mean_d.values()) or 1.0
    sources = []
    for br in ("health", "acoustic", "text"):
        rows = groups[br]
        worse = [r for r in rows if r.get("status") == "worse"]
        better = [r for r in rows if r.get("status") == "better"]
        d = mean_d.get(br)
        sources.append({
            "source": br, "label": signals.SOURCE_LABEL[br],
            "mean_deviation": None if d is None else round(d, 2),
            "share": 0.0 if d is None else round(abs(d) / total, 3),
            "direction": None if d is None else ("worse" if d > 0.3 else "better" if d < -0.3 else "same"),
            "summary": _source_summary(worse, better, d),
            "signals": sorted(rows, key=lambda r: -(abs(r.get("z", 0)))),
        })

    # 3. Words that stood out and the most telling entry
    personal = _personal_checkins(s, pid, since=since)
    weights, last_seen, best = defaultdict(float), {}, None
    for c, f in personal:
        t = _real_text(c)
        if not t:
            continue
        r = tp.analyze_text(t, explain_top=10)
        for w, v in r["words_up"]:
            if _telling(w):
                weights[w] += v
                last_seen[w] = c.ts.date().isoformat()
        if best is None or r["features"]["text_logit"] > best[1]["features"]["text_logit"]:
            best = (c, r, t)
    stand_out = [{"word": w, "date": last_seen[w]} for w, _ in sorted(weights.items(), key=lambda x: -x[1])[:8]]
    quote = None
    if best:
        c, r, t = best
        quote = {"date": c.ts.date().isoformat(), "text": t if len(t) <= 260 else t[:257] + "...",
                 "highlights": [w for w, _ in r["words_up"] if _telling(w)][:5]}

    # 4. Check-in habits (shown, not part of the score)
    base_ci = _personal_checkins(s, pid, since=active.started_at, until=active.locked_at)
    locked = baseline._aware(active.locked_at) if active.locked_at else now
    base_days = max(1.0, (locked - baseline._aware(active.started_at)).total_seconds() / 86400)
    words_of = lambda cis: [len(_real_text(c).split()) for c, _ in cis if _real_text(c)]  # noqa: E731
    days_with = {c.ts.date() for c, _ in personal}
    missed, run = 0, 0
    for k in range(days):
        if (now - timedelta(days=k)).date() in days_with:
            break
        run += 1
    missed = run
    habits = {
        "per_week_usual": round(len(base_ci) / base_days * 7, 1),
        "per_week_now": round(len(personal) / days * 7, 1),
        "words_usual": round(statistics.fmean(words_of(base_ci)), 0) if words_of(base_ci) else None,
        "words_now": round(statistics.fmean(words_of(personal)), 0) if words_of(personal) else None,
        "days_since_last_checkin": missed,
    }

    wear_days = len({d for d, _ in values.get("steps", []) + values.get("sleep_hours", [])})
    summary = _overall_summary(sources)
    return {
        "available": True, "days": days, "summary": summary,
        "data_note": f"Based on {wear_days} wearable days and {len(personal)} check-ins in this period.",
        "sources": sources, "words": {"stand_out": stand_out, "quote": quote}, "habits": habits,
        "exact_explanation": explanation.latest_explanation(s, pid, window=days),
    }


def _overall_summary(sources: list[dict]) -> str:
    """'Mostly words and activity & sleep, both worse than usual. Voice close to usual.'"""
    name = lambda x: x["label"].lower()  # noqa: E731
    moved = [x for x in sorted(sources, key=lambda x: -x["share"]) if x["direction"] in ("worse", "better")]
    same = [name(x) for x in sources if x["direction"] == "same"]
    if not moved:
        return "All sources are close to this patient's usual."
    top = moved[:2]
    if len(top) == 2 and top[0]["direction"] == top[1]["direction"]:
        out = f"Mostly {name(top[0])} and {name(top[1])}, both {top[0]['direction']} than usual."
    else:
        out = "Mostly " + " and ".join(f"{name(x)} ({x['direction']} than usual)" for x in top) + "."
    rest = [x for x in moved[2:]]
    if rest:
        out += " " + "; ".join(f"{name(x).capitalize()} a little {x['direction']}" for x in rest) + "."
    if same:
        out += f" {same[0].capitalize()} close to usual."
    return out


# Words that move the model but tell a clinician nothing
_STOP = set("""i me my a an the and or but so to of in on at for with from by as is am are was were be been being have has
had do did does doing it its this that these those there here then than just really very also too today yesterday tomorrow
day days get got go went going make made one some any all about up out off over into again still even""".split())


def _telling(word: str) -> bool:
    """A single word the model weighs strongly towards low mood (coefficient >= 2). Drops function words
    and weakly weighted everyday words ('answered', 'took') that tell a clinician nothing."""
    if " " in word or word in _STOP:
        return False
    m = tp._model()
    j = m["index"].get(word)
    return j is not None and m["coef"][j] >= 2.0


def _source_summary(worse: list[dict], better: list[dict], d: float | None) -> str:
    """One line per source that agrees with the source's overall direction."""
    if d is None:
        return "Not enough data in this period."
    w = [signals.WORSE.get(r["feature"], r["name"].lower()) for r in worse[:3]]
    b = [r["name"].lower() for r in better[:2]]
    if d > 0.3:
        return _sentence(w) if w else "A little worse overall, no single signal stands out."
    if d < -0.3:
        return ("Better than usual: " + ", ".join(b) + ".") if b else "A little better overall, no single signal stands out."
    if w and b:
        return f"Mixed: {', '.join(w)}, but {', '.join(b)} better."
    return ("Mostly close to usual. " + _sentence(w)) if w else "Close to usual."
