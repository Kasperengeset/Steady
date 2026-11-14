"""Fixed "healthy" baseline after TMS (STATUS.md T1).

Lifecycle:  building ──(period over + enough check-ins)──▶ active ──(new one locks)──▶ retired
                 └──────────── clinician rejects ────────────▶ rejected (a new one starts)

- The first check-in of a patient without a baseline starts one automatically.
- While a newer baseline is building, the previous active one keeps being used for scoring.
- A baseline is never updated automatically – only a clinician (or a device change they approve)
  starts a new one. Otherwise a slow relapse would leak into the reference ("boiling frog").
"""

from collections import defaultdict
from datetime import datetime, timezone

from sqlmodel import Session, col, select

from .config import settings
from .engine import state
from .models import Baseline, BaselineStat, BaselineStatus, CheckIn, Feature

SLOTS = ("morning", "evening")
FEATURE_BRANCH = {f: b for b, dirs in state.DIRECTION.items() for f in dirs}


def _aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo; everything is stored in UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def latest(s: Session, patient_id: int, status: BaselineStatus) -> Baseline | None:
    return s.exec(
        select(Baseline).where(Baseline.patient_id == patient_id, Baseline.status == status)
        .order_by(col(Baseline.version).desc())
    ).first()


def start(
    s: Session, patient_id: int, *, period_days: int | None = None, min_checkins: int | None = None,
    note: str | None = None, now: datetime | None = None,
) -> Baseline:
    """Start a new building baseline (any previous building one is dropped as rejected)."""
    for old in s.exec(select(Baseline).where(Baseline.patient_id == patient_id, Baseline.status == BaselineStatus.building)):
        old.status = BaselineStatus.rejected
        s.add(old)
    last = s.exec(select(Baseline).where(Baseline.patient_id == patient_id).order_by(col(Baseline.version).desc())).first()
    b = Baseline(
        patient_id=patient_id,
        version=(last.version + 1) if last else 1,
        started_at=now or datetime.now(timezone.utc),
        period_days=settings.healthy_period_days if period_days is None else period_days,
        min_checkins=settings.healthy_min_checkins if min_checkins is None else min_checkins,
        note=note,
    )
    s.add(b)
    s.flush()
    return b


def _period_values(s: Session, b: Baseline) -> dict[str, dict[str, list[float]]]:
    """slot -> feature -> values, from check-ins since the baseline started."""
    rows = s.exec(
        select(CheckIn.slot, Feature.name, Feature.value)
        .join(Feature, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == b.patient_id, CheckIn.ts >= b.started_at)
    ).all()
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for slot, name, v in rows:
        out[slot.value if hasattr(slot, "value") else slot][name].append(v)
        out["all"][name].append(v)
    return out


def _counts(s: Session, b: Baseline) -> dict[str, int]:
    rows = s.exec(select(CheckIn.slot).where(CheckIn.patient_id == b.patient_id, CheckIn.ts >= b.started_at)).all()
    counts = {slot: 0 for slot in SLOTS}
    for slot in rows:
        counts[slot.value if hasattr(slot, "value") else slot] += 1
    return counts


def progress(s: Session, b: Baseline, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    counts = _counts(s, b)
    days = (now - _aware(b.started_at)).total_seconds() / 86400
    return {
        "version": b.version,
        "status": b.status.value,
        "started_at": _aware(b.started_at).isoformat(),
        "locked_at": _aware(b.locked_at).isoformat() if b.locked_at else None,
        "period_days": b.period_days,
        "days_elapsed": round(min(days, b.period_days) if b.status == BaselineStatus.building else days, 1),
        "checkins": counts,
        "min_checkins": b.min_checkins,
        "confirmed_by": b.confirmed_by,
    }


def try_lock(s: Session, b: Baseline, now: datetime | None = None) -> bool:
    """Lock when the period is over and there are enough check-ins (in total). Slots with too few
    check-ins get no slot-specific stats and fall back to the pooled 'all' stats when scoring."""
    now = now or datetime.now(timezone.utc)
    if b.status != BaselineStatus.building:
        return False
    if (now - _aware(b.started_at)).total_seconds() < b.period_days * 86400:
        return False
    if sum(_counts(s, b).values()) < b.min_checkins:
        return False  # period over but too few check-ins → keep building (extended)

    k = settings.sd_prior_strength
    for slot, feats in _period_values(s, b).items():
        for f, vals in feats.items():
            if slot != "all" and len(vals) < b.min_checkins:
                continue
            branch = FEATURE_BRANCH.get(f)
            prior = state.prior_sd(state.load_branch(branch), f) if branch else None
            c, sc = state.robust_stats(vals, prior, k)
            s.add(BaselineStat(baseline_id=b.id, slot=slot, feature=f, center=c, scale=sc, n=len(vals)))

    prev = latest(s, b.patient_id, BaselineStatus.active)
    if prev:
        prev.status = BaselineStatus.retired
        s.add(prev)
    b.status, b.locked_at = BaselineStatus.active, now
    s.add(b)
    s.flush()
    return True


def stats_for(s: Session, b: Baseline, slot: str) -> dict[str, tuple[float, float, int]]:
    """feature -> (centre, scale, n) for this slot, falling back to pooled 'all' per feature."""
    rows = s.exec(select(BaselineStat).where(BaselineStat.baseline_id == b.id)).all()
    pooled = {r.feature: (r.center, r.scale, r.n) for r in rows if r.slot == "all"}
    specific = {r.feature: (r.center, r.scale, r.n) for r in rows if r.slot == slot}
    return {**pooled, **specific}


def reject(s: Session, b: Baseline, note: str | None = None, now: datetime | None = None) -> Baseline:
    """Clinician: this period was not representative of the healthy state → start over."""
    b.status = BaselineStatus.rejected
    b.note = note or b.note
    s.add(b)
    s.flush()
    return start(s, b.patient_id, period_days=b.period_days, min_checkins=b.min_checkins, now=now)
