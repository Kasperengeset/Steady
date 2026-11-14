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
HOUR_SLOT = "hour"  # BaselineStat rows holding the time-of-day slope (B30), not a baseline slot
FEATURE_BRANCH = {f: b for b, dirs in state.DIRECTION.items() for f in dirs}


def clock() -> datetime:
    """Single clock for the baseline lifecycle and check-ins (tests replace it to advance days)."""
    return datetime.now(timezone.utc)


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
        started_at=now or clock(),
        period_days=settings.healthy_period_days if period_days is None else period_days,
        min_checkins=settings.healthy_min_checkins if min_checkins is None else min_checkins,
        note=note,
    )
    s.add(b)
    s.flush()
    return b


def _period_values(s: Session, b: Baseline) -> dict[str, dict[str, list[tuple[float, float | None]]]]:
    """slot -> feature -> [(value, local hour)], from check-ins since the baseline started."""
    rows = s.exec(
        select(CheckIn.slot, CheckIn.local_hour, Feature.name, Feature.value)
        .join(Feature, col(Feature.checkin_id) == CheckIn.id)
        .where(CheckIn.patient_id == b.patient_id, CheckIn.ts >= b.started_at)
    ).all()
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    hour_mode = settings.time_of_day == "hour"
    for slot, hour, name, v in rows:
        if not hour_mode:  # B30: with the hour as a covariate there are no slot baselines
            out[slot.value if hasattr(slot, "value") else slot][name].append((v, hour))
        out["all"][name].append((v, hour))
    return out


def _counts(s: Session, b: Baseline) -> dict[str, int]:
    """Check-ins in the baseline period: since start, and only up to the lock for a locked baseline."""
    q = select(CheckIn.slot).where(CheckIn.patient_id == b.patient_id, CheckIn.ts >= b.started_at)
    if b.locked_at is not None:
        q = q.where(CheckIn.ts <= b.locked_at)
    rows = s.exec(q).all()
    counts = {slot: 0 for slot in SLOTS}
    for slot in rows:
        counts[slot.value if hasattr(slot, "value") else slot] += 1
    return counts


def progress(s: Session, b: Baseline, now: datetime | None = None) -> dict:
    now = now or clock()
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
    """Lock when the period is over, there are enough check-ins in total, and at least one feature has
    enough values (min(min_checkins, settings.min_feature_n)). Otherwise the period is extended.
    Slots with too few check-ins get no slot-specific stats and fall back to the pooled 'all' stats."""
    now = now or clock()
    if b.status != BaselineStatus.building:
        return False
    if (now - _aware(b.started_at)).total_seconds() < b.period_days * 86400:
        return False
    if sum(_counts(s, b).values()) < b.min_checkins:
        return False  # period over but too few check-ins → keep building (extended)

    k = settings.sd_prior_strength
    # Every feature needs a minimum number of values (text features have no corpus SD to lean on, so a
    # scale from 1–2 entries would be a guess). Capped so short demo baselines still work.
    min_n = min(b.min_checkins, settings.min_feature_n)
    rows = []
    for slot, feats in _period_values(s, b).items():
        for f, pairs in feats.items():
            if len(pairs) < (min_n if slot == "all" else b.min_checkins):
                continue
            vals = [v for v, _ in pairs]
            branch = FEATURE_BRANCH.get(f)
            if settings.time_of_day == "hour" and branch:
                # B30: fit the feature's time-of-day slope, then centre and scale the hour-adjusted values.
                # Stored as slot "hour": center = slope per hour, scale = reference hour, n = values with an hour.
                fit = state.hour_slope(vals, [h for _, h in pairs], k)
                if fit is not None:
                    slope, h0, n_h = fit
                    rows.append(BaselineStat(baseline_id=b.id, slot=HOUR_SLOT, feature=f, center=slope, scale=h0, n=n_h))
                    vals = [v - slope * (h - h0) if h is not None else v for v, h in pairs]
            prior = state.prior_sd(state.load_branch(branch), f) if branch else None
            c, sc = state.robust_stats(vals, prior, k)
            rows.append(BaselineStat(baseline_id=b.id, slot=slot, feature=f, center=c, scale=sc, n=len(vals)))
    if not rows:
        # E.g. 3 spoken + 3 written check-ins: no single feature has enough values. Locking would give
        # an active baseline that can never score (and it is never re-anchored automatically) → extend.
        return False
    s.add_all(rows)

    prev = latest(s, b.patient_id, BaselineStatus.active)
    if prev:
        prev.status = BaselineStatus.retired
        s.add(prev)
    b.status, b.locked_at = BaselineStatus.active, now
    s.add(b)
    s.flush()
    return True


def stats_for(s: Session, b: Baseline, slot: str) -> dict[str, tuple[float, float, int]]:
    """feature -> (centre, scale, n) for this slot, falling back to pooled 'all' per feature.
    With the hour as a covariate (B30) only the pooled stats are used."""
    rows = s.exec(select(BaselineStat).where(BaselineStat.baseline_id == b.id)).all()
    pooled = {r.feature: (r.center, r.scale, r.n) for r in rows if r.slot == "all"}
    if settings.time_of_day == "hour":
        return pooled
    specific = {r.feature: (r.center, r.scale, r.n) for r in rows if r.slot == slot}
    return {**pooled, **specific}


def hour_slopes(s: Session, b: Baseline) -> dict[str, tuple[float, float]]:
    """B30: feature -> (slope per hour, reference hour) fitted on this baseline; empty for old baselines."""
    if settings.time_of_day != "hour":
        return {}
    rows = s.exec(select(BaselineStat).where(BaselineStat.baseline_id == b.id, BaselineStat.slot == HOUR_SLOT)).all()
    return {r.feature: (r.center, r.scale) for r in rows}


def reject(s: Session, b: Baseline, note: str | None = None, now: datetime | None = None) -> Baseline:
    """Clinician: this period was not representative of the healthy state → start over."""
    b.status = BaselineStatus.rejected
    b.note = note or b.note
    s.add(b)
    s.flush()
    return start(s, b.patient_id, period_days=b.period_days, min_checkins=b.min_checkins, now=now)
