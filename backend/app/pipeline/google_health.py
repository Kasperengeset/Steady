"""Google Health API (the Fitbit successor) -> one wearable day per date, for the health branch.

Reads the signed-in user's own data, read-only, from https://health.googleapis.com/v4:
  steps                          per-minute intervals -> minute_steps (1440 values from 00:00)
  sleep                          sessions; the main sleep ending on day D -> sleep_hours, sleep_midpoint
  daily-resting-heart-rate       -> resting_hr
  daily-heart-rate-variability   -> hrv_rmssd

Output per date: the keyword arguments of pipeline.health.wearable_features, so a day can be sent
straight to POST /api/checkins/wearable. Days are civil (the user's own clock), as the API reports
them. Field names follow the v4 reference; parsing is defensive because the docs show some fields
only by example (run scripts/google_health_sync.py --debug to see the raw shapes).

Auth is done by the caller (scripts/google_health_sync.py): pass an authorised requests session.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
import math

API = "https://health.googleapis.com/v4/users/me"
SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",       # steps
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",                      # sleep
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",  # resting HR, HRV
]
MINUTES = 1440


# ------------------------------------------------------------------ time helpers

def _date(v) -> date:
    """google.type.Date ({"year", "month", "day"}) or an ISO string."""
    if isinstance(v, dict):
        return date(int(v["year"]), int(v["month"]), int(v["day"]))
    return date.fromisoformat(str(v)[:10])


def _civil(v) -> datetime:
    """CivilDateTime ({"date": {...}, "time": {"hours", "minutes", ...}}) or an ISO string, no zone."""
    if isinstance(v, dict):
        d, t = _date(v["date"]), v.get("time") or {}
        return datetime(d.year, d.month, d.day, int(t.get("hours", 0)), int(t.get("minutes", 0)), int(t.get("seconds", 0)))
    return datetime.fromisoformat(str(v).replace("Z", "")).replace(tzinfo=None)


def _local(interval: dict, which: str) -> datetime:
    """Civil time of an interval's start or end; falls back to physical time + UTC offset."""
    civil = interval.get(f"civil{which.capitalize()}Time")
    if civil:
        return _civil(civil)
    physical = datetime.fromisoformat(interval[f"{which}Time"].replace("Z", "+00:00"))
    offset = float(str(interval.get(f"{which}UtcOffset", "0s")).rstrip("s") or 0)
    return (physical + timedelta(seconds=offset)).replace(tzinfo=None)


# ------------------------------------------------------------------ parsing (pure, unit-tested)

def minute_steps_by_day(points: list[dict]) -> dict[date, list[float]]:
    """Per-minute steps for each day. Missing minutes are 0 (no steps, or watch not worn)."""
    days: dict[date, list[float]] = defaultdict(lambda: [0.0] * MINUTES)
    for p in points:
        s = p.get("steps") or {}
        if "count" not in s:
            continue
        start, end = _local(s["interval"], "start"), _local(s["interval"], "end")
        n = max(1, round((end - start).total_seconds() / 60))  # spread a longer interval evenly
        for k in range(n):
            t = start + timedelta(minutes=k)
            days[t.date()][t.hour * 60 + t.minute] += float(s["count"]) / n
    return dict(days)


def sleep_by_day(points: list[dict]) -> dict[date, dict[str, float]]:
    """Main sleep per wake-up day: hours asleep and the midpoint in hours after midnight
    (03:30 -> 3.5; before midnight is negative)."""
    best: dict[date, tuple[float, dict]] = {}
    for p in points:
        s = p.get("sleep") or {}
        if "interval" not in s:
            continue
        start, end = _local(s["interval"], "start"), _local(s["interval"], "end")
        asleep = float((s.get("summary") or {}).get("minutesAsleep") or (end - start).total_seconds() / 60)
        meta = s.get("metadata") or {}
        main = bool(meta.get("mainSleep", meta.get("main", False)))
        key = (main, asleep)  # prefer the main sleep, then the longest
        day = end.date()
        if day not in best or key > best[day][0]:
            mid = start + (end - start) / 2
            midnight = datetime(day.year, day.month, day.day)
            best[day] = (key, {"sleep_hours": asleep / 60, "sleep_midpoint": (mid - midnight).total_seconds() / 3600})
    return {d: v for d, (_, v) in best.items()}


def resting_hr_by_day(points: list[dict]) -> dict[date, float]:
    out = {}
    for p in points:
        r = p.get("dailyRestingHeartRate") or {}
        if "date" in r and r.get("beatsPerMinute") is not None:
            out[_date(r["date"])] = float(r["beatsPerMinute"])
    return out


def hrv_by_day(points: list[dict]) -> dict[date, float]:
    """Daily HRV in ms: deep-sleep RMSSD when present (explicitly RMSSD), else the daily average."""
    out = {}
    for p in points:
        h = p.get("dailyHeartRateVariability") or {}
        v = h.get("deepSleepRootMeanSquareOfSuccessiveDifferencesMilliseconds", h.get("averageHeartRateVariabilityMilliseconds"))
        if "date" in h and v is not None:
            out[_date(h["date"])] = float(v)
    return out


def exercise_minutes_by_day(points: list[dict]) -> dict[date, float]:
    """Actual workout activeDuration, excluding pauses. Split cross-midnight sessions proportionally.

    Session duration is not a substitute for missing active duration: a paused workout
    must not earn food for time spent resting. Reconciled data is preferred by the caller.
    """
    out: dict[date, float] = defaultdict(float)
    seen = set()
    for point in points:
        exercise = point.get("exercise") or {}
        duration = exercise.get("activeDuration")
        if duration is None or "interval" not in exercise:
            continue
        try:
            start = _local(exercise["interval"], "start")
            end = _local(exercise["interval"], "end")
            active = float(str(duration).removesuffix("s"))
        except (ValueError, KeyError, TypeError):
            continue
        elapsed = (end - start).total_seconds()
        if not math.isfinite(active) or active < 0 or elapsed <= 0 or active > elapsed + 1:
            continue
        key = point.get("name") or (start, end, exercise.get("exerciseType"))
        if key in seen:
            continue
        seen.add(key)
        cursor = start
        while cursor < end:
            boundary = datetime.combine(cursor.date() + timedelta(days=1), datetime.min.time())
            until = min(end, boundary)
            out[cursor.date()] += active * (until - cursor).total_seconds() / elapsed / 60
            cursor = until
    return {day: min(1440.0, minutes) for day, minutes in out.items()}


def assemble(start: date, end: date, steps: dict, sleep: dict, rhr: dict, hrv: dict,
             exercise: dict | None = None) -> dict[date, dict]:
    """{date: wearable_features kwargs} for start <= date < end; days with no data are left out."""
    out = {}
    d = start
    while d < end:
        day: dict = {}
        if d in steps and sum(steps[d]) > 0:
            day["minute_steps"] = steps[d]
        if d in sleep:
            day.update(sleep[d])
        if d in rhr:
            day["resting_hr"] = rhr[d]
        if d in hrv:
            day["hrv_rmssd"] = hrv[d]
        if exercise and d in exercise:
            day["active_minutes"] = exercise[d]
        if day:
            out[d] = day
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ API client

class GoogleHealth:
    """Thin read-only client. `session` is an authorised requests session (refreshes its own token)."""

    def __init__(self, session):
        self.s = session
        self.errors: dict[str, str] = {}  # data type -> reason it could not be read (e.g. scope not granted)

    def _safe(self, data_type: str, *args, **kw) -> list[dict]:
        try:
            return self.points(data_type, *args, **kw)
        except Exception as e:  # noqa: BLE001 – e.g. 403 MISSING_OAUTH_SCOPE if a permission was unticked
            resp = getattr(e, "response", None)
            self.errors[data_type] = f"{getattr(resp, 'status_code', '')} {getattr(resp, 'text', str(e))[:200]}".strip()
            return []

    def identity(self) -> dict:
        r = self.s.get(f"{API}/identity", headers={"Accept": "application/json"}, timeout=30)
        if r.status_code == 400 and "ACCOUNT_NOT_LINKED" in r.text:
            raise RuntimeError("Account not linked: sign in to the Google Health mobile app once, then retry.")
        r.raise_for_status()
        return r.json()

    def points(self, data_type: str, filter_: str, page_size: int, reconcile: bool = False) -> list[dict]:
        """All data points of a type matching an AIP-160 filter, following page tokens."""
        url = f"{API}/dataTypes/{data_type}/dataPoints" + (":reconcile" if reconcile else "")
        out, token = [], None
        while True:
            params = {"filter": filter_, "pageSize": page_size, **({"pageToken": token} if token else {})}
            r = self.s.get(url, params=params, headers={"Accept": "application/json"}, timeout=60)
            r.raise_for_status()
            j = r.json()
            out += j.get("dataPoints") or j.get("reconciledDataPoints") or []
            token = j.get("nextPageToken")
            if not token:
                return out

    def fetch_days(self, start: date, end: date, raw: dict | None = None) -> dict[date, dict]:
        """Days start <= d < end. `raw`, if given, receives the raw points per type (for --debug)."""
        a, b = start.isoformat(), end.isoformat()
        steps_filter = f'steps.interval.civil_start_time >= "{a}" AND steps.interval.civil_start_time < "{b}"'
        try:  # reconcile de-duplicates phone + watch steps; fall back to list if it is not available
            steps = self.points("steps", steps_filter, 10000, reconcile=True)
        except Exception:  # noqa: BLE001
            steps = self._safe("steps", steps_filter, 10000)
        sleep = self._safe("sleep", f'sleep.interval.civil_end_time >= "{a}" AND sleep.interval.civil_end_time < "{b}"', 25)
        rhr = self._safe("daily-resting-heart-rate", f'daily_resting_heart_rate.date >= "{a}" AND daily_resting_heart_rate.date < "{b}"', 1000)
        hrv = self._safe("daily-heart-rate-variability", f'daily_heart_rate_variability.date >= "{a}" AND daily_heart_rate_variability.date < "{b}"', 1000)
        exercise_filter = f'exercise.interval.civil_start_time >= "{a}" AND exercise.interval.civil_start_time < "{b}"'
        try:
            exercise = self.points("exercise", exercise_filter, 1000, reconcile=True)
        except Exception:  # noqa: BLE001 – fall back if reconciliation is unavailable
            exercise = self._safe("exercise", exercise_filter, 1000)
        if raw is not None:
            raw.update({"steps": steps, "sleep": sleep, "daily-resting-heart-rate": rhr, "daily-heart-rate-variability": hrv,
                        "exercise": exercise})
        return assemble(start, end, minute_steps_by_day(steps), sleep_by_day(sleep), resting_hr_by_day(rhr), hrv_by_day(hrv),
                        exercise_minutes_by_day(exercise))
