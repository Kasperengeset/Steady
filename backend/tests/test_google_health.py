"""Google Health API parsing and client (pipeline/google_health.py) with fake responses – no network.

Shapes follow the v4 reference and its examples; both the object and the string form of civil times
are accepted because the docs show some fields only by example.
"""

from datetime import date

import pytest

from app.pipeline import google_health as gh
from app.pipeline import health as hp


def civil(y, m, d, h=0, mi=0):
    return {"date": {"year": y, "month": m, "day": d}, "time": {"hours": h, "minutes": mi}}


def step(start, end, count):
    return {"steps": {"interval": {"civilStartTime": start, "civilEndTime": end}, "count": str(count)}}


def test_minute_steps_bucketed_by_day_and_minute():
    pts = [
        step(civil(2026, 11, 10, 8, 0), civil(2026, 11, 10, 8, 1), 40),
        step("2026-11-10T08:01:00", "2026-11-10T08:02:00", 30),          # string form
        step(civil(2026, 11, 10, 9, 0), civil(2026, 11, 10, 9, 2), 100),  # 2-minute interval → 50 + 50
        step(civil(2026, 11, 11, 0, 0), civil(2026, 11, 11, 0, 1), 5),    # next day
    ]
    days = gh.minute_steps_by_day(pts)
    d10 = days[date(2026, 11, 10)]
    assert len(d10) == 1440 and d10[480] == 40 and d10[481] == 30 and d10[540] == d10[541] == 50
    assert sum(d10) == 170 and sum(days[date(2026, 11, 11)]) == 5


def test_physical_time_with_offset_when_civil_missing():
    p = {"steps": {"interval": {"startTime": "2026-11-10T06:00:00Z", "startUtcOffset": "7200s",
                                "endTime": "2026-11-10T06:01:00Z", "endUtcOffset": "7200s"}, "count": "12"}}
    assert gh.minute_steps_by_day([p])[date(2026, 11, 10)][8 * 60] == 12  # 06:00 UTC + 2 h = 08:00 local


def test_sleep_prefers_main_sleep_and_gives_midpoint():
    nap = {"sleep": {"interval": {"civilStartTime": civil(2026, 11, 11, 14), "civilEndTime": civil(2026, 11, 11, 15)},
                     "summary": {"minutesAsleep": "50"}, "metadata": {"mainSleep": False}}}
    night = {"sleep": {"interval": {"civilStartTime": civil(2026, 11, 10, 23), "civilEndTime": civil(2026, 11, 11, 7)},
                       "summary": {"minutesAsleep": "420"}, "metadata": {"main": True}}}
    s = gh.sleep_by_day([nap, night])[date(2026, 11, 11)]
    assert s["sleep_hours"] == pytest.approx(7.0)
    assert s["sleep_midpoint"] == pytest.approx(3.0)  # 23:00–07:00 → 03:00


def test_resting_hr_and_hrv():
    rhr = [{"dailyRestingHeartRate": {"date": {"year": 2026, "month": 11, "day": 10}, "beatsPerMinute": "58"}}]
    hrv = [{"dailyHeartRateVariability": {"date": "2026-11-10", "averageHeartRateVariabilityMilliseconds": 40.0,
                                          "deepSleepRootMeanSquareOfSuccessiveDifferencesMilliseconds": 47.5}},
           {"dailyHeartRateVariability": {"date": "2026-11-11", "averageHeartRateVariabilityMilliseconds": 39.0}}]
    assert gh.resting_hr_by_day(rhr) == {date(2026, 11, 10): 58.0}
    assert gh.hrv_by_day(hrv) == {date(2026, 11, 10): 47.5, date(2026, 11, 11): 39.0}  # RMSSD preferred


def test_assembled_days_feed_the_health_features():
    steps = {date(2026, 11, 10): [10.0] * 1440, date(2026, 11, 11): [0.0] * 1440}  # 11th: watch not worn
    days = gh.assemble(date(2026, 11, 10), date(2026, 11, 13), steps,
                       {date(2026, 11, 11): {"sleep_hours": 7.0, "sleep_midpoint": 3.0}}, {date(2026, 11, 10): 58.0}, {})
    assert set(days) == {date(2026, 11, 10), date(2026, 11, 11)}  # 12th has nothing → left out
    assert "minute_steps" not in days[date(2026, 11, 11)]
    f = hp.wearable_features(**days[date(2026, 11, 10)])
    assert f["steps"] == 14400 and f["resting_hr"] == 58


# ---------------------------------------------------------------- client with a fake session

class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            e = RuntimeError(f"{self.status_code}")
            e.response = self
            raise e


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for key, handler in self.routes.items():
            if url.endswith(key):
                return handler(params or {})
        return FakeResponse(404, {"error": "no route"})


def test_client_pages_falls_back_and_survives_a_missing_scope():
    pages = {None: {"dataPoints": [step(civil(2026, 11, 10, 8), civil(2026, 11, 10, 8, 1), 40)], "nextPageToken": "p2"},
             "p2": {"dataPoints": [step(civil(2026, 11, 10, 9), civil(2026, 11, 10, 9, 1), 60)]}}
    s = FakeSession({
        "steps/dataPoints:reconcile": lambda p: FakeResponse(400, {"error": "not supported"}),
        "steps/dataPoints": lambda p: FakeResponse(200, pages[p.get("pageToken")]),
        "sleep/dataPoints": lambda p: FakeResponse(403, {"error": {"status": "PERMISSION_DENIED", "message": "MISSING_OAUTH_SCOPE"}}),
        "daily-resting-heart-rate/dataPoints": lambda p: FakeResponse(200, {"dataPoints": [
            {"dailyRestingHeartRate": {"date": "2026-11-10", "beatsPerMinute": "57"}}]}),
        "daily-heart-rate-variability/dataPoints": lambda p: FakeResponse(200, {}),
    })
    client = gh.GoogleHealth(s)
    days = client.fetch_days(date(2026, 11, 10), date(2026, 11, 11))
    day = days[date(2026, 11, 10)]
    assert sum(day["minute_steps"]) == 100 and day["resting_hr"] == 57  # both pages read
    assert "sleep" in client.errors and "MISSING_OAUTH_SCOPE" in client.errors["sleep"]
    steps_call = next(p for u, p in s.calls if u.endswith("steps/dataPoints"))
    assert steps_call["filter"] == 'steps.interval.civil_start_time >= "2026-11-10" AND steps.interval.civil_start_time < "2026-11-11"'


def test_identity_not_linked_gives_a_clear_message():
    s = FakeSession({"identity": lambda p: FakeResponse(400, {"error": {"status": "FAILED_PRECONDITION", "reason": "ACCOUNT_NOT_LINKED"}})})
    with pytest.raises(RuntimeError, match="Google Health mobile app"):
        gh.GoogleHealth(s).identity()


def test_workouts_use_active_duration_ignore_duplicates_and_split_midnight():
    workout = {"name": "workout-1", "exercise": {
        "interval": {"civilStartTime": civil(2026, 11, 10, 23, 30),
                     "civilEndTime": civil(2026, 11, 11, 0, 30)},
        "exerciseType": "BIKING", "activeDuration": "1800s",
    }}
    missing_duration = {"exercise": {"interval": workout["exercise"]["interval"]}}
    invalid = {"exercise": {**workout["exercise"], "activeDuration": "NaNs"}}
    out = gh.exercise_minutes_by_day([workout, workout, missing_duration, invalid])
    assert out == {date(2026, 11, 10): 15.0, date(2026, 11, 11): 15.0}


def test_exercise_only_day_is_not_lost_from_assembled_wearable_days():
    day = date(2026, 11, 10)
    out = gh.assemble(day, date(2026, 11, 11), {}, {}, {}, {}, {day: 25.0})
    assert out[day] == {"active_minutes": 25.0}
