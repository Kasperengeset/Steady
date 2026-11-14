"""Load the demo: four patients with believable histories, built through the real API and engine.

    backend/.venv/bin/python scripts/seed_demo.py          # resets data/app.db, then fills it

Run it again right before recording, so the story ends today. No Gemini calls are made.

Alex (patient 1, the main demo), 44 days after TMS, ending this morning:
  days 1-14   healthy baseline (confirmed by Dr. Virtanen)
  days 15-32  well
  days 33-44  gradual decline -> friends nudged, then the clinic flagged
  today       morning Fitbit sync done; the live check-in happens in the demo
  Activity and sleep: the team member's real Fitbit days (data/google_health_days.json, gitignored)
      if present, else synthetic. In the decline, those real days are changed: fewer steps, later
      activity, shorter and later sleep, higher resting heart rate, lower HRV.
  Words: one written diary each evening, the team-written diary set plus a sentence bank mixed by
      the day's state (synthetic).
  Voice: simulated morning check-ins from the Androids corpus statistics in acoustic.json.
  Actions: the real bandit; simulated answers with hidden helpfulness (a walk helps Alex most).
Sara: well, then a sudden change, and yesterday a check-in mentioning self-harm.
Jonas: baseline still building (day 9 of 14).
Mia: steady, baseline waiting for the clinician to confirm.
Everything except Alex's Fitbit days is simulated or team-written. Say so in the demo.
"""

import argparse
import csv
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import Session, SQLModel  # noqa: E402

from app import baseline as baseline_mod  # noqa: E402
from app import db, appie  # noqa: E402
from app.engine import state  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Arm, Contact, ContactKind, FeatureSource, Patient  # noqa: E402
from app.pipeline import PipelineResult  # noqa: E402
from app.routers import checkins as checkins_router  # noqa: E402
from app.seed import ARMS  # noqa: E402

FITBIT = ROOT / "data" / "google_health_days.json"
DIARY = ROOT / "ml" / "text" / "diary_sanity.csv"
rng = random.Random(42)

# ------------------------------------------------------------------ sentence bank (synthetic diary text)
GOOD = [
    "I slept well and got up before my alarm.", "I went for a walk by the water after lunch.",
    "I had coffee with Sam and we laughed a lot.", "My work went fine today and I finished the report early.",
    "I cooked a proper dinner and tried a new recipe.", "I called my sister in the evening, and it was nice to catch up.",
    "I enjoyed the cold bright weather and biked to the shop.", "I read a few chapters before bed.",
    "I felt rested and clear-headed most of the day.", "I went to the gym with Maya and stretched afterwards.",
    "I planned our weekend trip and booked my tickets.", "I spent the afternoon fixing my bike with my neighbour.",
]
NEUTRAL = [
    "Ordinary day at work with a few meetings.", "Did the laundry and some cleaning.",
    "Watched a film in the evening.", "Bought groceries on the way home.",
    "Answered emails for most of the morning.", "Rainy day, so I stayed mostly indoors.",
    "Had leftovers for dinner.", "Took the bus into town for an errand.", "The day went by quickly.",
]
LOW = [
    "Did not get out of bed until late.", "I cancelled plans again, I just could not face it.",
    "Everything feels heavy and slow right now.", "I am tired all the time, even after sleeping.",
    "Did not really talk to anyone today.", "I keep thinking nothing I do makes a difference.",
    "Skipped lunch, I was not hungry.", "I am fine I guess, just tired of pretending.",
    "Stayed in my room and scrolled for hours.", "Could not focus at work and read the same email five times.",
    "Felt flat all day, nothing seemed worth the effort.", "I left the messages unanswered again.",
]
CRISIS_ENTRY = ("I can't do this anymore. I stayed in bed all day and ignored everyone. "
                "Some nights I think I don't want to be here anymore. I haven't told anyone.")


def diary(d: float) -> str:
    """5-7 sentences; each is low with probability d, else good or neutral."""
    out = []
    for _ in range(rng.randint(5, 7)):
        r = rng.random()
        pool = LOW if r < d else GOOD if r < d + (1 - d) * 0.6 else NEUTRAL
        s = rng.choice(pool)
        if s not in out:
            out.append(s)
    return " ".join(out)


def team_entries() -> dict[str, str]:
    rows = csv.DictReader(open(DIARY, encoding="utf-8"))
    return {r["id"]: r["text"] for r in rows if r["variant"] == "as written"}


# ------------------------------------------------------------------ simulated voice (corpus statistics)
STATS = state.load_branch("acoustic").meta["feature_stats"]


def voice(d: float, offset: dict) -> PipelineResult:
    res = PipelineResult()
    for f, st in STATS.items():
        v = offset[f] + st["control_mean"] + d * (st["patient_mean"] - st["control_mean"]) + rng.gauss(0, st["within_sd"] * 0.8)
        res.features.append((f, v, FeatureSource.emotion if f in ("arousal", "valence") else FeatureSource.prosody))
    return res


def person_offset() -> dict:
    return {f: rng.gauss(0, st["sd"] * 0.5) for f, st in STATS.items()}


# ------------------------------------------------------------------ wearable days
def synthetic_day(d: float) -> dict:
    wake, bed = 7.0 + 2.0 * d + rng.gauss(0, 0.4), 23.0 + rng.gauss(0, 0.4)
    level = 45 * (1 - 0.55 * d)
    minutes = [max(0.0, rng.gauss(level, 15)) if wake <= i / 60 < bed and rng.random() < 0.3 else 0.0 for i in range(1440)]
    return {"minute_steps": minutes, "sleep_hours": rng.gauss(7.4 - 1.3 * d, 0.4), "sleep_midpoint": rng.gauss(3.4 + 1.8 * d, 0.3),
            "resting_hr": rng.gauss(60 + 5 * d, 1.5), "hrv_rmssd": rng.gauss(48 * (1 - 0.3 * d), 4)}


def worsen(day: dict, f: float) -> dict:
    """A real day, changed in the RADAR-MDD directions by f in [0, 1]."""
    out = dict(day)
    if "minute_steps" in day:
        shift = int(150 * f)
        m = [v * (1 - 0.55 * f) for v in day["minute_steps"]]
        out["minute_steps"] = [0.0] * shift + m[: 1440 - shift]
    if "sleep_hours" in day:
        out["sleep_hours"] = max(3.5, day["sleep_hours"] - 1.3 * f)
    if "sleep_midpoint" in day:
        out["sleep_midpoint"] = day["sleep_midpoint"] + 1.8 * f
    if "resting_hr" in day:
        out["resting_hr"] = day["resting_hr"] + 5 * f
    if "hrv_rmssd" in day:
        out["hrv_rmssd"] = day["hrv_rmssd"] * (1 - 0.3 * f)
    return out


def real_days() -> list[dict]:
    if not FITBIT.exists():
        return []
    days = json.loads(FITBIT.read_text(encoding="utf-8"))
    return [days[k] for k in sorted(days)]


# ------------------------------------------------------------------ driver
class Demo:
    def __init__(self):
        SQLModel.metadata.drop_all(db.engine)
        db.init_db()
        self.c = TestClient(app)  # no lifespan: no default seed, no model warm-up
        self.t = datetime.now(timezone.utc)
        baseline_mod.clock = lambda: self.t
        self.voice_queue: list[PipelineResult] = []
        checkins_router.extract_features = lambda path: self.voice_queue.pop(0)

    def at(self, day: date, hh: int, mm: int = 0):
        self.t = datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)

    def patient(self, name: str, friends: list[str]) -> int:
        with Session(db.engine) as s:
            p = Patient(name=name)
            s.add(p)
            s.commit()
            s.refresh(p)
            for f in friends:
                s.add(Contact(patient_id=p.id, name=f, kind=ContactKind.friend))
            s.add(Contact(patient_id=p.id, name="Dr. Virtanen", kind=ContactKind.clinician))
            for key, label in ARMS:
                s.add(Arm(patient_id=p.id, key=key, label=label))
            s.commit()
            return p.id

    def ok(self, r):
        assert r.status_code == 200, r.text
        return r.json()

    def wearable(self, pid, day):
        return self.ok(self.c.post("/api/checkins/wearable", json={"patient_id": pid, "slot": "morning", **day}))

    def spoken(self, pid, res):
        self.voice_queue.append(res)
        return self.ok(self.c.post("/api/checkins", data={"patient_id": pid, "slot": "morning"},
                                   files={"audio": ("checkin.wav", b"RIFF", "audio/wav")}))

    def written(self, pid, text):
        return self.ok(self.c.post("/api/checkins/text", json={"patient_id": pid, "slot": "evening", "text": text}))

    def feedback(self, pid, day: date, d: float, helpful: dict):
        today = self.ok(self.c.get(f"/api/patients/{pid}/today"))
        if not today["action"] or today["action"]["date"] != day.isoformat():
            return
        arm = today["action"]["arm_key"]
        done = rng.random() < 0.85 - 0.45 * d
        helped = done and rng.random() < helpful.get(arm, 0.4) * (1 - 0.3 * d)
        rating = (rng.randint(4, 5) if helped else rng.randint(1, 3)) if done else None
        self.ok(self.c.post("/api/interventions/feedback",
                            json={"patient_id": pid, "date": day.isoformat(), "completed": done, "rating": rating}))

    def confirm(self, pid):
        self.ok(self.c.post(f"/api/patients/{pid}/baseline/confirm", json={"clinician": "Dr. Virtanen"}))

    def live_day(self, pid, day: date, d: float, wear: dict, text: str | None, offset: dict | None, helpful: dict,
                 feedback: bool = True):
        self.at(day, 7, 40)
        self.wearable(pid, wear)
        if offset is not None:
            self.at(day, 8, 10)
            self.spoken(pid, voice(d, offset))
        if text:
            self.at(day, 20, 30)
            out = self.written(pid, text)
        if feedback:
            self.at(day, 21, 0)
            self.feedback(pid, day, d, helpful)
        return locals().get("out")


def ramp(i: int, start: int, length: int) -> float:
    return min(1.0, max(0.0, (i - start + 1) / length))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic local demo; existing data is kept unless --reset is explicit.")
    parser.add_argument("--reset", action="store_true", help="replace the selected demo database (never use for real patient data)")
    args = parser.parse_args()
    database_path = db.engine.url.database
    if database_path and database_path != ":memory:" and Path(database_path).exists() and not args.reset:
        parser.error("Database already exists. Use a separate DATABASE_URL, or --reset only for disposable demo data.")
    demo = Demo()
    today = datetime.now(timezone.utc).date()
    team = team_entries()
    real = real_days()
    print(f"Fitbit: {'real days from ' + str(FITBIT.relative_to(ROOT)) if real else 'not found, using synthetic days'}")

    # ---- Alex: 44 days + this morning
    alex = demo.patient("Alex", ["Sam", "Maya"])
    off = person_offset()
    helpful = {"walk_outside": 0.85, "call_friend": 0.65, "morning_light": 0.6, "small_task": 0.5, "music": 0.4, "breathing": 0.35}
    n, decline_start = 44, 32
    good_slots = {1: "good_1", 4: "good_2", 7: "good_3", 10: "good_4", 12: "good_5"}
    low_slots = {35: "low_5", 37: "low_3", 39: "low_2", 41: "low_4", 43: "low_1"}
    print("\nAlex (main demo)")
    for i in range(n):
        day = today - timedelta(days=n - i)
        f = ramp(i, decline_start, n - decline_start)
        d = 0.1 + 0.8 * f
        if real:
            base = real[i] if i < len(real) else real[len(real) - 14 + (i - len(real)) % 14]
            wear = worsen(base, f) if f > 0 else base
        else:
            wear = synthetic_day(d)
        text = team.get(good_slots.get(i, ""), None) or team.get(low_slots.get(i, ""), None) or diary(d)
        out = demo.live_day(alex, day, d, wear, text, off, helpful)
        if i == 14:
            demo.confirm(alex)
        st = out["state"] if out else {}
        print(f"  {day} {'baseline' if i < 14 else 'decline ' if f > 0 else 'well    '} "
              f"S={'-' if st.get('S') is None else format(st['S'], '+.2f')} level={st.get('level', 0)}")
    demo.at(today, 7, 40)
    demo.wearable(alex, worsen(real[-1], 1.0) if real else synthetic_day(0.9))

    # ---- Sara: well, then sudden change and a crisis check-in yesterday
    sara = demo.patient("Sara", ["Lina"])
    off_s = person_offset()
    for i in range(30):
        day = today - timedelta(days=30 - i)
        d = 0.1 if i < 26 else 0.75
        text = CRISIS_ENTRY if i == 29 else diary(d)
        demo.live_day(sara, day, d, synthetic_day(d), text, off_s, helpful, feedback=i < 29)
        if i == 15:
            demo.confirm(sara)

    # ---- Jonas: baseline building, day 9 of 14
    jonas = demo.patient("Jonas", ["Erik"])
    off_j = person_offset()
    for i in range(9):
        day = today - timedelta(days=9 - i)
        demo.live_day(jonas, day, 0.1, synthetic_day(0.1), diary(0.1), off_j, helpful, feedback=False)

    # ---- Mia: steady, baseline not yet confirmed
    mia = demo.patient("Mia", ["Jonas"])
    off_m = person_offset()
    for i in range(35):
        day = today - timedelta(days=35 - i)
        demo.live_day(mia, day, 0.1, synthetic_day(0.1), diary(0.1) if i % 2 == 0 else None, off_m, helpful)

    # ---- summary
    demo.at(today, 9, 0)
    invite_code = None
    for pid, nickname in ((alex, "Alex"), (sara, "Sara"), (jonas, "Jonas"), (mia, "Mia")):
        demo.ok(demo.c.put(f"/api/patients/{pid}/profile", json={
            "mode": "profile", "display_name": nickname, "social_consent": True,
            "share_avatar": True, "share_level": True, "share_streak": True, "share_checked_in": True,
        }))
        if invite_code is None:
            created = demo.ok(demo.c.post(f"/api/patients/{pid}/circle", json={"name": "Steady demo"}))
            invite_code = created["invite_code"]
        else:
            demo.ok(demo.c.post("/api/circles/join", json={"code": invite_code, "patient_id": pid}))
    with Session(db.engine) as session:
        for pid, xp, hobby in ((alex, 180, "Play guitar"), (sara, 80, "Make something"),
                               (jonas, 25, "Read a chapter"), (mia, 710, "Tend the plants")):
            habit = appie.add_habit(session, pid, hobby)
            appie.tick_habit(session, pid, habit.key)
            appie.feed(session, pid, min(20, xp))
        print(f"Appie demo ready: food, upgraded looks, hobbies, and circle {invite_code}.")
    c = demo.c
    pts = demo.ok(c.get("/api/patients"))
    print("\nPatients:", ", ".join(f"{p['name']} (S {p['S'] if p['S'] is None else round(p['S'], 2)}, level {p['level']})" for p in pts))
    print("Notifications:", [f"{n['patient']}: {n['kind']}" for n in demo.ok(c.get("/api/clinician/notifications"))])
    t = demo.ok(c.get(f"/api/patients/{alex}/today"))
    print(f"Alex today: {t['status_text']} | action: {t['action']['label'] if t['action'] else None} | {t['action']['reason'] if t['action'] else ''}")
    print("Alex what helps:", [(a["label"], a["learned"], f"{a['helped']}/{a['done']}") for a in demo.ok(c.get(f"/api/patients/{alex}/insights"))[:3]])
    e = demo.ok(c.get(f"/api/patients/{alex}/explain?days=14"))
    print("Alex why (14 days):", e.get("summary"))
    print(f"\nDemo loaded into {db.engine.url.database}. Start the backend and open the app.")


if __name__ == "__main__":
    main()
