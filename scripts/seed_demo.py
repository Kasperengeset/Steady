"""Demo data for the video and the clinician view: four patients with a believable history.

    backend/.venv/bin/python scripts/seed_demo.py            # seed an empty database (refuses if it has check-ins)
    backend/.venv/bin/python scripts/seed_demo.py --reset    # delete the database first, then seed
    backend/.venv/bin/python scripts/seed_demo.py --db /tmp/demo.db --reset

It drives the REAL API in-process (FastAPI TestClient) day by day with the shared clock
(app.baseline.clock) patched, like backend/tests/test_baseline.py. So baselines, scores,
escalations, explanations, prompts and rewards are what the app itself computes.

- Voice: no audio at all. Acoustic features are synthetic, drawn from the corpus summary stats in
  weights/acoustic.json["feature_stats"] (control → patient shift + within-person noise), as in the
  simulator. Never Androids audio.
- Text: team-written diaries in the first person, scored by the local text model. Nothing is sent to
  Gemini (the key is blanked and transcription is never called).

Patients (history ends today; nobody has checked in today, so the video can start with a check-in):
  Mia    – steady for 30 days, locked baseline, reminders answered, rewards, streak, a small circle
  Alex   – well for 16 days, then drifts into relapse over 14 days: friends nudge, then the clinician,
           with "Why is this changing?" naming the features
  Sara   – steady for 20 days, then a diary entry with a crisis phrase today: clinician at once
  Jonas  – started 6 days ago, healthy baseline still building; in Mia's circle with his own avatar
"""

import argparse
import os
import random
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Europe/Helsinki")

GOOD = [
    "I slept well and went for a walk by the sea this morning. I had lunch with my sister and we laughed a lot. I feel calm and I am looking forward to cooking dinner tonight.",
    "I had a busy day at work but it went fine. I finished the report I was worried about and my manager was happy with it. In the evening I played guitar for an hour.",
    "I met two friends for coffee and we talked about our summer plans. I feel rested today and I had energy to clean the flat and do the laundry.",
    "I went to the gym after work and I feel good about it. I made a new recipe for dinner and it turned out nice. I am going to read a bit before bed.",
    "Today was quiet. I worked from home, I took a long walk at lunch and I called my mother. I feel content and I am sleeping well this week.",
    "I helped a colleague with a tricky problem and I felt useful. After work I went swimming. I am a bit tired now but in a good way.",
    "I spent the morning in the garden and the afternoon with my nephew at the park. I enjoyed the sun. I am grateful for a simple, good day.",
    "I had a relaxed weekend day. I went to a small concert with a friend and I really enjoyed the music. I feel like myself again.",
    "I got up early and went running before work. My day was productive and I still had energy in the evening to meet my neighbours for tea.",
    "I cooked with my partner tonight and we watched a funny film. Work was normal. I feel steady and I am glad the week went well.",
]
LOW = [
    "I slept badly again and I did not want to get out of bed. I cancelled lunch with my friend because I could not face it. Everything feels heavy.",
    "I stayed at home all day. I tried to work but I could not concentrate on anything. I feel tired and empty and nothing seems worth the effort.",
    "I did not answer my phone today. I feel like I am a burden to the people around me. I keep thinking that nothing will get better.",
    "I woke up at four and could not fall asleep again. I feel numb. I did not eat much and I did not go outside at all.",
    "I forced myself to go to work but I barely spoke to anyone. I feel alone even when people are around. I just want the day to end.",
    "I have no energy for anything. I used to enjoy cooking but now I only eat toast. I feel hopeless and I keep blaming myself for everything.",
]
CRISIS = (
    "I have not slept for days and I feel completely alone. I don't want to be alive anymore and I keep "
    "thinking that everyone would be better off without me. I do not know what to do."
)


# Tuned so the stories read as intended: Mia stays below the friend threshold; Alex gets the friends nudge
# first and the clinician a few days later. Change with care, re-run and read the printout.
STORY_SEEDS = {"Mia": 0, "Alex": 0, "Sara": 0, "Jonas": 0}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reset", action="store_true", help="delete the database file first")
    p.add_argument("--db", help="SQLite file (default: DATABASE_URL from .env, else data/app.db)")
    p.add_argument("--seed", type=int, default=0, help="shift every story's random seed (default 0 = the tuned demo)")
    return p.parse_args()


def main() -> int:
    args = parse()
    if args.db:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(args.db).resolve()}"
    os.environ["GEMINI_API_KEY"] = ""  # belt and braces: the demo never calls Gemini
    sys.path.insert(0, str(REPO / "backend"))

    from app.config import settings

    db_path = Path(settings.database_url.removeprefix("sqlite:///"))
    if args.reset and db_path.exists():
        db_path.unlink()
        print(f"deleted {db_path}")

    from fastapi.testclient import TestClient
    from sqlmodel import Session, select

    from app import baseline as baseline_mod
    from app import db
    from app.engine import state
    from app.main import app
    from app.models import Arm, CheckIn, Contact, ContactKind, Escalation, FeatureSource, Patient
    from app.pipeline import PipelineResult, extract_text_features
    from app.routers import checkins as checkins_router
    from app.seed import ARMS, seed

    settings.gemini_api_key = ""
    db.init_db()
    seed()  # Mia (id 1) + arms + contacts, as on a fresh install
    with Session(db.engine) as s:
        if s.exec(select(CheckIn)).first() is not None:
            print(f"{db_path} already has check-ins – run with --reset to start over.")
            return 1

    class R:  # one random stream per patient, so tuning one story never changes another
        rng = random.Random()

    def story(name: str) -> None:
        R.rng = random.Random(f"{name}:{STORY_SEEDS[name] + args.seed}")
    stats = state.load_branch("acoustic").meta["feature_stats"]
    emotion = {"arousal", "valence"}

    def acoustic(depression: float) -> list[tuple[str, float, FeatureSource]]:
        out = []
        for f, st in stats.items():
            mu = st["control_mean"] + depression * (st["patient_mean"] - st["control_mean"])
            v = mu + R.rng.gauss(0, st["within_sd"] * 0.5)
            out.append((f, v, FeatureSource.emotion if f in emotion else FeatureSource.prosody))
        return out

    queue: list[PipelineResult] = []
    checkins_router.extract_features = lambda path: queue.pop(0)
    checkins_router.extract_text_features = lambda text: queue.pop(0)
    clock = {"now": datetime.now(timezone.utc)}
    wall_start = datetime.now(timezone.utc) - timedelta(minutes=1)
    baseline_mod.clock = lambda: clock["now"]
    c = TestClient(app)

    def ok(r):
        assert r.status_code < 300, f"{r.request.method} {r.request.url}: {r.status_code} {r.text}"
        return r.json()

    def at(day, hh: int, mm: int) -> None:
        clock["now"] = datetime.combine(day, time(hh, mm), tzinfo=TZ).astimezone(timezone.utc)

    def backdate_escalations() -> None:
        """Escalation.ts defaults to the wall clock, not app.baseline.clock: move new rows to the simulated day."""
        with Session(db.engine) as s:
            sim = clock["now"]
            for e in s.exec(select(Escalation)).all():
                ts = e.ts if e.ts.tzinfo else e.ts.replace(tzinfo=timezone.utc)
                if ts > wall_start and sim < wall_start:  # stamped by the wall clock just now
                    e.ts = sim
                    s.add(e)
            s.commit()

    def spoken(pid: int, diary: str, depression: float, slot: str = "evening") -> dict:
        res = extract_text_features(diary)  # local text model on the "transcript"
        res.features = acoustic(depression) + res.features
        res.speech_s = round(R.rng.uniform(18, 29), 1)
        queue.append(res)
        out = ok(c.post("/api/checkins", data={"patient_id": pid, "slot": slot},
                        files={"audio": ("memo.wav", b"demo", "audio/wav")}))
        backdate_escalations()
        return out

    def written(pid: int, diary: str, slot: str = "evening") -> dict:
        queue.append(extract_text_features(diary))
        out = ok(c.post("/api/checkins/text", json={"patient_id": pid, "slot": slot, "text": diary}))
        backdate_escalations()
        return out

    def new_patient(name: str, friend: str) -> int:
        with Session(db.engine) as s:
            p = Patient(name=name)
            s.add(p)
            s.commit()
            s.refresh(p)
            s.add(Contact(patient_id=p.id, name=friend, kind=ContactKind.friend))
            s.add(Contact(patient_id=p.id, name="Dr. Virtanen", kind=ContactKind.clinician))
            for key, label in ARMS:
                s.add(Arm(patient_id=p.id, key=key, label=label))
            s.commit()
            return p.id

    def day_of_prompted_checkin(pid, day, diary, depression, action: bool | None = None, late=False):
        """The daily moment: the prompt fires, the person answers (usually within the hour)."""
        hh, mm = R.rng.randint(10, 18), R.rng.choice([0, 10, 20, 30, 40, 50])
        at(day, hh, mm)
        ok(c.post(f"/api/patients/{pid}/prompt/now"))
        delay = R.rng.randint(150, 300) if late else R.rng.randint(3, 70)
        clock["now"] += timedelta(minutes=delay)
        out = spoken(pid, diary, depression)
        if action is not None:
            clock["now"] += timedelta(hours=2)
            ok(c.post(f"/api/patients/{pid}/habits/action", json={"done": action, "rating": R.rng.randint(3, 5) if action else None}))
        return out

    today = datetime.now(TZ).date()
    created = []

    # --- Mia: steady, 30 days, ends yesterday -------------------------------------------------
    mia = 1
    story("Mia")
    for i in range(30, 0, -1):
        if i in (12, 21):  # a couple of missed days: nothing is lost, the companion is kind
            continue
        last = day_of_prompted_checkin(mia, today - timedelta(days=i), R.rng.choice(GOOD), 0.0,
                                       action=R.rng.random() < 0.75, late=R.rng.random() < 0.15)
    created.append(("Mia", "steady 30 days", last))

    # --- Alex: well, then a slow drift into relapse ---------------------------------------------
    alex = new_patient("Alex", "Sanna")
    story("Alex")
    days = 30
    for k in range(days):
        i = days - k  # days ago (30 … 1)
        drift = 0.0 if k < 16 else min(1.6, (k - 15) * 0.12)
        # Low entries become more common as the voice drifts – not all at once (the text branch is strong)
        diary = R.rng.choice(LOW) if R.rng.random() < min(1.0, max(0.0, (drift - 0.3) / 1.6)) else R.rng.choice(GOOD)
        last = day_of_prompted_checkin(alex, today - timedelta(days=i), diary, drift,
                                       action=R.rng.random() < (0.7 if drift < 0.6 else 0.3), late=drift > 1)
    created.append(("Alex", "relapse drift over the last 14 days", last))

    # --- Sara: steady, then a crisis phrase in today's written diary ----------------------------
    sara = new_patient("Sara", "Ida")
    story("Sara")
    for i in range(20, 0, -1):
        last = day_of_prompted_checkin(sara, today - timedelta(days=i), R.rng.choice(GOOD), 0.0, action=True)
    at(today, 8, 15)
    clock["now"] = min(clock["now"], datetime.now(timezone.utc))
    last = written(sara, CRISIS, slot="morning")
    created.append(("Sara", "crisis phrase in today's diary", last))

    # --- Jonas: baseline still building --------------------------------------------------------
    jonas = new_patient("Jonas", "Mia")
    story("Jonas")
    for i in range(6, 0, -1):
        last = day_of_prompted_checkin(jonas, today - timedelta(days=i), R.rng.choice(GOOD), 0.0, action=True)
    created.append(("Jonas", "baseline building (6 days)", last))

    # --- Circle: Mia's, with Jonas (his own avatar) and a light friend account -----------------
    clock["now"] = datetime.now(timezone.utc) - timedelta(minutes=5)
    for pid, name in ((mia, "Mia"), (jonas, "Jonas")):
        ok(c.put(f"/api/patients/{pid}/profile", json={
            "social_consent": True, "mode": "profile", "display_name": name,
            "share_avatar": True, "share_level": True, "share_streak": True, "share_checked_in": True}))
    circle = ok(c.post(f"/api/patients/{mia}/circle", json={"name": "Mia's circle"}))
    code = circle.get("invite_code") or circle.get("circle", {}).get("invite_code")
    jonas_m = ok(c.post("/api/circles/join", json={"code": code, "patient_id": jonas}))
    emil = ok(c.post("/api/circles/join", json={"code": code, "name": "Emil"}))
    mia_member = circle.get("member_id") or circle.get("me", {}).get("member_id")
    reactions = 0
    if mia_member:
        for token, kind in ((emil["token"], "sun"), (jonas_m["token"], "hug")):
            if c.post("/api/circles/react", json={"token": token, "to_member_id": mia_member, "kind": kind}).status_code < 300:
                reactions += 1

    # --- Report ---------------------------------------------------------------------------------
    clock["now"] = datetime.now(timezone.utc)
    print(f"\nSeeded {db_path}\n")
    print(f"{'patient':8} {'story':40} {'baseline':9} {'S':>6} {'level':>5}  rewards")
    ids = {"Mia": mia, "Alex": alex, "Sara": sara, "Jonas": jonas}
    for name, story, out in created:
        pid = ids[name]
        rw = ok(c.get(f"/api/patients/{pid}/rewards"))  # also pays every reward that is due
        S = out["state"]["S"]
        print(f"{name:8} {story:40} {out['baseline']['status']:9} {S if S is None else round(S, 2)!s:>6} "
              f"{out['state']['level']:>5}  balance {rw.get('balance')}, streak {rw.get('streak')}")
    with Session(db.engine) as s:
        ladder = [f"{(e.ts.replace(tzinfo=timezone.utc).astimezone(TZ)).date()} → {e.recipient.value} ({e.reason})"
                  for e in s.exec(select(Escalation).where(Escalation.patient_id == alex).order_by(Escalation.ts))]
    print("\nAlex – escalations: " + ("; ".join(ladder) or "none"))
    alex_detail = ok(c.get(f"/api/patients/{alex}"))
    if alex_detail.get("explanation"):
        print(f"\nAlex – why: {alex_detail['explanation']['sentence']}")
    print(f"\nCircle '{circle.get('name', "Mia's circle")}' (code {code}): Mia, Jonas, Emil; {reactions} reactions")
    print("Mia has not checked in today: open the app and press \"Demo: send today's prompt now\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
