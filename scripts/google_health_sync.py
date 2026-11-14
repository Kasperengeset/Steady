"""Pull your own Fitbit data from the Google Health API and run it through the health branch.

    backend/.venv/bin/python scripts/google_health_sync.py               # last 30 days
    backend/.venv/bin/python scripts/google_health_sync.py --days 45
    ... --manual   # sign-in by copy-paste, if the localhost browser sign-in does not work
    ... --debug    # also print one raw data point per type (to check field names)

Setup (README, "Fitbit via the Google Health API"): GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env.

Saves, gitignored and blocked by the pre-commit hook (personal health data, never commit):
    data/google_health_token.json   OAuth token (in Testing mode the refresh token expires after 7 days)
    data/google_health_days.json    the pulled days
Then replays the days through the real engine in a throwaway database: the first --baseline-days
build the healthy baseline, the rest are scored. Nothing is written to the app database.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN = DATA / "google_health_token.json"
DAYS_FILE = DATA / "google_health_days.json"

# The replay must never touch the real app database: point the app at a temp DB before importing it.
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'replay.db'}"
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")  # allow partial consent
sys.path.insert(0, str(ROOT / "backend"))
from app.config import settings  # noqa: E402
from app.pipeline import google_health as gh  # noqa: E402
from app.pipeline import health as hp  # noqa: E402


def credentials(manual: bool):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), gh.SCOPES)
        if creds.valid:
            return creds
        if creds.refresh_token:
            try:
                creds.refresh(Request())
                TOKEN.write_text(creds.to_json(), encoding="utf-8")
                return creds
            except Exception:  # noqa: BLE001
                print("Stored sign-in expired (7-day limit in Testing mode). Signing in again.")

    if not settings.google_client_id or not settings.google_client_secret:
        sys.exit("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing in .env. See README, 'Fitbit via the Google Health API'.")
    client = {"client_id": settings.google_client_id, "client_secret": settings.google_client_secret,
              "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token"}
    if manual:  # the docs' own flow: Web client, redirect https://www.google.com, copy the address back
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_config({"web": {**client, "redirect_uris": ["https://www.google.com"]}},
                                       scopes=gh.SCOPES, redirect_uri="https://www.google.com")
        url, _ = flow.authorization_url(access_type="offline", prompt="consent")
        print("\n1. Open this address and sign in with the Google account your Fitbit is linked to:\n\n" + url)
        print("\n2. You land on google.com. Copy the whole address from the address bar.")
        flow.fetch_token(authorization_response=input("\nPaste it here: ").strip())
        creds = flow.credentials
    else:  # Desktop client: a browser opens, Google sends you back to a local port
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_config({"installed": {**client, "redirect_uris": ["http://localhost"]}}, gh.SCOPES)
        print("A browser window opens. Sign in with the Google account your Fitbit is linked to and allow all three.")
        creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    DATA.mkdir(exist_ok=True)
    TOKEN.write_text(creds.to_json(), encoding="utf-8")
    return creds


def show(days: dict[date, dict]) -> None:
    print(f"\n{'date':10} {'steps':>7} {'inactive':>8} {'M10 at':>6} {'sleep h':>7} {'mid':>5} {'rest HR':>7} {'HRV':>5}")
    for d, day in sorted(days.items()):
        f = hp.wearable_features(**day)
        cell = lambda k, fmt: format(f[k], fmt) if k in f else "-"  # noqa: E731
        print(f"{d.isoformat():10} {cell('steps', '7.0f')} {cell('inactive_share', '8.0%')} {cell('m10_onset', '5.0f')}h"
              f" {cell('sleep_hours', '7.1f')} {cell('sleep_midpoint', '5.1f')} {cell('resting_hr', '7.0f')} {cell('hrv_rmssd', '5.0f')}")


def replay(days: dict[date, dict], baseline_days: int) -> None:
    from fastapi.testclient import TestClient
    from sqlmodel import SQLModel

    from app import baseline, db
    from app.main import app
    from app.seed import seed

    SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    seed()
    c = TestClient(app)
    ordered = sorted(days)
    t0 = datetime(ordered[0].year, ordered[0].month, ordered[0].day, 8, tzinfo=timezone.utc)
    current = {"d": ordered[0]}
    baseline.clock = lambda: t0 + timedelta(days=(current["d"] - ordered[0]).days)
    c.post("/api/patients/1/baseline/start", json={"period_days": 0, "min_checkins": baseline_days})

    print(f"\nReplay through the engine (throwaway database). First {baseline_days} days = your healthy baseline.")
    print(f"{'date':10} {'baseline':8} {'health D':>8} {'S':>6}  level")
    scored = []
    for d in ordered:
        current["d"] = d
        r = c.post("/api/checkins/wearable", json={"patient_id": 1, **days[d]})
        if r.status_code != 200:
            print(f"{d.isoformat():10} skipped: {r.text[:80]}")
            continue
        j = r.json()
        s, hd = j["state"]["S_anchored"], j["state"]["branches"].get("health")
        if s is not None:
            scored.append(s)
        print(f"{d.isoformat():10} {j['baseline']['status']:8} {'-' if hd is None else f'{hd:+8.2f}'} "
              f"{'-' if s is None else f'{s:+6.2f}'}  {j['state']['level']}")
    if scored:
        print(f"\nAfter the baseline: {len(scored)} days scored. S above 1 on {sum(s > 1 for s in scored)}, "
              f"above 2 on {sum(s > 2 for s in scored)}. For ordinary days, most should stay under 1.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="how many past days to pull (today is left out)")
    ap.add_argument("--baseline-days", type=int, default=0, help="days for the baseline (default: half, max 14)")
    ap.add_argument("--manual", action="store_true", help="copy-paste sign-in (Web client, redirect https://www.google.com)")
    ap.add_argument("--debug", action="store_true", help="print one raw data point per type")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    from google.auth.transport.requests import AuthorizedSession

    client = gh.GoogleHealth(AuthorizedSession(credentials(args.manual)))
    ident = client.identity()
    print(f"Linked to Google Health: yes (health user id ...{str(ident.get('healthUserId', ''))[-4:]})")

    end = date.today()
    start = end - timedelta(days=args.days)
    raw: dict = {}
    days = client.fetch_days(start, end, raw=raw if args.debug else None)
    for t, why in client.errors.items():
        print(f"Could not read {t}: {why}")
    if args.debug:
        for t, pts in raw.items():
            print(f"\n--- {t}: {len(pts)} points; first: {json.dumps(pts[0] if pts else None)[:600]}")
    if not days:
        sys.exit(f"No data between {start} and {end}. Has the Fitbit synced to the Google Health app?")

    DATA.mkdir(exist_ok=True)
    DAYS_FILE.write_text(json.dumps({d.isoformat(): v for d, v in sorted(days.items())}), encoding="utf-8")
    show(days)
    print(f"\n{len(days)} days saved to {DAYS_FILE.relative_to(ROOT)} (gitignored)")

    if not args.no_replay:
        if len(days) < 8:
            print("Fewer than 8 days: too few to build a baseline and score. Pull more with --days.")
            return
        replay(days, args.baseline_days or min(14, len(days) // 2))


if __name__ == "__main__":
    main()
