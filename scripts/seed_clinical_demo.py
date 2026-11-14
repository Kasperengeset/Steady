"""Append an isolated clinical demo without resetting existing patients or calling cloud models."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("drift", "stable", "building"), default="drift")
    args = parser.parse_args()
    from sqlmodel import Session
    from app import clinical_demo, db

    db.init_db()
    with Session(db.engine) as session:
        result = clinical_demo.seed(session, args.scenario)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
