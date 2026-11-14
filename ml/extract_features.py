"""Extract acoustic features for every Androids clip with the *same* code the app uses.

  backend/.venv/bin/python ml/extract_features.py --task interview      (~874 clips)
  backend/.venv/bin/python ml/extract_features.py --task reading --limit 5

Output (gitignored, resumable – already-processed clips are skipped):
  data/features/<task>_clips.csv          one row per clip: meta + 88 eGeMAPS + app features + emotion
  data/features/emb/<task>/<clip>.npy     1024-d wav2vec2 embedding per clip
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from _venv import ensure_backend_venv  # noqa: E402 (script dir is on sys.path)

ensure_backend_venv()

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.pipeline import audio, emotion, prosody  # noqa: E402

from corpus import FEATURES, corpus_dir, emb_name, list_files, parse_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["interview", "reading"], default="interview")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--no-emotion", action="store_true", help="skip wav2vec2 (prosody only)")
    args = ap.parse_args()

    base = corpus_dir()
    files = list_files(args.task)
    if args.limit:
        files = files[: args.limit]
    out_csv = FEATURES / f"{args.task}_clips.csv"
    emb_dir = FEATURES / "emb" / args.task
    emb_dir.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    header: list[str] | None = None
    if out_csv.exists():
        with out_csv.open() as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            done = {r["clip"] for r in reader}
        if ("app_arousal" in header) == args.no_emotion:
            sys.exit(f"{out_csv} was made with a different --no-emotion setting; delete it to start over")
    todo = [p for p in files if str(p.relative_to(base)) not in done]
    print(f"{args.task}: {len(files)} clips, {len(done)} done, {len(todo)} to go (emotion={'off' if args.no_emotion else emotion._device()})")

    writer = None
    fh = out_csv.open("a", newline="")
    t0, skipped = time.time(), 0
    try:
        for i, path in enumerate(todo, 1):
            meta = parse_meta(path)
            if meta is None:
                skipped += 1
                continue
            x = audio.load(str(path))
            full = prosody.extract_full_signal(x)
            row = {"clip": str(path.relative_to(base)), **meta, "duration_s": len(x) / audio.SR}
            row.update({f"app_{k}": v for k, v in prosody.from_full(full).items()})
            if not args.no_emotion:
                emb, emo = emotion.run_signal(x)
                row.update({f"app_{k}": v for k, v in emo.items()})
                np.save(emb_dir / emb_name(row["clip"]), emb.astype(np.float32))
            row.update({f"eg_{k}": v for k, v in full.items()})
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=header or list(row))
                if header is None:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f} clips/s  eta {(len(todo) - i) / rate / 60:.1f} min", flush=True)
    finally:
        fh.close()
    print(f"done → {out_csv}  (skipped {skipped} clips with unknown condition)")


if __name__ == "__main__":
    main()
