"""Scripts that import the app pipeline (opensmile, transformers, sqlmodel …) need backend/.venv.
If started with another interpreter, re-run the same command with backend/.venv/bin/python."""

import os
import sys
from pathlib import Path

VENV = Path(__file__).resolve().parents[1] / "backend" / ".venv"


def ensure_backend_venv() -> None:
    py = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists() or Path(sys.prefix).resolve() == VENV.resolve():
        return
    print(f"(switching to {py.relative_to(VENV.parents[1])} – this script needs the backend packages)", file=sys.stderr)
    if os.name == "nt":  # execv does not replace the process on Windows and breaks on spaces in paths
        import subprocess

        sys.exit(subprocess.call([str(py), *sys.argv]))
    os.execv(str(py), [str(py), *sys.argv])
