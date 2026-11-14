import os
import sys
import tempfile
from pathlib import Path

# Isolated DB per test session – must be set before the app (and its settings) is imported
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
