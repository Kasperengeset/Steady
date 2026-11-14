import os
import sys
import tempfile
from pathlib import Path

# Isolated DB per test session – must be set before the app (and its settings) is imported
# Tests never call Gemini: an empty key in the environment overrides the one in .env
os.environ["GEMINI_API_KEY"] = ""
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
