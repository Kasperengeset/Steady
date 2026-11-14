import os
import sys
import tempfile
from pathlib import Path

# Isolated DB per test session – must be set before the app (and its settings) is imported
# Tests never call Gemini: an empty key in the environment overrides the one in .env
os.environ["GEMINI_API_KEY"] = ""
TEST_DATABASE = Path(tempfile.mkdtemp(prefix="steady-tests-")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DATABASE}"
os.environ["WARMUP_MODELS"] = "false"
os.environ["HEALTH_SYNC_ENABLED"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_sessionstart(session):
    from app.db import engine
    if Path(engine.url.database).resolve() != TEST_DATABASE.resolve():
        raise RuntimeError("Tests must use their own temporary database, never the running app database.")
