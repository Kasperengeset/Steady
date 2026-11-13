from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    database_url: str = f"sqlite:///{DATA_DIR / 'app.db'}"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Baseline window (days) and EWMA smoothing – see docs/plan.md §4
    baseline_days: int = 14
    ewma_alpha: float = 0.3


settings = Settings()
