from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    database_url: str = f"sqlite:///{DATA_DIR / 'app.db'}"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Rolling baseline (last N days) and EWMA smoothing – see docs/plan.md §4
    baseline_days: int = 14
    rolling_min_n: int = 7  # check-ins needed before the rolling score is computed (T10)
    ewma_alpha: float = 0.3
    # Fixed "healthy" baseline after TMS (STATUS.md T1). Overridable per baseline via the API (demo).
    healthy_period_days: int = 14
    healthy_min_checkins: int = 10  # per slot; slots with fewer fall back to all slots pooled
    min_feature_n: int = 5  # a feature needs this many baseline values (capped at the baseline's min_checkins)
    # Shrink personal SD toward the corpus within-person SD: sigma^2 = (n s^2 + k s0^2) / (n + k)
    sd_prior_strength: float = 5.0


settings = Settings()
