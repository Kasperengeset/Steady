from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    # Spoken check-ins: verbatim transcript (pipeline/llm.py). The text model is English only.
    transcribe_model: str = "gemini-3.5-transcribe"
    transcribe_language: str = "en-US"
    # Web Push for the daily prompt (#22, P2). Only in .env – create with scripts/gen_vapid.py.
    # The subject is a contact the push services can reach: "mailto:you@…" or an https URL.
    vapid_public_key: str = ""
    vapid_private_key: str = ""
    vapid_subject: str = ""
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
    # Time of day (STATUS.md B30): "hour" = one pooled baseline + each feature's slope on CheckIn.local_hour;
    # "slots" = the old morning/evening baselines (T10). Random prompt times (#22) make slots too thin.
    time_of_day: str = "hour"
    # Shrink personal SD toward the corpus within-person SD: sigma^2 = (n s^2 + k s0^2) / (n + k)
    sd_prior_strength: float = 5.0
    # Elevated flag (clinician only): S_anchored < -elevated_S for elevated_days scored days in a row
    elevated_S: float = 1.0
    elevated_days: int = 3
    # EXPERIMENT, not used by the app: relapse HMM (engine/relapse.py, STATUS.md B19). Transition rates are assumptions;
    # δ acoustic is read from acoustic.json (held-out detectability), δ text is a rough diary estimate.
    relapse_delta: dict[str, float] = {"acoustic": 1.58, "text": 1.5, "health": 1.5}
    relapse_h_relapse: float = 0.005  # well → relapse per day
    relapse_h_elevated: float = 0.002  # well → elevated per day
    relapse_r_relapse: float = 0.02  # relapse → well per day
    relapse_r_elevated: float = 0.1  # elevated → well per day
    relapse_prior: list[float] = [0.98, 0.01, 0.01]  # at baseline lock
    # Alert thresholds on P(state) – calibrated in the simulator to a false-alarm target (see STATUS.md)
    p_friends: float = 0.3
    p_clinician: float = 0.6
    p_elevated: float = 0.6


settings = Settings()
