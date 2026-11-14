from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from .config import DATA_DIR, settings

DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    from . import models  # noqa: F401 – register tables
    from . import models_rewards, models_social  # noqa: F401 – #24, #26, #25

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
