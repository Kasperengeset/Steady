from collections.abc import Iterator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from .config import DATA_DIR, settings

DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    from . import models, models_prompt, models_push  # noqa: F401 – register tables
    from . import models_rewards, models_social  # noqa: F401 – #24, #26, #25
    from . import health_sync, models_appie  # noqa: F401 – wearable sync and avatar state

    SQLModel.metadata.create_all(engine)
    add_missing_columns()
    _required_columns()


def add_missing_columns() -> None:
    """create_all() makes new tables but never alters existing ones. Add any new *nullable* column to
    an existing SQLite table, so an old data/app.db keeps working after a model gains a field.
    Shared by every feature branch: add the field to the model as `X | None = None` and that is all."""
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have or not col.nullable or col.primary_key:
                    continue
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def _required_columns() -> None:
    """Add nonnullable fields introduced by merged branches without replacing existing tables."""
    if engine.dialect.name != "sqlite":
        return
    columns = {
        "statescore": {"evidence_json": "TEXT NOT NULL DEFAULT '{}'", "elevated": "BOOLEAN NOT NULL DEFAULT 0"},
        "dailyprompt": {"source": "TEXT NOT NULL DEFAULT 'random'"},
        "escalation": {"handled": "BOOLEAN NOT NULL DEFAULT 0"},
    }
    with engine.begin() as conn:
        for table, fields in columns.items():
            existing = {row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table}")')}
            for name, ddl in fields.items():
                if name not in existing:
                    conn.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {ddl}')
        conn.exec_driver_sql('CREATE UNIQUE INDEX IF NOT EXISTS uq_appie_checkin_request ON "checkin" (patient_id, request_key)')
