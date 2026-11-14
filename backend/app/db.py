from collections.abc import Iterator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from .config import DATA_DIR, settings

DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    from . import models, models_prompt, models_push  # noqa: F401 – register tables

    SQLModel.metadata.create_all(engine)
    add_missing_columns()


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
