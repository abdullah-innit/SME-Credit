"""Engine factory, foreign-key enforcement, transactional sessions and bootstrap."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base


def make_engine(url: str, *, echo: bool = False) -> Engine:
    options = {}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            options["poolclass"] = StaticPool
    result = create_engine(url, echo=echo, **options)
    if result.dialect.name == "sqlite":

        @event.listens_for(result, "connect")
        def enable_foreign_keys(connection, record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return result


engine = make_engine(settings.DATABASE_URL, echo=settings.DEBUG)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """Closing dependency. Routes explicitly commit their writes."""
    with SessionLocal() as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """Commit on success; rollback and close on failure."""
    with SessionLocal.begin() as session:
        yield session


def init_db(bind: Engine = engine, *, seed_hook: Callable[[Session], None] | None = None) -> None:
    from app.db import models  # noqa: F401 — register every table

    Base.metadata.create_all(bind)
    if bind.dialect.name == "sqlite":
        with bind.connect() as conn:
            cursor = conn.connection.cursor()
            cols = [r[1] for r in cursor.execute("PRAGMA table_info(ai_call_logs)").fetchall()]
            if cols:
                if "tokens" not in cols:
                    cursor.execute(
                        "ALTER TABLE ai_call_logs ADD COLUMN tokens INTEGER DEFAULT 0 NOT NULL"
                    )
                if "cost_usd" not in cols:
                    cursor.execute(
                        "ALTER TABLE ai_call_logs ADD COLUMN cost_usd REAL DEFAULT 0.0 NOT NULL"
                    )
                conn.connection.commit()
            cursor.close()
    if seed_hook is not None:
        with Session(bind) as session, session.begin():
            seed_hook(session)
