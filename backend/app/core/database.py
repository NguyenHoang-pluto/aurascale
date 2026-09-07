"""Database engine and session management.

The engine is created once at startup and disposed at shutdown. Nothing outside
this module and the repositories imports SQLAlchemy, which is what keeps the
PostgreSQL migration a configuration change (`docs/architecture.md` § 2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _is_memory(url: str) -> bool:
    return ":memory:" in url


def create_engine(settings: Settings) -> AsyncEngine:
    """Build an engine for the configured database."""
    url = settings.resolved_database_url

    kwargs: dict[str, object] = {"echo": False, "future": True}
    if _is_sqlite(url) and _is_memory(url):
        # An in-memory SQLite database lives inside its connection, so tests
        # must share exactly one connection or each session sees an empty
        # database.
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_async_engine(url, **kwargs)

    if _is_sqlite(url):

        @event.listens_for(engine.sync_engine, "connect")
        def _configure_sqlite(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                # WAL lets the progress writer and API readers work
                # concurrently instead of blocking each other.
                cursor.execute("PRAGMA journal_mode=WAL")
                # SQLite does not enforce foreign keys unless asked to.
                cursor.execute("PRAGMA foreign_keys=ON")
                # Wait rather than raising "database is locked" the instant a
                # writer holds the lock.
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    return engine


def init_engine(settings: Settings) -> AsyncEngine:
    """Create the process-wide engine and session factory."""
    global _engine, _session_factory

    _engine = create_engine(settings)
    _session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        autoflush=False,
    )
    logger.info(
        "database ready",
        extra={"dialect": _engine.dialect.name, "driver": _engine.dialect.driver},
    )
    return _engine


async def dispose_engine() -> None:
    """Close pooled connections at shutdown."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database is not initialised; init_engine was not called.")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session scope, committing on success and rolling back on error.

    Used by background workers, which have no request to hang a dependency off.
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
