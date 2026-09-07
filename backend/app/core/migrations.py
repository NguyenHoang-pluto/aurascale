"""Programmatic Alembic access.

Migrations are applied at startup when AUTO_MIGRATE is on, which keeps local
development a single command. Production should set it false and run
`alembic upgrade head` as an explicit deploy step, so a schema change is a
deliberate act rather than a side effect of a restart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from app.core.config import BACKEND_ROOT, Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

ALEMBIC_INI = BACKEND_ROOT / "alembic.ini"
ALEMBIC_DIR = BACKEND_ROOT / "alembic"


def _config() -> Config:
    config = Config(str(ALEMBIC_INI))
    # Resolve the script location absolutely so migrations work regardless of
    # the process working directory.
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    return config


def _sync_url(settings: Settings) -> str:
    """Synchronous form of the database URL.

    Alembic's revision bookkeeping here is synchronous; the async driver is
    only needed by env.py when it runs the migrations themselves.
    """
    return settings.resolved_database_url.replace("+aiosqlite", "").replace("+asyncpg", "")


def head_revision() -> str | None:
    """Latest revision available in the migrations directory."""
    return ScriptDirectory.from_config(_config()).get_current_head()


def current_revision(settings: Settings) -> str | None:
    """Revision the database is currently at, or None if it has never migrated."""
    engine = create_engine(_sync_url(settings))
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def _upgrade_blocking(revision: str) -> None:
    command.upgrade(_config(), revision)


async def upgrade_to_head(settings: Settings) -> None:
    """Bring the database up to the latest revision.

    Run in a worker thread: alembic's env.py drives an async engine with
    `asyncio.run`, which cannot be called from inside a running event loop.
    """
    settings.ensure_directories()

    before = await asyncio.to_thread(current_revision, settings)
    head = head_revision()

    if before == head:
        logger.debug("database schema up to date", extra={"revision": head})
        return

    logger.info("applying migrations", extra={"from": before, "to": head})
    await asyncio.to_thread(_upgrade_blocking, "head")
    logger.info("migrations applied", extra={"revision": head})


async def assert_up_to_date(settings: Settings) -> None:
    """Fail loudly when the schema is behind and auto-migration is disabled.

    Serving requests against an outdated schema produces confusing errors far
    from their cause, so this refuses to start instead.
    """
    revision = await asyncio.to_thread(current_revision, settings)
    head = head_revision()

    if revision != head:
        raise RuntimeError(
            "Database schema is out of date "
            f"(at {revision or 'none'}, expected {head}). "
            "Run 'alembic upgrade head', or set AUTO_MIGRATE=true."
        )


def migrations_dir_is_present() -> bool:
    return ALEMBIC_INI.is_file() and Path(ALEMBIC_DIR / "versions").is_dir()
