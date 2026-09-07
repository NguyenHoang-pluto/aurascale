"""Application startup and shutdown.

Each subsystem registers here rather than in main.py, and each is torn down in
reverse order so a failure part-way through startup does not leak resources.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.core import migrations
from app.core.config import get_settings
from app.core.database import dispose_engine, init_engine, session_scope
from app.core.logging import get_logger
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.recovery_service import fail_interrupted_jobs
from app.services.storage_service import StorageService
from app.services.system_service import SystemService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    StorageService(settings).ensure_ready()

    logger.info(
        "starting",
        extra={"version": __version__, "environment": settings.environment},
    )

    # Logged once at startup so a support question about slow jobs can be
    # answered from the log alone.
    logger.info("execution environment", extra=SystemService(settings).describe_for_log())

    if settings.auto_migrate:
        await migrations.upgrade_to_head(settings)
    else:
        await migrations.assert_up_to_date(settings)

    init_engine(settings)

    async with session_scope() as session:
        await fail_interrupted_jobs(SqlAlchemyJobRepository(session))

    # Phase 6: model manager warm-up
    # Phase 7: job worker pool and storage cleanup task

    try:
        yield
    finally:
        await dispose_engine()
        logger.info("shutdown complete")
