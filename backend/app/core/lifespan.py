"""Application startup and shutdown.

Kept deliberately small: each subsystem added in a later phase (model manager,
job worker, cleanup sweeper) registers itself here rather than in main.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_directories()

    logger.info(
        "starting",
        extra={
            "version": __version__,
            "environment": settings.environment,
            "device_preference": settings.device,
        },
    )

    # Phase 5: database engine + repositories
    # Phase 6: model manager warm-up
    # Phase 7: job worker pool + storage cleanup task

    yield

    logger.info("shutdown complete")
