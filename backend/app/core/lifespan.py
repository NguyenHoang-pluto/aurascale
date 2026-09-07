"""Application startup and shutdown.

Each subsystem registers here rather than in main.py, and each is torn down in
reverse order so a failure part-way through startup does not leak resources.

Models are deliberately not warmed up here: loading is lazy so startup stays
fast and a machine that never runs a job never pays for CUDA initialisation
(docs/architecture.md § 7).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.core import migrations
from app.core.config import get_settings
from app.core.database import dispose_engine, init_engine, session_scope
from app.core.logging import get_logger
from app.core.runtime import Runtime, repository_scope, set_runtime
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.cleanup_service import CleanupScheduler, CleanupService
from app.services.enhancement_service import EnhancementService
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.recovery_service import fail_interrupted_jobs
from app.services.storage_service import StorageService
from app.services.system_service import SystemService
from app.workers.progress import ProgressBroker
from app.workers.queue import InProcessJobQueue
from app.workers.runner import JobRunner

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    storage = StorageService(settings)

    storage.ensure_ready()

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

    models = ModelService(settings)
    broker = ProgressBroker()
    broker.bind_loop(asyncio.get_running_loop())

    enhancement = EnhancementService(settings, models=models)

    # The queue needs a handler and the runner needs the queue, so the runner
    # is built first and the queue is given a closure over it.
    runner_holder: dict[str, JobRunner] = {}

    async def handle(job_id: str) -> None:
        await runner_holder["runner"].run(job_id)

    queue = InProcessJobQueue(settings, handle)
    runner = JobRunner(
        settings,
        scope=repository_scope,
        queue=queue,
        broker=broker,
        enhancement=enhancement,
        images=ImageService(settings),
        storage=storage,
        models=models,
    )
    runner_holder["runner"] = runner

    cleanup = CleanupScheduler(settings, CleanupService(settings, repository_scope, storage))

    await queue.start()
    await cleanup.start()
    set_runtime(
        Runtime(
            queue=queue,
            broker=broker,
            enhancement=enhancement,
            runner=runner,
            cleanup=cleanup,
        )
    )

    try:
        yield
    finally:
        # Reverse order, and the queue first: running jobs are asked to stop
        # before the database they write their final state to goes away.
        await cleanup.stop()
        await queue.stop()
        enhancement.release()
        set_runtime(None)
        await dispose_engine()
        logger.info("shutdown complete")
