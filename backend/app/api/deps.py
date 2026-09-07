"""FastAPI dependencies.

Wiring only. Anything with behaviour belongs in a service, and anything that
touches the database goes through a repository.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_session_factory
from app.core.runtime import runtime
from app.repositories.base import JobRepository
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.image_service import ImageService
from app.services.job_service import JobService
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.services.system_service import SystemService
from app.workers.progress import ProgressBroker
from app.workers.queue import JobQueue

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request, committed on success and rolled back on error.

    The commit lives here rather than in each route so a handler that raises
    after a partial write cannot leave the change behind.
    """
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_job_repository(session: SessionDep) -> JobRepository:
    """Repository bound to the request's session.

    Returns the interface, not the implementation, so route code cannot reach
    past the abstraction into SQLAlchemy.
    """
    return SqlAlchemyJobRepository(session)


JobRepositoryDep = Annotated[JobRepository, Depends(get_job_repository)]


def get_system_service(settings: SettingsDep) -> SystemService:
    return SystemService(settings)


def get_model_service(settings: SettingsDep) -> ModelService:
    return ModelService(settings)


def get_storage_service(settings: SettingsDep) -> StorageService:
    return StorageService(settings)


SystemServiceDep = Annotated[SystemService, Depends(get_system_service)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]


def get_progress_broker() -> ProgressBroker:
    """The broker the worker publishes into and SSE streams read from.

    Held on the application state rather than constructed per request: the
    subscribers are the point, and a fresh broker would have none.
    """
    return runtime().broker


def get_job_queue() -> JobQueue:
    return runtime().queue


def get_image_service(settings: SettingsDep) -> ImageService:
    return ImageService(settings)


def get_job_service(
    settings: SettingsDep,
    repository: JobRepositoryDep,
    images: Annotated[ImageService, Depends(get_image_service)],
    storage: StorageServiceDep,
    models: ModelServiceDep,
) -> JobService:
    """A job service bound to this request's transaction."""
    current = runtime()

    return JobService(
        settings,
        repository,
        queue=current.queue,
        broker=current.broker,
        images=images,
        storage=storage,
        models=models,
        enhancement=current.enhancement,
    )


ProgressBrokerDep = Annotated[ProgressBroker, Depends(get_progress_broker)]
JobQueueDep = Annotated[JobQueue, Depends(get_job_queue)]
ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
