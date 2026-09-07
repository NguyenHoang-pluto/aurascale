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
from app.repositories.base import JobRepository
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.services.system_service import SystemService

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
