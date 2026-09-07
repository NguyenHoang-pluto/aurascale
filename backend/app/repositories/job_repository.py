"""SQLAlchemy implementation of the job repository."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Job
from app.models.enums import JobStatus
from app.repositories.base import JobRepository, Page


class SqlAlchemyJobRepository(JobRepository):
    """Job persistence backed by SQLAlchemy.

    Holds a session rather than creating one, so a caller can compose several
    repository calls into a single transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job: Job) -> Job:
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(self, job_id: str) -> Job | None:
        return await self._session.get(Job, job_id)

    async def update(self, job_id: str, **changes: Any) -> Job | None:
        job = await self._session.get(Job, job_id)
        if job is None:
            return None

        for field, value in changes.items():
            if not hasattr(job, field):
                raise AttributeError(f"Job has no field {field!r}")
            setattr(job, field, value)

        await self._session.flush()
        return job

    async def delete(self, job_id: str) -> bool:
        job = await self._session.get(Job, job_id)
        if job is None:
            return False

        await self._session.delete(job)
        await self._session.flush()
        return True

    async def list_jobs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: JobStatus | None = None,
    ) -> Page:
        conditions = [Job.status == status] if status is not None else []

        total = await self._session.scalar(select(func.count()).select_from(Job).where(*conditions))

        rows = await self._session.scalars(
            select(Job)
            .where(*conditions)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(limit)
            .offset(offset)
        )

        return Page(items=list(rows), total=total or 0, limit=limit, offset=offset)

    async def list_by_status(self, *statuses: JobStatus) -> list[Job]:
        if not statuses:
            return []
        rows = await self._session.scalars(
            select(Job).where(Job.status.in_(statuses)).order_by(Job.created_at)
        )
        return list(rows)

    async def list_expired(self, before: datetime, *, limit: int = 100) -> list[Job]:
        rows = await self._session.scalars(
            select(Job)
            .where(Job.expires_at.is_not(None), Job.expires_at < before)
            .order_by(Job.expires_at)
            .limit(limit)
        )
        return list(rows)

    async def commit(self) -> None:
        await self._session.commit()

    async def count_active(self) -> int:
        total = await self._session.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING]))
        )
        return total or 0
