"""Repository interfaces.

Services depend on these abstractions, never on SQLAlchemy. That is what makes
the PostgreSQL migration a change of driver and URL rather than a rewrite, and
it lets service tests run against an in-memory fake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.db import Job
from app.models.enums import JobStatus


@dataclass(frozen=True, slots=True)
class Page:
    """One page of results plus the total, for history pagination."""

    items: list[Job]
    total: int
    limit: int
    offset: int


class JobRepository(ABC):
    """Persistence operations for enhancement jobs."""

    @abstractmethod
    async def create(self, job: Job) -> Job:
        """Persist a new job."""

    @abstractmethod
    async def get(self, job_id: str) -> Job | None:
        """Return a job, or None when the id is unknown."""

    @abstractmethod
    async def update(self, job_id: str, **changes: Any) -> Job | None:
        """Apply field changes and return the updated job."""

    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Remove a job. Returns False when it did not exist."""

    @abstractmethod
    async def list_jobs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: JobStatus | None = None,
    ) -> Page:
        """Jobs newest-first, optionally filtered by status."""

    @abstractmethod
    async def list_by_status(self, *statuses: JobStatus) -> list[Job]:
        """All jobs in any of the given states."""

    @abstractmethod
    async def list_expired(self, before: datetime, *, limit: int = 100) -> list[Job]:
        """Jobs whose retention window has passed, for the storage sweeper."""

    @abstractmethod
    async def count_active(self) -> int:
        """Jobs queued or processing, used to enforce the queue bound."""

    @abstractmethod
    async def commit(self) -> None:
        """Make pending changes durable and visible to other connections.

        Normally the caller's transaction scope handles this. It is explicit
        here for the one case that cannot wait: a job row has to be committed
        *before* its id is handed to the worker, or the worker can look it up
        and find nothing.
        """
