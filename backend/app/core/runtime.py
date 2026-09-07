"""Process-wide components that outlive a request.

The queue, its worker pool, the progress broker and the loaded models are all
per-process, not per-request: a broker rebuilt for each request would have no
subscribers, and a model manager rebuilt for each request would reload the
weights every time.

They are assembled in `lifespan` and read here, following the same pattern as
`core/database.py` so there is one obvious place where process state lives.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.core.database import session_scope
from app.repositories.base import JobRepository
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.cleanup_service import CleanupScheduler
from app.services.enhancement_service import EnhancementService
from app.workers.progress import ProgressBroker
from app.workers.queue import InProcessJobQueue
from app.workers.runner import JobRunner


@dataclass(frozen=True, slots=True)
class Runtime:
    """Everything the job endpoints need that is not per-request."""

    queue: InProcessJobQueue
    broker: ProgressBroker
    enhancement: EnhancementService
    runner: JobRunner
    cleanup: CleanupScheduler


_runtime: Runtime | None = None


def set_runtime(value: Runtime | None) -> None:
    """Install (or clear, with None) the process runtime."""
    global _runtime
    _runtime = value


def runtime() -> Runtime:
    """The running components.

    Raises rather than building something on the fly: a lazily created queue
    would have no consumers, so jobs would be accepted and never run.
    """
    if _runtime is None:
        raise RuntimeError("The job runtime is not started; lifespan did not run.")
    return _runtime


@asynccontextmanager
async def repository_scope() -> AsyncIterator[JobRepository]:
    """A short-lived repository for background work.

    Each use is its own transaction, committed on exit, which is what makes a
    worker's progress writes visible to polling clients while the job is still
    running.
    """
    async with session_scope() as session:
        yield SqlAlchemyJobRepository(session)
