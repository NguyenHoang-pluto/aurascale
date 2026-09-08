"""Retention sweeping.

Two jobs, run on a timer:

  * delete jobs whose retention window has passed, and their files;
  * delete files with no job record, which accumulate when a process dies
    between writing a file and committing its row.

Without this, `storage/` grows until the disk is full - and `MAX_TOTAL_STORAGE_GB`
would then refuse every new job, which looks like a bug and is really a
missing broom.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.logging import get_logger
from app.repositories.base import JobRepository
from app.services.storage_service import StorageService

logger = get_logger(__name__)

RepositoryScope = Callable[[], AbstractAsyncContextManager[JobRepository]]

# How many expired jobs one pass removes. Bounded so a long-neglected instance
# does not spend minutes in a single sweep holding a write lock.
SWEEP_BATCH = 200


@dataclass(frozen=True, slots=True)
class SweepResult:
    expired_jobs: int
    orphan_files: int

    @property
    def total(self) -> int:
        return self.expired_jobs + self.orphan_files


class CleanupService:
    """Removes expired jobs and orphaned files."""

    def __init__(
        self,
        settings: Settings,
        scope: RepositoryScope,
        storage: StorageService | None = None,
    ) -> None:
        self._settings = settings
        self._scope = scope
        self._storage = storage or StorageService(settings)

    async def sweep(self, now: datetime | None = None) -> SweepResult:
        """One pass. Safe to call at any time, including concurrently with jobs."""
        moment = now or datetime.now(UTC)
        expired = 0
        files = 0

        async with self._scope() as repository:
            for job in await repository.list_expired(moment, limit=SWEEP_BATCH):
                files += self._storage.delete(
                    job.input_path,
                    job.output_path,
                    self._storage.preview_path(job.id),
                    self._storage.thumbnail_path(job.id),
                )
                await repository.delete(job.id)
                expired += 1

        orphans = await self._sweep_orphans()

        if expired or orphans:
            logger.info(
                "storage swept",
                extra={"expired_jobs": expired, "files_removed": files, "orphans": orphans},
            )

        return SweepResult(expired_jobs=expired, orphan_files=orphans)

    async def _sweep_orphans(self) -> int:
        """Remove stored files that no job refers to any more."""
        async with self._scope() as repository:
            known = await repository.list_jobs(limit=10_000, offset=0)

        stems = {job.id for job in known.items}
        return await asyncio.to_thread(self._storage.sweep_orphans, stems)


class CleanupScheduler:
    """Runs the sweeper on `CLEANUP_INTERVAL_MINUTES`."""

    def __init__(self, settings: Settings, service: CleanupService) -> None:
        self._settings = settings
        self._service = service
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:  # pragma: no cover - defensive
            return
        self._task = asyncio.create_task(self._loop(), name="storage-sweeper")
        logger.info(
            "storage sweeper started",
            extra={"interval_minutes": self._settings.cleanup_interval_minutes},
        )

    async def stop(self) -> None:
        if self._task is None:
            return

        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        interval = self._settings.cleanup_interval_minutes * 60

        while True:
            # Sleep first: startup is busy enough, and nothing can have expired
            # in the second since the process began.
            await asyncio.sleep(interval)
            try:
                await self._service.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed sweep must not kill the loop; the next pass will
                # pick up whatever this one left behind.
                logger.exception("storage sweep failed")
