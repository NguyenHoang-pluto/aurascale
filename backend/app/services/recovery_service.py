"""Startup recovery for jobs interrupted by a restart.

The in-process job queue lives in memory (`docs/architecture.md` § 3), so a job
that was queued or running when the process died has no worker to resume it.
Leaving those rows as "processing" would show a progress bar that never moves.

They are marked failed with an explanation instead — an honest terminal state
the user can act on by re-submitting.
"""

from __future__ import annotations

from app.core.exceptions import ErrorCode
from app.core.logging import get_logger
from app.models.db import utcnow
from app.models.enums import JobStatus
from app.repositories.base import JobRepository

logger = get_logger(__name__)

INTERRUPTED_MESSAGE = (
    "This job was interrupted when the server restarted. Submit the image again to retry it."
)


async def fail_interrupted_jobs(repository: JobRepository) -> int:
    """Mark queued and processing jobs as failed. Returns how many were changed."""
    stale = await repository.list_by_status(JobStatus.QUEUED, JobStatus.PROCESSING)

    for job in stale:
        await repository.update(
            job.id,
            status=JobStatus.FAILED,
            stage=None,
            error_code=ErrorCode.INTERNAL_ERROR.value,
            error_message=INTERRUPTED_MESSAGE,
            error_technical=f"process restarted while job was {job.status}",
            finished_at=utcnow(),
        )

    if stale:
        logger.warning(
            "failed interrupted jobs on startup",
            extra={"count": len(stale), "job_ids": ",".join(job.id for job in stale)},
        )

    return len(stale)
