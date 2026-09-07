"""Startup recovery of jobs interrupted by a restart."""

from __future__ import annotations

from app.core.exceptions import ErrorCode
from app.models.enums import JobStage, JobStatus
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.recovery_service import fail_interrupted_jobs
from tests.conftest import make_job


async def test_marks_queued_and_processing_jobs_as_failed(
    repository: SqlAlchemyJobRepository,
) -> None:
    queued = await repository.create(make_job(status=JobStatus.QUEUED))
    running = await repository.create(
        make_job(status=JobStatus.PROCESSING, stage=JobStage.RUNNING_INFERENCE, progress=40)
    )

    changed = await fail_interrupted_jobs(repository)

    assert changed == 2
    for job_id in (queued.id, running.id):
        job = await repository.get(job_id)
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.stage is None
        assert job.finished_at is not None


async def test_explains_the_failure_in_terms_the_user_can_act_on(
    repository: SqlAlchemyJobRepository,
) -> None:
    job = await repository.create(make_job(status=JobStatus.PROCESSING))

    await fail_interrupted_jobs(repository)

    recovered = await repository.get(job.id)
    assert recovered is not None
    assert recovered.error_code == ErrorCode.INTERNAL_ERROR.value
    assert "server restarted" in (recovered.error_message or "")
    assert "Submit the image again" in (recovered.error_message or "")
    assert recovered.error_technical is not None


async def test_leaves_finished_jobs_untouched(
    repository: SqlAlchemyJobRepository,
) -> None:
    completed = await repository.create(
        make_job(status=JobStatus.COMPLETED, progress=100, processing_ms=8420)
    )
    failed = await repository.create(make_job(status=JobStatus.FAILED))
    cancelled = await repository.create(make_job(status=JobStatus.CANCELLED))

    changed = await fail_interrupted_jobs(repository)

    assert changed == 0
    untouched = await repository.get(completed.id)
    assert untouched is not None
    assert untouched.status == JobStatus.COMPLETED
    assert untouched.processing_ms == 8420
    assert (await repository.get(failed.id)).status == JobStatus.FAILED  # type: ignore[union-attr]
    assert (await repository.get(cancelled.id)).status == JobStatus.CANCELLED  # type: ignore[union-attr]


async def test_is_a_no_op_on_an_empty_database(
    repository: SqlAlchemyJobRepository,
) -> None:
    assert await fail_interrupted_jobs(repository) == 0


async def test_is_idempotent(repository: SqlAlchemyJobRepository) -> None:
    """A second startup must not re-process what the first already handled."""
    await repository.create(make_job(status=JobStatus.PROCESSING))

    assert await fail_interrupted_jobs(repository) == 1
    assert await fail_interrupted_jobs(repository) == 0
