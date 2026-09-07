"""Repository tests against a real migrated SQLite database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.models.enums import JobStage, JobStatus
from app.repositories.job_repository import SqlAlchemyJobRepository
from tests.conftest import make_job


async def test_create_and_get_round_trip(repository: SqlAlchemyJobRepository) -> None:
    job = make_job(model_name="RealESRGAN_x2plus", scale=2)

    created = await repository.create(job)
    fetched = await repository.get(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.model_name == "RealESRGAN_x2plus"
    assert fetched.scale == 2
    assert fetched.status == JobStatus.QUEUED
    assert fetched.progress == 0
    assert fetched.cancel_requested is False


async def test_get_returns_none_for_unknown_id(repository: SqlAlchemyJobRepository) -> None:
    assert await repository.get("does-not-exist") is None


async def test_update_applies_changes(repository: SqlAlchemyJobRepository) -> None:
    job = await repository.create(make_job())

    updated = await repository.update(
        job.id,
        status=JobStatus.PROCESSING,
        stage=JobStage.RUNNING_INFERENCE,
        progress=50,
        device="cuda",
    )

    assert updated is not None
    assert updated.status == JobStatus.PROCESSING
    assert updated.stage == JobStage.RUNNING_INFERENCE
    assert updated.progress == 50
    assert updated.device == "cuda"


async def test_update_returns_none_for_unknown_id(repository: SqlAlchemyJobRepository) -> None:
    assert await repository.update("nope", progress=10) is None


async def test_update_rejects_unknown_field(repository: SqlAlchemyJobRepository) -> None:
    job = await repository.create(make_job())

    # A typo in a field name must fail loudly rather than silently doing nothing.
    with pytest.raises(AttributeError, match="no field"):
        await repository.update(job.id, progres=50)


async def test_json_column_round_trips(repository: SqlAlchemyJobRepository) -> None:
    options = {"sharpenStrength": 0.4, "denoiseStrength": 0.0, "tileSize": None}
    job = await repository.create(make_job(enhance_options=options))

    fetched = await repository.get(job.id)

    assert fetched is not None
    assert fetched.enhance_options == options


async def test_delete_removes_the_row(repository: SqlAlchemyJobRepository) -> None:
    job = await repository.create(make_job())

    assert await repository.delete(job.id) is True
    assert await repository.get(job.id) is None


async def test_delete_reports_a_missing_row(repository: SqlAlchemyJobRepository) -> None:
    assert await repository.delete("missing") is False


async def test_list_jobs_returns_newest_first(repository: SqlAlchemyJobRepository) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(3):
        await repository.create(
            make_job(
                model_name=f"model-{index}",
                created_at=base + timedelta(minutes=index),
            )
        )

    page = await repository.list_jobs()

    assert page.total == 3
    assert [job.model_name for job in page.items] == ["model-2", "model-1", "model-0"]


async def test_list_jobs_paginates(repository: SqlAlchemyJobRepository) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(5):
        await repository.create(
            make_job(model_name=f"m{index}", created_at=base + timedelta(minutes=index))
        )

    page = await repository.list_jobs(limit=2, offset=1)

    # Total reflects the whole collection, not the page.
    assert page.total == 5
    assert page.limit == 2
    assert page.offset == 1
    assert [job.model_name for job in page.items] == ["m3", "m2"]


async def test_list_jobs_filters_by_status(repository: SqlAlchemyJobRepository) -> None:
    await repository.create(make_job(status=JobStatus.COMPLETED))
    await repository.create(make_job(status=JobStatus.FAILED))
    await repository.create(make_job(status=JobStatus.COMPLETED))

    page = await repository.list_jobs(status=JobStatus.COMPLETED)

    assert page.total == 2
    assert all(job.status == JobStatus.COMPLETED for job in page.items)


async def test_list_by_status_accepts_several_states(
    repository: SqlAlchemyJobRepository,
) -> None:
    await repository.create(make_job(status=JobStatus.QUEUED))
    await repository.create(make_job(status=JobStatus.PROCESSING))
    await repository.create(make_job(status=JobStatus.COMPLETED))

    active = await repository.list_by_status(JobStatus.QUEUED, JobStatus.PROCESSING)

    assert len(active) == 2


async def test_list_by_status_with_no_states_returns_nothing(
    repository: SqlAlchemyJobRepository,
) -> None:
    await repository.create(make_job())

    assert await repository.list_by_status() == []


async def test_count_active_counts_only_unfinished_jobs(
    repository: SqlAlchemyJobRepository,
) -> None:
    await repository.create(make_job(status=JobStatus.QUEUED))
    await repository.create(make_job(status=JobStatus.PROCESSING))
    await repository.create(make_job(status=JobStatus.COMPLETED))
    await repository.create(make_job(status=JobStatus.CANCELLED))

    assert await repository.count_active() == 2


async def test_list_expired_selects_only_past_expiry(
    repository: SqlAlchemyJobRepository,
) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    await repository.create(make_job(expires_at=now - timedelta(hours=1)))
    await repository.create(make_job(expires_at=now + timedelta(hours=1)))
    # A job with no expiry must never be swept.
    await repository.create(make_job(expires_at=None))

    expired = await repository.list_expired(now)

    assert len(expired) == 1
    assert expired[0].expires_at is not None
