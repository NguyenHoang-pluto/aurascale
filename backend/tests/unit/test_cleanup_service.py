"""Retention sweeping, against a real database and real files."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.repositories.base import JobRepository
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.cleanup_service import CleanupService
from app.services.storage_service import StorageService
from tests.conftest import make_job


@pytest.fixture
def scope(session_factory: async_sessionmaker[AsyncSession]):  # type: ignore[no-untyped-def]
    """A repository scope over the test's database."""

    @asynccontextmanager
    async def factory() -> AsyncIterator[JobRepository]:
        async with session_factory() as session:
            repository = SqlAlchemyJobRepository(session)
            yield repository
            await session.commit()

    return factory


async def test_an_expired_job_and_its_files_are_removed(
    migrated_settings: Settings,
    scope,
    session_factory,  # type: ignore[no-untyped-def]
) -> None:
    storage = StorageService(migrated_settings)
    storage.ensure_ready()

    job = make_job(expires_at=datetime.now(UTC) - timedelta(hours=1))
    paths = storage.paths_for(job.id, input_ext="png", output_ext="png")
    paths.input.write_bytes(b"input")
    paths.output.write_bytes(b"output")
    job.input_path = str(paths.input)
    job.output_path = str(paths.output)

    async with session_factory() as session:
        await SqlAlchemyJobRepository(session).create(job)
        await session.commit()

    result = await CleanupService(migrated_settings, scope, storage).sweep()

    assert result.expired_jobs == 1
    assert not paths.input.exists()
    assert not paths.output.exists()

    async with session_factory() as session:
        assert await SqlAlchemyJobRepository(session).get(job.id) is None


async def test_a_job_inside_its_window_is_left_alone(
    migrated_settings: Settings,
    scope,
    session_factory,  # type: ignore[no-untyped-def]
) -> None:
    job = make_job(expires_at=datetime.now(UTC) + timedelta(hours=5))

    async with session_factory() as session:
        await SqlAlchemyJobRepository(session).create(job)
        await session.commit()

    result = await CleanupService(migrated_settings, scope).sweep()

    assert result.expired_jobs == 0
    async with session_factory() as session:
        assert await SqlAlchemyJobRepository(session).get(job.id) is not None


async def test_orphaned_files_are_removed(
    migrated_settings: Settings,
    scope,  # type: ignore[no-untyped-def]
) -> None:
    """These accumulate when a process dies between writing a file and its row."""
    storage = StorageService(migrated_settings)
    storage.ensure_ready()
    orphan = migrated_settings.outputs_dir / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
    orphan.write_bytes(b"nobody owns this")

    result = await CleanupService(migrated_settings, scope, storage).sweep()

    assert result.orphan_files == 1
    assert not orphan.exists()


async def test_files_belonging_to_a_live_job_are_not_swept(
    migrated_settings: Settings,
    scope,
    session_factory,  # type: ignore[no-untyped-def]
) -> None:
    storage = StorageService(migrated_settings)
    storage.ensure_ready()

    job = make_job(expires_at=datetime.now(UTC) + timedelta(hours=5))
    paths = storage.paths_for(job.id, input_ext="png", output_ext="png")
    paths.input.write_bytes(b"input")
    job.input_path = str(paths.input)

    async with session_factory() as session:
        await SqlAlchemyJobRepository(session).create(job)
        await session.commit()

    await CleanupService(migrated_settings, scope, storage).sweep()

    assert paths.input.exists()


async def test_a_sweep_with_nothing_to_do_is_harmless(
    migrated_settings: Settings,
    scope,  # type: ignore[no-untyped-def]
) -> None:
    StorageService(migrated_settings).ensure_ready()

    result = await CleanupService(migrated_settings, scope).sweep()

    assert result.total == 0
