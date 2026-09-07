"""Shared pytest fixtures.

Tests run against a real SQLite database created by the real Alembic
migrations, not `create_all`. The migration chain is therefore exercised on
every run and cannot silently drift from the ORM models.

Settings are overridden through environment variables rather than by patching
`get_settings`, so the real configuration-loading path is what gets tested.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

os.environ.setdefault("ENVIRONMENT", "test")

from app.core import migrations
from app.core.config import Settings, get_settings
from app.core.database import create_engine
from app.main import create_app
from app.models.db import Job
from app.models.enums import JobStatus
from app.repositories.job_repository import SqlAlchemyJobRepository
from app.services.storage_service import new_job_id


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings pointed at an isolated temporary storage tree and database.

    Environment variables take precedence over .env in pydantic-settings, so
    these override whatever the developer has configured locally.
    """
    storage = tmp_path / "storage"
    models = tmp_path / "models"

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("STORAGE_DIR", str(storage))
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("AUTO_MIGRATE", "true")

    get_settings.cache_clear()
    resolved = get_settings()
    resolved.ensure_directories()

    yield resolved

    get_settings.cache_clear()


@pytest.fixture
async def migrated_settings(settings: Settings) -> Settings:
    """Settings whose database has had the real migrations applied."""
    await migrations.upgrade_to_head(settings)
    return settings


@pytest.fixture
async def session_factory(
    migrated_settings: Settings,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_engine(migrated_settings)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as active:
        yield active


@pytest.fixture
async def repository(session: AsyncSession) -> SqlAlchemyJobRepository:
    return SqlAlchemyJobRepository(session)


def make_job(**overrides: object) -> Job:
    """A Job with realistic defaults, so tests state only what they care about."""
    fields: dict[str, object] = {
        "id": new_job_id(),
        "status": JobStatus.QUEUED,
        "model_name": "RealESRGAN_x4plus",
        "scale": 4,
        "output_format": "png",
        "input_path": "inputs/example.png",
        "input_width": 1280,
        "input_height": 720,
        "input_bytes": 1_887_437,
        "input_format": "PNG",
    }
    fields.update(overrides)
    return Job(**fields)


@pytest.fixture
def app(settings: Settings):  # type: ignore[no-untyped-def]
    """Application built against the test settings."""
    del settings  # ensures the fixture ordering applies the env overrides first
    return create_app()
