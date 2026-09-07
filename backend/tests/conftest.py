"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Iterator[Settings]:
    """Settings pointed at an isolated temporary storage tree."""
    get_settings.cache_clear()
    overridden = Settings(
        environment="test",
        storage_dir=tmp_path / "storage",
        models_dir=tmp_path / "models",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
    )
    overridden.ensure_directories()
    get_settings.cache_clear()
    yield overridden
    get_settings.cache_clear()


@pytest.fixture
def app():  # type: ignore[no-untyped-def]
    return create_app()
