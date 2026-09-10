"""Production configuration and the storage reservation, checked directly.

Two small behaviours that only differ between a developer's machine and a
served one, which is exactly why neither had a test: everything passes locally
either way.

Both were found by the Phase 10 audit and are fixed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import StorageFullError
from app.services.storage_service import StorageService


def settings_for(tmp_path: Path, environment: str, *, cap_gb: float | None = None) -> Settings:
    """Settings in one environment, pointed at an isolated tree.

    `Settings` is frozen, so a smaller storage cap has to be supplied here
    rather than assigned afterwards.
    """
    extra = {} if cap_gb is None else {"max_total_storage_gb": cap_gb}
    return Settings(
        environment=environment,
        storage_dir=tmp_path / "storage",
        models_dir=tmp_path / "models",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        **extra,  # type: ignore[arg-type]
    )


# ------------------------------------------------------- the environment flag


@pytest.mark.parametrize(
    ("environment", "expected"),
    [("production", True), ("development", False), ("test", False)],
)
def test_only_production_counts_as_production(
    tmp_path: Path, environment: str, expected: bool
) -> None:
    """`test` is deliberately not production.

    The suite asserts on developer-facing behaviour - a traceback in an
    unexpected-error body, the docs being mounted - and would have to be written
    twice if running the tests changed those.
    """
    assert settings_for(tmp_path, environment).is_production is expected


def test_the_default_environment_is_development() -> None:
    """A local run keeps its docs and its tracebacks without configuring
    anything. Serving people is the case that has to be opted into.

    Read off the model rather than an instance: the suite sets
    `ENVIRONMENT=test`, and an environment variable outranks the field default.
    """
    assert Settings.model_fields["environment"].default == "development"


# ------------------------------------------------------- storage reservation


def fill_storage(service: StorageService, settings: Settings, bytes_used: int) -> None:
    """Put `bytes_used` of real file into the tree, so `total_bytes` sees it."""
    settings.ensure_directories()
    (settings.inputs_dir / "filler.bin").write_bytes(b"\0" * bytes_used)


def test_capacity_reserves_the_upload_it_is_about_to_accept(tmp_path: Path) -> None:
    """The gap this closes.

    `assert_capacity()` with no argument asks "is there room right now", which
    is satisfied at one byte under the cap and then exceeded by the very next
    file. Asking with the upload limit reserves the worst case instead.
    """
    # ~1 MB, so the test writes little to fill it.
    settings = settings_for(tmp_path, "test", cap_gb=0.001)
    service = StorageService(settings)

    limit = int(settings.max_total_storage_gb * 1024**3)
    # One byte of headroom: enough for "is there room", not for an upload.
    fill_storage(service, settings, limit - 1)

    # The old call still passes, which is what made the gap invisible.
    service.assert_capacity()

    with pytest.raises(StorageFullError):
        service.assert_capacity(settings.max_upload_size_bytes)


def test_capacity_still_accepts_a_reservation_that_fits(tmp_path: Path) -> None:
    """The reservation must not refuse an empty tree."""
    settings = settings_for(tmp_path, "test")
    service = StorageService(settings)
    settings.ensure_directories()

    service.assert_capacity(settings.max_upload_size_bytes)


def test_a_refusal_names_the_numbers_without_naming_a_path(tmp_path: Path) -> None:
    """The message reaches a user, so it carries sizes and not the disk layout."""
    settings = settings_for(tmp_path, "test", cap_gb=0.001)
    service = StorageService(settings)

    limit = int(settings.max_total_storage_gb * 1024**3)
    fill_storage(service, settings, limit - 1)

    with pytest.raises(StorageFullError) as raised:
        service.assert_capacity(settings.max_upload_size_bytes)

    error = raised.value
    assert error.status_code == 507
    assert error.context is not None
    assert error.context["limitBytes"] == limit
    assert str(tmp_path) not in error.message


# ------------------------------------------------------- the interactive docs


def routes_of(environment: str, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Every path a freshly built app serves in one environment.

    Built through `create_app` rather than read off configuration, because the
    claim is about what is reachable and `docs_url=None` unmounts a route
    rather than hiding it.

    Walked recursively: the API lives behind an included router, so a top-level
    scan sees the docs and nothing else - which would make "the docs are absent"
    pass against an app that served nothing at all. Recursion is on anything
    carrying `routes` rather than on a particular class, because the wrapper
    FastAPI uses for an included router is private and has changed name before.
    """
    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("ENVIRONMENT", environment)
    get_settings.cache_clear()
    try:
        found: set[str] = set()

        def walk(routes: object) -> None:
            for route in routes:  # type: ignore[attr-defined]
                path = getattr(route, "path", None)
                if isinstance(path, str):
                    found.add(path)
                # An included router is wrapped, and the wrapper keeps the
                # real thing on `original_router` rather than exposing routes.
                inner = getattr(route, "original_router", None)
                nested = getattr(inner if inner is not None else route, "routes", None)
                if nested is not None:
                    walk(nested)

        walk(create_app().routes)
        return found
    finally:
        get_settings.cache_clear()


DOC_PATHS = ["/docs", "/redoc", "/openapi.json"]


@pytest.mark.parametrize("path", DOC_PATHS)
def test_the_docs_are_mounted_for_a_developer(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local run keeps them; they are the reason the API is explorable."""
    assert path in routes_of("development", monkeypatch)


@pytest.mark.parametrize("path", DOC_PATHS)
def test_the_docs_are_absent_in_production(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unmounted rather than hidden.

    In production these are a map of the API for anyone who reaches the host.
    The case this closes is a tunnel or proxy pointed at the API instead of at
    the frontend, where `/docs` would otherwise answer.
    """
    assert path not in routes_of("production", monkeypatch)


def test_production_still_serves_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard against unmounting more than intended.

    Without this, "no docs in production" would be satisfied by an app that
    mounted nothing, which is the failure mode a negative assertion invites.
    """
    paths = routes_of("production", monkeypatch)

    # Without the `/api` prefix: that is applied by the parent router, and the
    # walk collects each route's own path rather than reassembling the tree.
    assert "/health" in paths
    assert "/models" in paths
    assert "/jobs" in paths
    assert "/jobs/{job_id}/events" in paths


def test_only_the_docs_differ_between_the_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production serves the same API surface, minus the documentation."""
    development = routes_of("development", monkeypatch)
    production = routes_of("production", monkeypatch)

    removed = development - production
    assert removed == {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
    assert production - development == set()
