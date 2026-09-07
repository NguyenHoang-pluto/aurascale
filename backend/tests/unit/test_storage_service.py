"""Storage service tests, exercised against a real temporary filesystem."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import StorageFullError
from app.services.storage_service import StorageService, new_job_id


def test_new_job_id_is_hex_and_unique() -> None:
    ids = {new_job_id() for _ in range(200)}

    assert len(ids) == 200
    for job_id in ids:
        assert len(job_id) == 32
        assert all(character in "0123456789abcdef" for character in job_id)


def test_paths_are_derived_from_the_job_id(settings: Settings) -> None:
    service = StorageService(settings)
    job_id = new_job_id()

    paths = service.paths_for(job_id, input_ext="png", output_ext="jpg")

    assert paths.input.name == f"{job_id}.png"
    assert paths.output.name == f"{job_id}.jpg"
    assert paths.thumbnail.name == f"{job_id}.webp"
    assert paths.input.parent == settings.inputs_dir
    assert paths.output.parent == settings.outputs_dir


@pytest.mark.parametrize(
    "hostile_id",
    [
        "../../etc/passwd",
        "..",
        "a/b",
        "job id",
        "AABBCC",  # uppercase is not produced by new_job_id
        "",
        "zzz",
    ],
)
def test_paths_reject_anything_that_is_not_a_generated_id(
    settings: Settings, hostile_id: str
) -> None:
    """Filenames must never be derived from client input (§ 16)."""
    with pytest.raises(ValueError, match="lowercase hex"):
        StorageService(settings).paths_for(hostile_id, input_ext="png", output_ext="png")


@pytest.mark.parametrize("extension", ["exe", "php", "../png", "svg", "gif"])
def test_paths_reject_unsupported_extensions(settings: Settings, extension: str) -> None:
    with pytest.raises(ValueError, match="unsupported extension"):
        StorageService(settings).paths_for(new_job_id(), input_ext=extension, output_ext="png")


def test_extensions_are_normalised(settings: Settings) -> None:
    paths = StorageService(settings).paths_for(new_job_id(), input_ext=".PNG", output_ext="JPEG")

    assert paths.input.suffix == ".png"
    assert paths.output.suffix == ".jpeg"


def test_is_within_storage_detects_escapes(settings: Settings, tmp_path: Path) -> None:
    service = StorageService(settings)

    assert service.is_within_storage(settings.inputs_dir / "a.png") is True
    assert service.is_within_storage(settings.storage_dir / ".." / "outside.png") is False
    assert service.is_within_storage(tmp_path / "elsewhere" / "a.png") is False


def test_delete_removes_files_inside_storage(settings: Settings) -> None:
    service = StorageService(settings)
    target = settings.inputs_dir / "a.png"
    target.write_bytes(b"data")

    assert service.delete(target) == 1
    assert not target.exists()


def test_delete_ignores_files_that_are_already_gone(settings: Settings) -> None:
    service = StorageService(settings)

    assert service.delete(settings.inputs_dir / "never-existed.png") == 0


def test_delete_refuses_paths_outside_storage(settings: Settings, tmp_path: Path) -> None:
    outsider = tmp_path / "important.txt"
    outsider.write_text("do not delete me", encoding="utf-8")

    removed = StorageService(settings).delete(outsider)

    assert removed == 0
    assert outsider.exists(), "a path outside the storage tree must never be deleted"


def test_delete_skips_none_entries(settings: Settings) -> None:
    assert StorageService(settings).delete(None, None) == 0


def test_total_bytes_sums_the_tree(settings: Settings) -> None:
    service = StorageService(settings)
    (settings.inputs_dir / "a.png").write_bytes(b"x" * 100)
    (settings.outputs_dir / "b.png").write_bytes(b"y" * 250)

    assert service.total_bytes() == 350


def test_assert_capacity_allows_usage_below_the_cap(settings: Settings) -> None:
    StorageService(settings).assert_capacity(1024)


def test_assert_capacity_refuses_when_the_cap_would_be_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MAX_TOTAL_STORAGE_GB", "0.000001")  # ~1 KB
    tight = Settings()
    tight.ensure_directories()
    (tight.inputs_dir / "big.png").write_bytes(b"x" * 2048)

    with pytest.raises(StorageFullError) as excinfo:
        StorageService(tight).assert_capacity()

    problem = excinfo.value.to_problem()
    assert problem["code"] == "storage_full"
    assert "run out of space" in problem["detail"]


def test_expiry_uses_the_configured_retention(settings: Settings) -> None:
    created = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    expiry = StorageService(settings).expiry_for(created)

    assert (expiry - created).total_seconds() == settings.temp_retention_hours * 3600


def test_sweep_orphans_removes_only_unknown_files(settings: Settings) -> None:
    service = StorageService(settings)
    known = new_job_id()
    orphan = new_job_id()
    (settings.inputs_dir / f"{known}.png").write_bytes(b"keep")
    (settings.inputs_dir / f"{orphan}.png").write_bytes(b"drop")
    (settings.outputs_dir / f"{orphan}.png").write_bytes(b"drop")

    removed = service.sweep_orphans({known})

    assert removed == 2
    assert (settings.inputs_dir / f"{known}.png").exists()
    assert not (settings.inputs_dir / f"{orphan}.png").exists()
