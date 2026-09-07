"""Model registry tests.

Exercised against both temporary manifests and the real one shipped in the
repository, so a mistake in models/manifest.json fails here rather than at
first inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import BACKEND_ROOT, Settings
from app.core.exceptions import ModelNotFoundError
from app.services.model_service import ModelService, _read_manifest

REPO_MANIFEST = BACKEND_ROOT.parent / "models" / "manifest.json"

MANIFEST = {
    "models": [
        {
            "id": "TestModel_x4",
            "name": "Test x4",
            "description": "A test model.",
            "arch": "RRDBNet",
            "scale": 4,
            "file": "test_x4.pth",
            "url": "https://example.test/test_x4.pth",
            "sha256": None,
            "arch_params": {"num_block": 23},
        },
        {
            "id": "TestModel_wdn",
            "name": "Test denoise weights",
            "description": "Not selectable on its own.",
            "arch": "SRVGGNetCompact",
            "scale": 4,
            "file": "test_wdn.pth",
            "url": "https://example.test/test_wdn.pth",
            "selectable": False,
        },
    ]
}


@pytest.fixture
def service(settings: Settings) -> ModelService:
    _read_manifest.cache_clear()
    (settings.models_dir / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return ModelService(settings)


def test_lists_models_from_the_manifest(service: ModelService) -> None:
    statuses = service.list()

    assert [status.entry.id for status in statuses] == ["TestModel_x4", "TestModel_wdn"]


def test_reports_weights_as_missing_before_download(service: ModelService) -> None:
    status = service.get("TestModel_x4")

    assert status.downloaded is False
    assert status.size_bytes is None


def test_reports_weights_present_once_on_disk(service: ModelService, settings: Settings) -> None:
    (settings.models_dir / "test_x4.pth").write_bytes(b"x" * 4096)

    status = service.get("TestModel_x4")

    assert status.downloaded is True
    assert status.size_bytes == 4096


def test_hides_non_selectable_weight_sets(service: ModelService) -> None:
    """The DNI denoise counterpart is blended, never chosen directly."""
    selectable = service.list(selectable_only=True)

    assert [status.entry.id for status in selectable] == ["TestModel_x4"]


def test_unknown_model_raises_a_specific_error(service: ModelService) -> None:
    with pytest.raises(ModelNotFoundError) as excinfo:
        service.get("NoSuchModel")

    problem = excinfo.value.to_problem()
    assert problem["code"] == "model_not_found"
    assert problem["status"] == 404
    # The message names what is available, rather than only what failed.
    assert "TestModel_x4" in problem["technical"]


def test_missing_manifest_is_reported_clearly(settings: Settings) -> None:
    _read_manifest.cache_clear()

    with pytest.raises(ModelNotFoundError, match="manifest is missing"):
        ModelService(settings).list()


def test_malformed_manifest_is_reported_clearly(settings: Settings) -> None:
    _read_manifest.cache_clear()
    (settings.models_dir / "manifest.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ModelNotFoundError, match="could not be read"):
        ModelService(settings).list()


def test_manifest_with_unexpected_fields_is_rejected(settings: Settings) -> None:
    """extra="forbid" catches a typo in the manifest at load time rather than
    letting a silently-ignored key change behaviour."""
    _read_manifest.cache_clear()
    payload = {"models": [{**MANIFEST["models"][0], "scail": 4}]}
    (settings.models_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        ModelService(settings).list()


def test_default_model_must_exist_in_the_manifest(
    service: ModelService, settings: Settings
) -> None:
    del service
    with pytest.raises(ModelNotFoundError):
        # The configured default is RealESRGAN_x4plus, absent from the test manifest.
        ModelService(settings).default_model_id()


# ------------------------------------------------------ the shipped manifest


def test_repository_manifest_is_valid() -> None:
    """The real manifest must parse and satisfy the schema."""
    _read_manifest.cache_clear()
    entries = _read_manifest(REPO_MANIFEST)

    assert len(entries) >= 3
    ids = [entry.id for entry in entries]
    assert len(ids) == len(set(ids)), "model ids must be unique"


def test_repository_manifest_covers_every_documented_scale() -> None:
    _read_manifest.cache_clear()
    entries = _read_manifest(REPO_MANIFEST)

    scales = {entry.scale for entry in entries if entry.selectable}
    # 8x is produced by chaining 4x and 2x, so both must exist.
    assert {2, 4} <= scales


def test_repository_manifest_denoise_pairs_resolve() -> None:
    _read_manifest.cache_clear()
    entries = _read_manifest(REPO_MANIFEST)
    by_id = {entry.id: entry for entry in entries}

    for entry in entries:
        if entry.denoise_pair is not None:
            assert entry.denoise_pair in by_id, (
                f"{entry.id} names a denoise pair that is not in the manifest"
            )
            assert entry.supports_denoise is True


def test_repository_manifest_urls_are_https(settings: Settings) -> None:
    del settings
    _read_manifest.cache_clear()

    for entry in _read_manifest(REPO_MANIFEST):
        assert entry.url.startswith("https://"), f"{entry.id} must download over https"


def test_repository_manifest_default_model_is_present() -> None:
    _read_manifest.cache_clear()
    ids = {entry.id for entry in _read_manifest(REPO_MANIFEST)}

    assert Settings().default_model in ids


def test_manifest_cache_is_keyed_by_path(settings: Settings, tmp_path: Path) -> None:
    """Distinct paths must not share a cache entry, or tests would leak into
    each other and a manifest edit would need a restart."""
    _read_manifest.cache_clear()
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(MANIFEST), encoding="utf-8")
    second.write_text(json.dumps({"models": [MANIFEST["models"][0]]}), encoding="utf-8")
    del settings

    assert len(_read_manifest(first)) == 2
    assert len(_read_manifest(second)) == 1
