"""Weight downloading: verification, and what happens when it goes wrong.

No network is touched. The fetch step is replaced with one that writes a local
file, so what is under test is the part that decides whether a download is
trustworthy - which is the part that matters, and the part that is hard to
exercise against a real 64 MB transfer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import ModelDownloadError
from app.services.model_download_service import ModelDownloadService, sha256_of
from app.services.model_service import ModelService

PAYLOAD = b"pretend these are weights"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def write_manifest(models_dir: Path, **overrides: Any) -> None:
    entry: dict[str, Any] = {
        "id": "demo",
        "name": "Demo model",
        "description": "",
        "arch": "SRVGGNetCompact",
        "scale": 4,
        "file": "demo.pth",
        "url": "https://example.invalid/demo.pth",
        "sha256": None,
        "arch_params": {},
    }
    entry.update(overrides)

    pair: dict[str, Any] = {
        **entry,
        "id": "demo-wdn",
        "file": "demo-wdn.pth",
        "selectable": False,
        "supports_denoise": False,
        "denoise_pair": None,
    }
    (models_dir / "manifest.json").write_text(
        json.dumps({"models": [entry, pair]}), encoding="utf-8"
    )


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    return directory


def make_settings(models_dir: Path) -> Settings:
    return Settings(
        environment="test", models_dir=models_dir, storage_dir=models_dir.parent / "storage"
    )


def stub_fetch(service: ModelDownloadService, content: bytes = PAYLOAD) -> None:
    """Replace the network step with a local write of `content`."""

    def fetch(entry: Any, *, on_progress: Any = None) -> Path:
        temporary = Path(service._settings.models_dir) / ".stub.part"
        temporary.write_bytes(content)
        return temporary

    service._fetch_to_temporary = fetch  # type: ignore[method-assign]


def test_a_download_with_no_recorded_digest_is_refused(models_dir: Path) -> None:
    """An unverified checkpoint is exactly what a supply-chain problem looks
    like, so it takes an explicit opt-in."""
    write_manifest(models_dir, sha256=None)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)
    stub_fetch(service)

    with pytest.raises(ModelDownloadError) as caught:
        service.download(ModelService(settings).get("demo"))

    assert "no verified checksum" in caught.value.message.lower()
    assert not (models_dir / "demo.pth").exists()


def test_trusting_the_first_download_records_what_arrived(models_dir: Path) -> None:
    write_manifest(models_dir, sha256=None)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)
    stub_fetch(service)

    result = service.download(ModelService(settings).get("demo"), trust_first_download=True)

    assert result.sha256 == DIGEST
    assert (models_dir / "demo.pth").read_bytes() == PAYLOAD


def test_a_matching_digest_is_accepted(models_dir: Path) -> None:
    write_manifest(models_dir, sha256=DIGEST)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)
    stub_fetch(service)

    result = service.download(ModelService(settings).get("demo"))

    assert result.already_present is False
    assert result.size_bytes == len(PAYLOAD)


def test_a_mismatched_digest_is_discarded_rather_than_installed(models_dir: Path) -> None:
    """A file that fails its checksum must not end up under the real name: it
    would then be reported as downloaded and fail at load time instead."""
    write_manifest(models_dir, sha256=DIGEST)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)
    stub_fetch(service, b"something else entirely")

    with pytest.raises(ModelDownloadError, match="did not match their checksum"):
        service.download(ModelService(settings).get("demo"))

    assert not (models_dir / "demo.pth").exists()


def test_a_failed_download_leaves_no_partial_file(models_dir: Path) -> None:
    write_manifest(models_dir, sha256=DIGEST)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)
    stub_fetch(service, b"wrong")

    with pytest.raises(ModelDownloadError):
        service.download(ModelService(settings).get("demo"))

    assert list(models_dir.glob("*.part")) == []


def test_a_non_https_url_is_refused(models_dir: Path) -> None:
    """Weights are executable-adjacent; fetching them in the clear is not on."""
    write_manifest(models_dir, url="http://example.invalid/demo.pth", sha256=DIGEST)
    settings = make_settings(models_dir)
    service = ModelDownloadService(settings)

    with pytest.raises(ModelDownloadError, match="not a secure URL"):
        service.download(ModelService(settings).get("demo"))


def test_an_already_present_model_is_not_downloaded_again(models_dir: Path) -> None:
    write_manifest(models_dir, sha256=DIGEST)
    (models_dir / "demo.pth").write_bytes(PAYLOAD)
    settings = make_settings(models_dir)

    result = ModelDownloadService(settings).ensure_available("demo")

    assert result.already_present is True
    assert result.path == models_dir / "demo.pth"


def test_a_denoise_capable_model_pulls_its_pair(models_dir: Path) -> None:
    """Half a DNI pair cannot produce a blend, so asking for one means both."""
    write_manifest(models_dir, supports_denoise=True, denoise_pair="demo-wdn")
    settings = make_settings(models_dir)

    required = [
        status.entry.id for status in ModelDownloadService(settings).required_models("demo")
    ]

    assert required == ["demo", "demo-wdn"]


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(PAYLOAD * 5000)  # larger than one read chunk

    assert sha256_of(path) == hashlib.sha256(PAYLOAD * 5000).hexdigest()
