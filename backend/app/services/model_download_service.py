"""Fetching model weights from the official release URLs.

Weights are not vendored: they are the Real-ESRGAN project's release artifacts,
downloaded on demand into the git-ignored models directory and verified against
the SHA-256 digest recorded in the manifest.

The download goes to a temporary file next to the destination and is moved into
place only after the digest matches. A half-written `.pth` that looks present
but fails to load is a far more confusing failure than a missing one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ModelDownloadError
from app.core.logging import get_logger
from app.services.model_service import ModelEntry, ModelService, ModelStatus

logger = get_logger(__name__)

CHUNK_BYTES = 1024 * 256
DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = "PixelForgeAI/0.1 (+https://github.com/xinntao/Real-ESRGAN)"

ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True, slots=True)
class DownloadResult:
    model_id: str
    path: Path
    size_bytes: int
    sha256: str
    already_present: bool


def sha256_of(path: Path) -> str:
    """Digest of a file, read in chunks so a 64 MB checkpoint is not buffered."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class ModelDownloadService:
    """Downloads and verifies the weights named in the manifest."""

    def __init__(self, settings: Settings, models: ModelService | None = None) -> None:
        self._settings = settings
        self._models = models or ModelService(settings)

    def ensure_available(
        self,
        model_id: str,
        *,
        trust_first_download: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> DownloadResult:
        """Return the local path to a model's weights, downloading if needed."""
        status = self._models.get(model_id)

        if status.downloaded:
            return DownloadResult(
                model_id=model_id,
                path=status.path,
                size_bytes=status.size_bytes or status.path.stat().st_size,
                sha256="",
                already_present=True,
            )

        return self.download(
            status, trust_first_download=trust_first_download, on_progress=on_progress
        )

    def required_models(self, model_id: str) -> list[ModelStatus]:
        """A model and anything it cannot run without.

        The denoise-capable model is useless on its own: DNI needs both halves
        of the pair, so asking for one is asking for both.
        """
        status = self._models.get(model_id)
        required = [status]

        if status.entry.denoise_pair is not None:
            required.append(self._models.get(status.entry.denoise_pair))

        return required

    def download(
        self,
        status: ModelStatus,
        *,
        trust_first_download: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> DownloadResult:
        """Download one checkpoint and verify it before it becomes visible."""
        entry = status.entry
        expected = entry.sha256

        if expected is None and not trust_first_download:
            raise ModelDownloadError(
                f"No verified checksum is recorded for {entry.name}, so it will not be "
                "downloaded automatically.",
                technical=(
                    f"manifest sha256 for {entry.id} is null; re-run with "
                    "--trust-first-download to record it"
                ),
                context={"model": entry.id},
            )

        status.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading weights", extra={"model": entry.id, "url": entry.url})

        temporary = self._fetch_to_temporary(entry, on_progress=on_progress)

        try:
            digest = sha256_of(temporary)
            if expected is not None and digest != expected:
                raise ModelDownloadError(
                    f"The downloaded weights for {entry.name} did not match their checksum, "
                    "so they were discarded.",
                    technical=f"expected {expected[:16]}…, got {digest[:16]}…",
                    context={"model": entry.id},
                )
            size = temporary.stat().st_size
            # Move into place only once verified: an unverified file under the
            # real name would be reported as "downloaded" by /api/models.
            shutil.move(str(temporary), str(status.path))
        finally:
            _remove_quietly(temporary)

        logger.info(
            "weights ready",
            extra={"model": entry.id, "bytes": size, "sha256": digest[:16]},
        )

        return DownloadResult(
            model_id=entry.id,
            path=status.path,
            size_bytes=size,
            sha256=digest,
            already_present=False,
        )

    def _fetch_to_temporary(
        self, entry: ModelEntry, *, on_progress: ProgressCallback | None
    ) -> Path:
        """Stream the URL to a temporary file in the models directory."""
        if not entry.url.startswith("https://"):
            raise ModelDownloadError(
                f"The download location for {entry.name} is not a secure URL.",
                technical=f"url={entry.url!r}",
                context={"model": entry.id},
            )

        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{entry.id}.", suffix=".part", dir=self._settings.models_dir
        )
        # Close the descriptor mkstemp handed back and reopen by path. Holding
        # it open across a failed request leaves Windows unable to delete the
        # partial file, turning a network error into a stuck .part forever.
        os.close(handle)
        temporary = Path(temporary_name)

        # Some CDNs treat the default urllib agent as a bot and stall rather
        # than refuse, which surfaces as an unexplained timeout.
        request = urllib.request.Request(entry.url, headers={"User-Agent": USER_AGENT})

        try:
            with (
                urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
                temporary.open("wb") as destination,
            ):
                declared = response.headers.get("Content-Length")
                total = int(declared) if declared is not None and declared.isdigit() else None
                received = 0

                while chunk := response.read(CHUNK_BYTES):
                    destination.write(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received, total)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _remove_quietly(temporary)
            raise ModelDownloadError(
                f"{entry.name} could not be downloaded. Check the network connection "
                "and try again.",
                technical=f"{type(exc).__name__}: {exc}",
                context={"model": entry.id},
            ) from exc

        return temporary


def _remove_quietly(path: Path) -> None:
    """Delete a temporary file without letting cleanup mask the real error.

    A failure to remove a partial download is worth a log line, never a
    replacement for the network error that caused it.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - platform file-locking quirk
        logger.warning("could not remove partial download", extra={"error": str(exc)})
