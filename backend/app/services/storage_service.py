"""Filesystem storage for job inputs, outputs and thumbnails.

Every filename is generated here from a job id; nothing derived from a client
string ever reaches the filesystem (§ 16). Paths are checked to be inside the
storage tree before any write or delete, so a bug elsewhere cannot turn into a
path traversal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import StorageFullError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Extensions we will ever write. A format outside this set is a programming
# error, not user input, so it fails loudly.
SAFE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp"})


@dataclass(frozen=True, slots=True)
class JobPaths:
    input: Path
    output: Path
    thumbnail: Path


def new_job_id() -> str:
    """Opaque, unguessable job identifier, also used as the filename stem."""
    return uuid.uuid4().hex


class StorageService:
    """Owns the on-disk layout under STORAGE_DIR."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ---------------------------------------------------------------- paths

    @property
    def root(self) -> Path:
        return self._settings.storage_dir

    def ensure_ready(self) -> None:
        self._settings.ensure_directories()

    def is_within_storage(self, path: Path) -> bool:
        """True when `path` resolves inside the storage tree.

        Uses resolved paths so symlinks and `..` segments cannot escape.
        """
        try:
            path.resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return False
        return True

    def paths_for(self, job_id: str, *, input_ext: str, output_ext: str) -> JobPaths:
        """Storage locations for a job.

        `job_id` must be a generated hex id. Rejecting anything else keeps a
        caller from ever routing user input into a filename.
        """
        if not job_id or not all(character in "0123456789abcdef" for character in job_id):
            raise ValueError(f"job_id must be lowercase hex, got {job_id!r}")

        input_ext = self._normalise_extension(input_ext)
        output_ext = self._normalise_extension(output_ext)

        return JobPaths(
            input=self._settings.inputs_dir / f"{job_id}.{input_ext}",
            output=self._settings.outputs_dir / f"{job_id}.{output_ext}",
            thumbnail=self._settings.thumbs_dir / f"{job_id}.webp",
        )

    @staticmethod
    def _normalise_extension(extension: str) -> str:
        cleaned = extension.lower().lstrip(".")
        if cleaned not in SAFE_EXTENSIONS:
            raise ValueError(f"unsupported extension {extension!r}")
        return cleaned

    # ------------------------------------------------------------ retention

    def expiry_for(self, created: datetime | None = None) -> datetime:
        """When a job's files become eligible for the sweeper."""
        base = created or datetime.now(UTC)
        return base + timedelta(hours=self._settings.temp_retention_hours)

    def total_bytes(self) -> int:
        """Bytes currently used by the storage tree."""
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def assert_capacity(self, incoming_bytes: int = 0) -> None:
        """Refuse work that would push storage past its cap."""
        limit = int(self._settings.max_total_storage_gb * 1024**3)
        used = self.total_bytes()

        if used + incoming_bytes > limit:
            raise StorageFullError(
                "The server has run out of space for new images. "
                "Older results are removed automatically; try again shortly.",
                technical=f"used={used} incoming={incoming_bytes} limit={limit}",
                context={"usedBytes": used, "limitBytes": limit},
            )

    # -------------------------------------------------------------- removal

    def delete(self, *paths: Path | str | None) -> int:
        """Delete files, ignoring those already gone. Returns how many were removed.

        Anything outside the storage tree is refused and logged rather than
        deleted — a defence against a bad path reaching this point.
        """
        removed = 0

        for candidate in paths:
            if candidate is None:
                continue

            path = Path(candidate)
            if not self.is_within_storage(path):
                logger.error(
                    "refused to delete a path outside storage",
                    extra={"file": path.name},
                )
                continue

            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning(
                    "could not delete file", extra={"file": path.name, "error": str(exc)}
                )

        return removed

    def sweep_orphans(self, known_stems: set[str]) -> int:
        """Delete stored files with no corresponding job record.

        These accumulate when a process dies between writing a file and
        committing its row.
        """
        removed = 0

        for directory in (
            self._settings.inputs_dir,
            self._settings.outputs_dir,
            self._settings.thumbs_dir,
        ):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.stem not in known_stems:
                    removed += self.delete(path)

        if removed:
            logger.info("removed orphaned files", extra={"count": removed})
        return removed
