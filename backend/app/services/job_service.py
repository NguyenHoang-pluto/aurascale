"""Job lifecycle: create, read, cancel, delete.

Everything a request does to a job goes through here. The service knows about
jobs and files; it does not know what an HTTP request is, and it never runs
inference itself - that happens on the worker, driven by the queue.

Isolation is per job and rests on three things (this is a local, single-user
service; there are no accounts to isolate between):

  * every filename is a server-generated UUID, so nothing a client sends ever
    reaches the filesystem;
  * every path is checked to be inside the storage tree before a read, write or
    delete;
  * paths live in the database and never in a response body.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import Settings
from app.core.exceptions import (
    JobNotCompletedError,
    JobNotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.db import Job
from app.models.enums import JobStatus, OutputFormat
from app.repositories.base import JobRepository
from app.services.enhancement_service import EnhancementService
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.storage_service import StorageService, new_job_id
from app.workers.progress import ProgressBroker
from app.workers.queue import JobQueue

logger = get_logger(__name__)

# Extensions used for the stored upload, by sniffed format.
INPUT_EXTENSION = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

STRENGTH_RANGE = (0.0, 1.0)
TILE_SIZE_RANGE = (0, 2048)
TILE_PAD_RANGE = (0, 128)


@dataclass(frozen=True, slots=True)
class EnhanceOptions:
    """The `settings` object from a submission, validated.

    Stored as JSON on the job so adding an option later does not need a
    migration, and read back by the runner when the job starts.
    """

    sharpen_strength: float = 0.0
    denoise_strength: float | None = None
    tile_size: int | None = None
    tile_pad: int | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"sharpenStrength": self.sharpen_strength}
        if self.denoise_strength is not None:
            payload["denoiseStrength"] = self.denoise_strength
        if self.tile_size is not None:
            payload["tileSize"] = self.tile_size
        if self.tile_pad is not None:
            payload["tilePad"] = self.tile_pad
        return payload


@dataclass(frozen=True, slots=True)
class SubmittedJob:
    """What a caller gets back from a successful submission."""

    job: Job
    queue_position: int


class JobService:
    """Creates jobs, answers questions about them, and stops them."""

    def __init__(
        self,
        settings: Settings,
        repository: JobRepository,
        *,
        queue: JobQueue,
        broker: ProgressBroker,
        images: ImageService | None = None,
        storage: StorageService | None = None,
        models: ModelService | None = None,
        enhancement: EnhancementService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._queue = queue
        self._broker = broker
        self._images = images or ImageService(settings)
        self._storage = storage or StorageService(settings)
        self._models = models or ModelService(settings)
        self._enhancement = enhancement or EnhancementService(settings, models=self._models)

    # ------------------------------------------------------------------ create

    async def create(
        self,
        upload: BinaryIO,
        *,
        original_filename: str | None = None,
        model: str | None = None,
        scale: int | None = None,
        output_format: str | None = None,
        quality: int | None = None,
        preserve_metadata: bool = True,
        settings_json: str | None = None,
    ) -> SubmittedJob:
        """Validate a submission, persist it, and queue it for the worker.

        The upload is streamed to disk under a generated name before anything
        looks at it, so the size cap is enforced against bytes on the wire
        rather than against a number the client supplied.
        """
        job_id = new_job_id()
        options = parse_options(settings_json)

        resolved_model = model or self._models.default_model_id()
        resolved_scale = scale if scale is not None else self._settings.default_scale

        # Model and scale are checked before a byte is written: rejecting a
        # 30 MB upload after storing it would be rude and pointless.
        self._models.get(resolved_model)
        self._enhancement.plan(resolved_model, resolved_scale)
        self._validate_denoise(resolved_model, options)

        self._storage.ensure_ready()
        self._storage.assert_capacity()

        staged = self._settings.inputs_dir / f"{job_id}.upload"
        size_bytes = self._images.stream_to_file(upload, staged)

        try:
            source_format, width, height = self._images.inspect(staged)
            self._images.assert_output_fits(width, height, resolved_scale)

            resolved_format = self._resolve_output_format(output_format, source_format)
            resolved_quality = (
                None
                if resolved_format is OutputFormat.PNG
                else self._images.validate_quality(quality)
            )

            paths = self._storage.paths_for(
                job_id,
                input_ext=INPUT_EXTENSION[source_format],
                output_ext=resolved_format.extension,
            )
            staged.replace(paths.input)
        except Exception:
            # Nothing is kept from a submission that was refused.
            staged.unlink(missing_ok=True)
            raise

        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            model_name=resolved_model,
            scale=resolved_scale,
            output_format=resolved_format.value,
            quality=resolved_quality,
            preserve_metadata=preserve_metadata,
            enhance_options=options.to_json(),
            input_path=str(paths.input),
            output_path=str(paths.output),
            input_width=width,
            input_height=height,
            input_bytes=size_bytes,
            input_format=source_format,
            original_filename=_safe_original_name(original_filename),
            expires_at=self._storage.expiry_for(),
        )

        await self._repository.create(job)
        # Durable before it is announced. The worker looks the job up in its
        # own session the moment the id reaches the queue, so an uncommitted
        # row would be a job that is accepted and then never runs.
        await self._repository.commit()

        try:
            position = await self._queue.submit(job_id)
        except Exception:
            # The queue refused it, so the row and the upload go too rather
            # than leaving behind a job nothing will ever pick up.
            self._storage.delete(paths.input)
            await self._repository.delete(job_id)
            await self._repository.commit()
            raise

        logger.info(
            "job created",
            extra={
                "job_id": job_id,
                "model": resolved_model,
                "scale": resolved_scale,
                "input": f"{width}x{height}",
                "bytes": size_bytes,
                "position": position,
            },
        )
        return SubmittedJob(job=job, queue_position=position)

    def _resolve_output_format(self, requested: str | None, source_format: str) -> OutputFormat:
        """Requested format, or the input's own format when none was asked for."""
        if requested is None:
            try:
                return OutputFormat(source_format.lower())
            except ValueError:  # pragma: no cover - sniffed formats all map
                return OutputFormat.PNG

        try:
            return OutputFormat(requested.lower())
        except ValueError as exc:
            raise ValidationError(
                f"{requested!r} is not an output format this server writes.",
                technical=f"supported: {', '.join(f.value for f in OutputFormat)}",
                context={"supported": [f.value for f in OutputFormat]},
            ) from exc

    def _validate_denoise(self, model_id: str, options: EnhanceOptions) -> None:
        """Refuse a denoise setting for a model that has no denoise weights."""
        if options.denoise_strength is None:
            return

        entry = self._models.get(model_id).entry
        if not entry.supports_denoise or entry.denoise_pair is None:
            raise ValidationError(
                f"{entry.name} does not have a denoise control.",
                technical=f"{model_id} declares no denoise_pair",
                context={"model": model_id},
            )

    # -------------------------------------------------------------------- read

    async def get(self, job_id: str) -> Job:
        """One job, or a 404-shaped error.

        The id is opaque and unguessable, which is what keeps one job's result
        from being reachable through another's link.
        """
        job = await self._repository.get(job_id)
        if job is None:
            raise JobNotFoundError(
                "That job does not exist. It may have been removed after its retention window.",
                technical=f"unknown job id {job_id[:8]}…",
            )
        return job

    def queue_position(self, job_id: str) -> int | None:
        return self._queue.position_of(job_id)

    async def result_path(self, job_id: str) -> tuple[Job, Path]:
        """The finished file, with every reason it might not be available."""
        job = await self.get(job_id)
        status = JobStatus(job.status)

        if status is not JobStatus.COMPLETED:
            raise JobNotCompletedError(
                _not_completed_message(status),
                technical=f"status={status.value}",
                context={"status": status.value},
            )

        if not job.output_path:  # pragma: no cover - completed implies an output
            raise JobNotFoundError(
                "The result for that job is no longer available.",
                technical="completed job has no output path",
            )

        path = Path(job.output_path)

        # Belt and braces: the path came from our own database, but a read is
        # still refused if it does not resolve inside the storage tree.
        if not self._storage.is_within_storage(path) or not path.is_file():
            raise JobNotFoundError(
                "The result for that job is no longer available. Results are removed "
                f"after {self._settings.temp_retention_hours} hours.",
                technical="output file is missing or outside the storage tree",
            )

        return job, path

    def download_name(self, job: Job) -> str:
        """A descriptive filename for the download.

        Built from the job id and the result's dimensions, never from the
        uploaded filename: that is client-supplied text and has no business in
        a `Content-Disposition` header.
        """
        extension = OutputFormat(job.output_format).extension
        size = (
            f"-{job.output_width}x{job.output_height}"
            if job.output_width and job.output_height
            else ""
        )
        return f"pixelforge-{job.id[:8]}{size}.{extension}"

    # ------------------------------------------------------------ cancel/delete

    async def cancel_or_delete(self, job_id: str) -> str:
        """Cancel a job that is still running, or delete a finished one.

        Returns "cancelled" or "deleted" so the caller can log which happened.
        `DELETE` means the same thing to a user either way: make it stop, and
        make it go away.
        """
        job = await self.get(job_id)
        status = JobStatus(job.status)

        if status.is_terminal:
            await self._delete(job)
            return "deleted"

        # Cooperative: the flag is set here and the tiling loop notices between
        # tiles, so cancellation lands within one tile rather than instantly.
        self._queue.request_cancel(job_id)
        await self._repository.update(job_id, cancel_requested=True)
        logger.info("cancellation requested", extra={"job_id": job_id, "status": status.value})
        return "cancelled"

    async def _delete(self, job: Job) -> None:
        removed = self._storage.delete(job.input_path, job.output_path, job.thumbnail_path)
        await self._repository.delete(job.id)
        self._broker.forget(job.id)
        logger.info("job deleted", extra={"job_id": job.id, "files_removed": removed})


def parse_options(settings_json: str | None) -> EnhanceOptions:
    """Validate the `settings` JSON blob from a submission."""
    if not settings_json:
        return EnhanceOptions()

    try:
        parsed = json.loads(settings_json)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "The settings field is not valid JSON.",
            technical=f"{type(exc).__name__}: {exc}",
        ) from exc

    if not isinstance(parsed, dict):
        raise ValidationError(
            "The settings field must be a JSON object.",
            technical=f"got {type(parsed).__name__}",
        )

    return EnhanceOptions(
        sharpen_strength=_strength(parsed.get("sharpenStrength"), "sharpenStrength") or 0.0,
        denoise_strength=_strength(parsed.get("denoiseStrength"), "denoiseStrength"),
        tile_size=_bounded_int(parsed.get("tileSize"), "tileSize", TILE_SIZE_RANGE),
        tile_pad=_bounded_int(parsed.get("tilePad"), "tilePad", TILE_PAD_RANGE),
    )


def _strength(value: Any, field: str) -> float | None:
    if value is None:
        return None

    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValidationError(
            f"{field} must be a number between 0 and 1.",
            technical=f"{field}={value!r}",
        )

    low, high = STRENGTH_RANGE
    if not low <= float(value) <= high:
        raise ValidationError(
            f"{field} must be between {low} and {high}.",
            technical=f"{field}={value}",
        )
    return float(value)


def _bounded_int(value: Any, field: str, bounds: tuple[int, int]) -> int | None:
    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{field} must be a whole number.",
            technical=f"{field}={value!r}",
        )

    low, high = bounds
    if not low <= value <= high:
        raise ValidationError(
            f"{field} must be between {low} and {high}.",
            technical=f"{field}={value}",
        )
    return value


def _safe_original_name(name: str | None) -> str | None:
    """Keep the user's filename for display, stripped of anything path-like.

    It is only ever shown back to the user; it never reaches the filesystem,
    which is why the stored name is a generated id instead.
    """
    if not name:
        return None

    cleaned = Path(name).name.strip()
    return cleaned[:255] or None


def _not_completed_message(status: JobStatus) -> str:
    if status is JobStatus.FAILED:
        return "That job failed, so there is no result to download."
    if status is JobStatus.CANCELLED:
        return "That job was cancelled, so there is no result to download."
    return "That job has not finished yet."
