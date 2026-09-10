"""Running one job, end to end.

This is where the HTTP world and the Phase 6 inference engine meet. It owns the
stage sequence, the progress arithmetic, and the guarantee that a job always
reaches a terminal state with anything half-written cleaned up.

Progress weights:

    validating + preprocessing   0 - 10 %
    loading the model           10 - 15 %
    inference                   15 - 90 %   <- real tile counts
    post-processing + encoding  90 - 100 %

Only the inference band is subdivided, because only inference has something
real to subdivide by. The other stages report their own boundaries and nothing
in between: a bar that advances on a timer is an invention, and this project
does not ship inventions.

Each database write gets its own short transaction. A single session held for
the length of a job would keep every progress update invisible until the job
ended, which would make `GET /api/jobs/{id}` useless as the polling fallback,
and would hold a SQLite write lock for minutes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ErrorCode, PixelForgeError
from app.core.logging import get_logger
from app.inference.device import free_vram_mb, recommended_tile_size, release_cuda_memory
from app.inference.tiler import plan_tiles
from app.inference.upscaler import JobCancelledError
from app.models.db import Job, utcnow
from app.models.enums import JobStage, JobStatus, OutputFormat
from app.repositories.base import JobRepository
from app.services.enhancement_service import (
    EnhancementRequest,
    EnhancementResult,
    EnhancementService,
)
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.workers.progress import ProgressBroker, ProgressEvent, progress_event, stage_event
from app.workers.queue import JobQueue

logger = get_logger(__name__)

RepositoryScope = Callable[[], AbstractAsyncContextManager[JobRepository]]

# Stage boundaries, as percentages of the whole job.
PREPARE_START = 0
PREPARE_END = 10
LOAD_END = 15
INFERENCE_END = 90
ENCODE_START = 95
COMPLETE = 100

# Writing to SQLite on every tile would put a transaction between each forward
# pass. The event stream carries every measurement; the row is a snapshot for
# polling clients, so it is updated on stage changes and at this granularity.
DB_PROGRESS_STEP = 5


@dataclass(frozen=True, slots=True)
class TilePlan:
    """Predicted tile counts, so progress can report real numbers.

    Computed with the engine's own public helpers on the same inputs, rather
    than guessed: `recommended_tile_size` then `plan_tiles`, exactly as the
    upscaler does. Free VRAM can move between prediction and execution, so the
    result is checked against what actually ran, and a disagreement is logged
    rather than shown to anyone.
    """

    per_pass: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.per_pass)

    def tiles_done(self, fraction: float) -> int:
        """How many tiles the reported fraction corresponds to.

        The engine reports `(pass_index + tile_fraction) / passes`, so both the
        pass and the tile within it can be recovered exactly.
        """
        passes = len(self.per_pass)
        if passes == 0:  # pragma: no cover - a plan always has at least one pass
            return 0

        scaled = max(0.0, min(fraction, 1.0)) * passes
        index = min(int(scaled), passes - 1)
        within = scaled - index

        done = sum(self.per_pass[:index])
        return min(done + round(within * self.per_pass[index]), self.total)


def predict_tiles(
    width: int,
    height: int,
    scales: list[int],
    *,
    settings: Settings,
    tile_size: int | None,
    tile_pad: int | None,
    free_mb: int | None,
) -> TilePlan:
    """Tile counts per pass, at the size the engine will actually use."""
    configured = tile_size if tile_size is not None else settings.tile_size
    pad = tile_pad if tile_pad is not None else settings.tile_pad
    effective = recommended_tile_size(configured, free_mb)

    counts: list[int] = []
    current_width, current_height = width, height

    for scale in scales:
        grid = plan_tiles(current_width, current_height, tile_size=effective, tile_pad=pad)
        counts.append(grid.count)
        current_width *= scale
        current_height *= scale

    return TilePlan(per_pass=tuple(counts))


class JobRunner:
    """Executes queued jobs against the real inference engine."""

    def __init__(
        self,
        settings: Settings,
        *,
        scope: RepositoryScope,
        queue: JobQueue,
        broker: ProgressBroker,
        enhancement: EnhancementService,
        images: ImageService,
        storage: StorageService,
        models: ModelService | None = None,
    ) -> None:
        self._settings = settings
        self._scope = scope
        self._queue = queue
        self._broker = broker
        self._enhancement = enhancement
        self._images = images
        self._storage = storage
        self._models = models or ModelService(settings)

    async def run(self, job_id: str) -> None:
        """Take a queued job to a terminal state.

        Every exit path writes a terminal status and publishes a terminal
        event. A job that ended any other way would leave the UI watching a bar
        that never moves.
        """
        job = await self._load(job_id)

        if job is None:
            # Deleted between submission and pickup. Anyone streaming this job
            # is owed a terminal event, or their stream hangs until it times out.
            logger.warning("queued job no longer exists", extra={"job_id": job_id})
            await self._broker.publish(
                job_id,
                ProgressEvent(
                    "failed",
                    {
                        "code": ErrorCode.JOB_NOT_FOUND.value,
                        "detail": "That job no longer exists.",
                    },
                ),
            )
            return

        if JobStatus(job.status).is_terminal:
            return

        # Cancelled while still waiting: no image has been decoded and no model
        # touched, so stop before any work begins.
        if self._queue.is_cancelled(job_id) or job.cancel_requested:
            await self._finish_cancelled(job_id, job.output_path)
            return

        await self._execute(job)

    # ---------------------------------------------------------------- stages

    async def _execute(self, job: Job) -> None:
        job_id = job.id
        output_path = Path(job.output_path) if job.output_path else None
        started = time.perf_counter()

        await self._update(
            job_id,
            status=JobStatus.PROCESSING,
            stage=JobStage.VALIDATING,
            progress=PREPARE_START,
            started_at=utcnow(),
        )
        await self._broker.publish(job_id, stage_event(JobStage.VALIDATING, PREPARE_START))

        try:
            # Validation is repeated here, not trusted from submission: the row
            # records what the client sent, and this is the last point before
            # the GPU is committed to it.
            decoded = await asyncio.to_thread(self._images.decode, Path(job.input_path))
            self._images.assert_output_fits(decoded.width, decoded.height, job.scale)

            await self._raise_if_cancelled(job_id)
            await self._set_stage(job_id, JobStage.PREPROCESSING, PREPARE_END)

            plan = self._enhancement.plan(job.model_name, job.scale)
            scales = [self._models.get(model_id).entry.scale for model_id in plan]
            request = _request_for(job)

            await self._raise_if_cancelled(job_id)
            await self._set_stage(job_id, JobStage.LOADING_MODEL, LOAD_END)

            tiles = predict_tiles(
                decoded.width,
                decoded.height,
                scales,
                settings=self._settings,
                tile_size=request.tile_size,
                tile_pad=request.tile_pad,
                free_mb=free_vram_mb(),
            )

            await self._set_stage(job_id, JobStage.RUNNING_INFERENCE, LOAD_END)
            result = await self._run_inference(job_id, decoded.pixels, request, tiles)

            await self._raise_if_cancelled(job_id)
            await self._set_stage(job_id, JobStage.POSTPROCESSING, INFERENCE_END)

            if output_path is None:  # pragma: no cover - always set at creation
                raise RuntimeError("job has no output path")

            await self._set_stage(job_id, JobStage.ENCODING, ENCODE_START)
            size_bytes = await asyncio.to_thread(
                self._images.encode,
                result.image,
                output_path,
                output_format=OutputFormat(job.output_format),
                quality=job.quality,
                source=decoded,
                preserve_metadata=job.preserve_metadata,
            )

            # The last checkpoint, and the one the others do not cover.
            #
            # Every earlier check happens before or during inference. Encoding
            # comes after all of them and is not instant - a 16x result is
            # hundreds of megapixels - so a `DELETE` arriving while this thread
            # was in `encode` used to be answered "cancelled" and then
            # contradicted: the job recorded `completed`, kept its output file,
            # and left the row carrying both `status=completed` and
            # `cancel_requested=True`.
            #
            # Raising here routes into `_finish_cancelled` below, which is what
            # removes the file that was just written. Encoding is not
            # interrupted - stopping mid-write would leave exactly the partial
            # file that path exists to avoid - so the work is finished and then
            # discarded, which costs one encode and keeps the outcome honest.
            await self._raise_if_cancelled(job_id)
            await self._finish_completed(job, result, size_bytes, started)

        except JobCancelledError:
            await self._finish_cancelled(job_id, job.output_path)

        except PixelForgeError as error:
            await self._finish_failed(job_id, error, output_path)

        # A job must reach a terminal state whatever went wrong, or the UI
        # watches a bar that never moves.
        except Exception as error:
            logger.exception("job failed unexpectedly", extra={"job_id": job_id})
            await self._finish_failed(job_id, _as_pixelforge_error(error), output_path)

        finally:
            # Return cached blocks to the driver between jobs. The model stays
            # resident by design; it is the allocator's fragmented free blocks
            # that would otherwise make the next job fail at a tile size this
            # one managed.
            release_cuda_memory()

    async def _run_inference(
        self,
        job_id: str,
        pixels: Any,
        request: EnhancementRequest,
        tiles: TilePlan,
    ) -> EnhancementResult:
        """Run the engine on the worker pool, forwarding measured progress."""
        loop = asyncio.get_running_loop()
        last_persisted = LOAD_END
        # Writes scheduled from the worker thread, kept so they can be drained
        # before the job moves on.
        writes: list[Future[None]] = []

        def on_progress(fraction: float) -> None:
            nonlocal last_persisted

            percent = LOAD_END + int(fraction * (INFERENCE_END - LOAD_END))
            self._broker.publish_threadsafe(
                job_id,
                progress_event(
                    percent,
                    JobStage.RUNNING_INFERENCE,
                    tiles_done=tiles.tiles_done(fraction),
                    tiles_total=tiles.total,
                ),
            )

            if percent - last_persisted >= DB_PROGRESS_STEP:
                last_persisted = percent
                future = asyncio.run_coroutine_threadsafe(
                    self._update(job_id, progress=percent), loop
                )
                # Nothing awaits this future individually, so without a callback
                # a failed write would disappear entirely and the polling
                # fallback would silently stop advancing.
                future.add_done_callback(_log_write_failure)
                writes.append(future)

        def should_cancel() -> bool:
            # A plain set behind a lock: this runs on the worker thread between
            # tiles, where awaiting a database read is not an option.
            return self._queue.is_cancelled(job_id)

        result = await loop.run_in_executor(
            _executor_of(self._queue),
            lambda: self._enhancement.enhance(
                pixels, request, on_progress=on_progress, should_cancel=should_cancel
            ),
        )

        # Drain them before going any further. A progress write that landed
        # after the completion write would leave a finished job reporting a
        # stale percentage - which is exactly what it looks like when a job
        # silently stops halfway.
        if writes:
            await asyncio.gather(
                *(asyncio.wrap_future(write) for write in writes), return_exceptions=True
            )

        predicted, actual = tiles.total, sum(report.tiles for report in result.reports)
        if predicted != actual:
            # Prediction and run disagreed, so free VRAM moved between them.
            # Worth knowing about; never worth showing to a user.
            logger.info(
                "tile prediction differed from the run",
                extra={"job_id": job_id, "predicted": predicted, "actual": actual},
            )

        await self._update(job_id, progress=INFERENCE_END)
        return result

    # ------------------------------------------------------------- outcomes

    async def _finish_completed(
        self, job: Job, result: EnhancementResult, size_bytes: int, started: float
    ) -> None:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        height, width = result.image.shape[:2]

        await self._update(
            job.id,
            status=JobStatus.COMPLETED,
            stage=None,
            progress=COMPLETE,
            output_width=width,
            output_height=height,
            output_bytes=size_bytes,
            device=result.device,
            tile_size=result.tile_size,
            passes=len(result.passes),
            processing_ms=elapsed_ms,
            finished_at=utcnow(),
            expires_at=self._storage.expiry_for(),
        )
        await self._broker.publish(
            job.id, ProgressEvent("completed", {"jobId": job.id, "processingMs": elapsed_ms})
        )

        logger.info(
            "job completed",
            extra={
                "job_id": job.id,
                "model": job.model_name,
                "scale": job.scale,
                "device": result.device,
                "tile": result.tile_size,
                "passes": len(result.passes),
                "output": f"{width}x{height}",
                "ms": elapsed_ms,
            },
        )

    async def _finish_cancelled(self, job_id: str, output_path: str | None) -> None:
        """Record a cancellation and remove anything half-written.

        No output is kept. A partial result presented as a result would be the
        worst possible outcome of a cancelled job.
        """
        removed = self._storage.delete(output_path)

        await self._update(
            job_id,
            status=JobStatus.CANCELLED,
            stage=None,
            progress=0,
            cancel_requested=True,
            output_path=None,
            finished_at=utcnow(),
            expires_at=self._storage.expiry_for(),
        )
        await self._broker.publish(job_id, ProgressEvent("cancelled", {"jobId": job_id}))
        logger.info("job cancelled", extra={"job_id": job_id, "files_removed": removed})

    async def _finish_failed(
        self, job_id: str, error: PixelForgeError, output_path: Path | None
    ) -> None:
        self._storage.delete(output_path)

        await self._update(
            job_id,
            status=JobStatus.FAILED,
            stage=None,
            error_code=error.code.value,
            error_message=error.message,
            error_technical=error.technical,
            output_path=None,
            finished_at=utcnow(),
            expires_at=self._storage.expiry_for(),
        )
        await self._broker.publish(
            job_id,
            ProgressEvent(
                "failed",
                {
                    "code": error.code.value,
                    "detail": error.message,
                    **({"technical": error.technical} if error.technical else {}),
                },
            ),
        )
        logger.warning(
            "job failed",
            extra={"job_id": job_id, "code": error.code.value, "detail": error.message},
        )

    # -------------------------------------------------------------- plumbing

    async def _load(self, job_id: str) -> Job | None:
        async with self._scope() as repository:
            return await repository.get(job_id)

    async def _update(self, job_id: str, **changes: Any) -> None:
        async with self._scope() as repository:
            await repository.update(job_id, **changes)

    async def _set_stage(self, job_id: str, stage: JobStage, percent: int) -> None:
        await self._update(job_id, stage=stage, progress=percent)
        await self._broker.publish(job_id, stage_event(stage, percent))

    async def _raise_if_cancelled(self, job_id: str) -> None:
        if self._queue.is_cancelled(job_id):
            raise JobCancelledError


def _request_for(job: Job) -> EnhancementRequest:
    """Translate the stored options bag into an engine request."""
    options = dict(job.enhance_options or {})

    return EnhancementRequest(
        model_id=job.model_name,
        scale=job.scale,
        denoise_strength=_optional_float(options.get("denoiseStrength")),
        sharpen_strength=float(options.get("sharpenStrength") or 0.0),
        tile_size=_optional_int(options.get("tileSize")),
        tile_pad=_optional_int(options.get("tilePad")),
        # Present only when the job asked for a target resolution. The planner
        # already checked the neural pass reaches it, so this is the exact
        # size the result is resampled down to.
        target_width=_optional_int(options.get("targetWidth")),
        target_height=_optional_int(options.get("targetHeight")),
    )


def _log_write_failure(future: Future[None]) -> None:
    """Surface an error from a progress write nobody is waiting on."""
    try:
        future.result()
    except Exception as error:  # pragma: no cover - only on a database fault
        logger.warning("progress write failed", extra={"error": f"{type(error).__name__}: {error}"})


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _executor_of(queue: JobQueue) -> Any:
    """The pool inference runs on.

    A queue implementation without its own pool - a fake, in tests - gets None,
    which asyncio reads as "the default executor".
    """
    return getattr(queue, "executor", None)


def _as_pixelforge_error(error: Exception) -> PixelForgeError:
    """Wrap an unexpected failure so the user still gets a usable message."""
    wrapped = PixelForgeError(
        "Something went wrong while enhancing this image.",
        technical=f"{type(error).__name__}: {error}",
    )
    wrapped.code = ErrorCode.INTERNAL_ERROR
    return wrapped
