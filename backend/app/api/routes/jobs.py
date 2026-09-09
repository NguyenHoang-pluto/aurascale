"""Job endpoints: submit, inspect, stream progress, download, cancel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.api.deps import JobServiceDep, ProgressBrokerDep
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.models.db import Job
from app.models.enums import EnhancementMode, JobStatus, OutputFormat, TargetResolution
from app.schemas.job import JobCreatedResponse, JobPage, JobResponse
from app.services.job_service import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Keeps idle streams alive through proxies that reap quiet connections.
HEARTBEAT_SECONDS = 15

# A result never changes, and job ids are unguessable, so a client may cache
# hard; "private" keeps intermediaries out of it.
IMMUTABLE_CACHE = {"Cache-Control": "private, max-age=86400"}

MEDIA_TYPE = {
    OutputFormat.PNG: "image/png",
    OutputFormat.JPEG: "image/jpeg",
    OutputFormat.WEBP: "image/webp",
}


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an image for enhancement",
    description=(
        "Accepts the image and returns immediately; inference happens on a worker "
        "thread. Follow `/jobs/{jobId}/events` for progress, or poll "
        "`/jobs/{jobId}`.\n\n"
        "The upload is validated by content, not by its filename or declared type, "
        "and is streamed to disk so the size limit is enforced against the bytes "
        "actually received."
    ),
)
async def create_job(
    service: JobServiceDep,
    image: Annotated[UploadFile, File(description="JPEG, PNG or WEBP")],
    model: Annotated[str | None, Form()] = None,
    scale: Annotated[int | None, Form()] = None,
    mode: Annotated[
        EnhancementMode | None,
        Form(description="standard or creative. Supplies defaults; explicit fields still win."),
    ] = None,
    target: Annotated[
        TargetResolution | None,
        Form(description="2k, 4k, 6k, 8k or 16k. A destination size, not an upscale factor."),
    ] = None,
    output_format: Annotated[str | None, Form(alias="format")] = None,
    quality: Annotated[int | None, Form()] = None,
    preserve_metadata: Annotated[bool, Form(alias="preserveMetadata")] = True,
    settings: Annotated[str | None, Form()] = None,
) -> JobCreatedResponse:
    submitted = await service.create(
        image.file,
        original_filename=image.filename,
        model=model,
        scale=scale,
        mode=mode,
        target=target,
        output_format=output_format,
        quality=quality,
        preserve_metadata=preserve_metadata,
        settings_json=settings,
    )

    return JobCreatedResponse(
        job_id=submitted.job.id,
        status=submitted.job.status,
        queue_position=submitted.queue_position,
        created_at=submitted.job.created_at,
    )


@router.get(
    "",
    response_model=JobPage,
    summary="Job history, newest first",
    description=(
        "A page of past jobs, most recent first, optionally filtered by status. "
        "`total` counts everything that matches the filter, not just this page. "
        "History is as durable as the files behind it: a job and its images are "
        "removed once their retention window passes, so this is a record of "
        "recent work rather than an archive."
    ),
)
async def list_jobs(
    service: JobServiceDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Page size")] = (
        DEFAULT_PAGE_SIZE
    ),
    offset: Annotated[int, Query(ge=0, description="How many to skip")] = 0,
    status_filter: Annotated[
        JobStatus | None, Query(alias="status", description="Only jobs in this state")
    ] = None,
) -> JobPage:
    page = await service.list_jobs(limit=limit, offset=offset, status=status_filter)
    return JobPage.from_page(page)


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Job record",
    description=(
        "The full record, and the polling fallback wherever Server-Sent Events "
        "are unavailable. Storage paths are never included."
    ),
)
async def get_job(job_id: str, service: JobServiceDep) -> JobResponse:
    return JobResponse.from_job(await service.get(job_id))


@router.get(
    "/{job_id}/events",
    summary="Progress stream (SSE)",
    description=(
        "Emits `progress`, `stage` and one terminal `completed`, `failed` or "
        "`cancelled` event, then closes. Progress during inference is "
        "`tilesDone / tilesTotal`, measured rather than estimated.\n\n"
        "A subscriber that connects late is sent the most recent event "
        "immediately, so a reconnecting client is never left with an empty bar."
    ),
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_events(
    job_id: str,
    request: Request,
    service: JobServiceDep,
    broker: ProgressBrokerDep,
) -> EventSourceResponse:
    # Resolve the job first so an unknown id is a problem document rather than
    # a stream that opens and immediately closes.
    job = await service.get(job_id)

    async def publisher() -> AsyncIterator[dict[str, str]]:
        # A job that finished before anyone subscribed still owes the client a
        # terminal event; without this the stream would hang until timeout.
        terminal = _terminal_event_for(job)
        if terminal is not None:
            yield terminal
            return

        try:
            async for event in broker.subscribe(job_id):
                if await request.is_disconnected():  # pragma: no cover - client hang-up
                    return
                yield {"event": event.name, "data": _json(event.data)}
        except asyncio.CancelledError:  # pragma: no cover - client hang-up
            raise

    return EventSourceResponse(publisher(), ping=HEARTBEAT_SECONDS)


@router.get(
    "/{job_id}/result",
    summary="Download the enhanced image",
    description=(
        "Returns the finished image with a descriptive filename. Range requests "
        "are supported so a large download can resume."
    ),
    responses={
        200: {"content": {"image/png": {}, "image/jpeg": {}, "image/webp": {}}},
        409: {"description": "The job has not completed"},
    },
)
async def download_result(job_id: str, service: JobServiceDep) -> FileResponse:
    job, path = await service.result_path(job_id)
    media_type = MEDIA_TYPE[OutputFormat(job.output_format)]

    return FileResponse(
        path,
        media_type=media_type,
        filename=service.download_name(job),
        headers=IMMUTABLE_CACHE,
    )


@router.get(
    "/{job_id}/preview",
    summary="A view-sized copy of the result, or a full-resolution crop",
    description=(
        "Without parameters: the result re-encoded as JPEG with its long edge "
        "capped, so a 200 MP output is never loaded into a browser whole. The "
        "capped copy is built once and cached; a result already inside the cap "
        "is not upscaled. "
        "With `x`, `y`, `w` and `h`: a full-resolution crop of that region "
        "instead, for inspecting detail above 100% zoom. The region is checked "
        "against the result and refused if it does not fit — never clamped, "
        "because a silently moved crop shows the wrong pixels."
    ),
    responses={
        200: {"content": {"image/jpeg": {}}},
        409: {"description": "The job has not completed"},
        422: {"description": "The requested region is invalid"},
    },
)
async def get_preview(
    job_id: str,
    service: JobServiceDep,
    x: Annotated[int | None, Query(description="Crop origin, in output pixels")] = None,
    y: Annotated[int | None, Query(description="Crop origin, in output pixels")] = None,
    w: Annotated[int | None, Query(description="Crop width, in output pixels")] = None,
    h: Annotated[int | None, Query(description="Crop height, in output pixels")] = None,
) -> Response:
    requested = (x, y, w, h)

    if any(value is not None for value in requested):
        if any(value is None for value in requested):
            raise ValidationError(
                "A crop needs all four of x, y, w and h.",
                technical=f"received x={x} y={y} w={w} h={h}",
            )

        crop = await service.crop_jpeg(job_id, x or 0, y or 0, w or 0, h or 0)
        return Response(content=crop, media_type="image/jpeg", headers=IMMUTABLE_CACHE)

    _, path = await service.preview_path(job_id)
    return FileResponse(path, media_type="image/jpeg", headers=IMMUTABLE_CACHE)


@router.get(
    "/{job_id}/thumbnail",
    summary="A small tile of the result, for the history grid",
    description=(
        "256 px on the long edge, WEBP. Built on first request and cached, so a "
        "grid of twenty costs one encode each rather than one per view. A result "
        "already smaller than that is served at its own size, never upscaled."
    ),
    responses={
        200: {"content": {"image/webp": {}}},
        409: {"description": "The job has not completed"},
    },
)
async def get_thumbnail(job_id: str, service: JobServiceDep) -> FileResponse:
    _, path = await service.thumbnail_path(job_id)
    return FileResponse(path, media_type="image/webp", headers=IMMUTABLE_CACHE)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a running job, or delete a finished one",
    description=(
        "Cancellation is cooperative: the flag is set and the tiling loop stops "
        "at the next tile boundary, so it takes effect within one tile. No "
        "partial result is ever kept. A job that has already finished is deleted "
        "along with its files."
    ),
)
async def cancel_or_delete_job(job_id: str, service: JobServiceDep) -> Response:
    outcome = await service.cancel_or_delete(job_id)
    logger.info("job delete requested", extra={"job_id": job_id, "outcome": outcome})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _terminal_event_for(job: Job) -> dict[str, str] | None:
    """The closing event for a job that had already finished, if it had.

    Without this, subscribing to a job that finished a minute ago would open a
    stream that never says anything.
    """
    status_value = JobStatus(job.status)
    if not status_value.is_terminal:
        return None

    if status_value is JobStatus.COMPLETED:
        return {
            "event": "completed",
            "data": _json({"jobId": job.id, "processingMs": job.processing_ms}),
        }
    if status_value is JobStatus.CANCELLED:
        return {"event": "cancelled", "data": _json({"jobId": job.id})}

    return {
        "event": "failed",
        "data": _json(
            {
                "code": job.error_code or "internal_error",
                "detail": job.error_message or "The job failed.",
            }
        ),
    }
