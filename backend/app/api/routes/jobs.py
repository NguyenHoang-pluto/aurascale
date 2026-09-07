"""Job endpoints: submit, inspect, stream progress, download, cancel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.api.deps import JobServiceDep, ProgressBrokerDep
from app.core.logging import get_logger
from app.models.db import Job
from app.models.enums import JobStatus, OutputFormat
from app.schemas.job import JobCreatedResponse, JobResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Keeps idle streams alive through proxies that reap quiet connections.
HEARTBEAT_SECONDS = 15

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
        # Results are immutable and job ids are unguessable, so a client may
        # cache aggressively; "private" keeps intermediaries out of it.
        headers={"Cache-Control": "private, max-age=86400"},
    )


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
