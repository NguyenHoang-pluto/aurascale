"""Job request and response schemas.

Mirrors the contract in docs/api.md. Storage paths are deliberately absent from
every model here: they live in the database and never in a response body
(architecture § 10).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.db import Job
from app.models.enums import JobStage, JobStatus, OutputFormat
from app.repositories.base import Page
from app.schemas.common import CamelModel


class EnhanceSettings(CamelModel):
    """The optional `settings` object of a submission."""

    sharpen_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Unsharp-mask post-process. Not an AI feature, and not labelled as one.",
    )
    denoise_strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "DNI interpolation coefficient, for models with supportsDenoise. "
            "1.0 denoises most; 0.0 preserves noise."
        ),
    )
    tile_size: int | None = Field(
        default=None,
        ge=0,
        le=2048,
        description="Override TILE_SIZE. Null decides automatically from free VRAM.",
    )
    tile_pad: int | None = Field(default=None, ge=0, le=128, description="Override TILE_PAD.")


class JobCreatedResponse(CamelModel):
    """202 Accepted: the job exists and is waiting for a worker."""

    job_id: str
    status: JobStatus
    queue_position: int = Field(description="0 means it is next, or already started")
    created_at: datetime


class ImageFacts(CamelModel):
    """Dimensions and weight of an image, input or output."""

    width: int
    height: int
    size_bytes: int
    format: str


class JobError(CamelModel):
    """Why a job failed, in the same shape as a problem document."""

    code: str
    detail: str
    technical: str | None = None


class JobResponse(CamelModel):
    """The full job record."""

    job_id: str
    status: JobStatus
    stage: JobStage | None = Field(default=None, description="Null outside processing")
    progress: int = Field(ge=0, le=100)
    model: str
    scale: int
    device: str | None = Field(default=None, description="Where it ran; null until it starts")
    input: ImageFacts
    output: ImageFacts | None = None
    processing_ms: int | None = Field(
        default=None, description="Measured wall time, excluding queue wait"
    )
    error: JobError | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_job(cls, job: Job) -> JobResponse:
        """Build the response from a row, leaving every path behind."""
        status = JobStatus(job.status)

        output: ImageFacts | None = None
        if status is JobStatus.COMPLETED and job.output_width and job.output_height:
            output = ImageFacts(
                width=job.output_width,
                height=job.output_height,
                size_bytes=job.output_bytes or 0,
                format=OutputFormat(job.output_format).value.upper(),
            )

        error: JobError | None = None
        if job.error_code is not None:
            error = JobError(
                code=job.error_code,
                detail=job.error_message or "The job failed.",
                technical=job.error_technical,
            )

        return cls(
            job_id=job.id,
            status=status,
            stage=JobStage(job.stage) if job.stage else None,
            progress=job.progress,
            model=job.model_name,
            scale=job.scale,
            device=job.device,
            input=ImageFacts(
                width=job.input_width,
                height=job.input_height,
                size_bytes=job.input_bytes,
                format=job.input_format,
            ),
            output=output,
            processing_ms=job.processing_ms,
            error=error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class JobPage(CamelModel):
    """One page of history.

    `total` is the size of the whole filtered set, not of this page, so a
    client can show "20 of 42" and size its pagination without a second call.
    """

    items: list[JobResponse]
    total: int = Field(description="Matching jobs in total, ignoring limit and offset")
    limit: int
    offset: int

    @classmethod
    def from_page(cls, page: Page) -> JobPage:
        return cls(
            items=[JobResponse.from_job(job) for job in page.items],
            total=page.total,
            limit=page.limit,
            offset=page.offset,
        )
