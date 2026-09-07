"""Enumerations shared by the ORM, schemas and services."""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    """Lifecycle of an enhancement job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True once the job can no longer change state on its own."""
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class JobStage(StrEnum):
    """Pipeline stage, meaningful only while a job is processing.

    These mirror the stages in docs/image-processing.md so progress reporting
    describes work that actually happens rather than invented milestones.
    """

    VALIDATING = "validating"
    LOADING_MODEL = "loading_model"
    PREPROCESSING = "preprocessing"
    RUNNING_INFERENCE = "running_inference"
    POSTPROCESSING = "postprocessing"
    ENCODING = "encoding"


class OutputFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"

    @property
    def is_lossless(self) -> bool:
        return self is OutputFormat.PNG

    @property
    def extension(self) -> str:
        return "jpg" if self is OutputFormat.JPEG else self.value


class DeviceType(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
