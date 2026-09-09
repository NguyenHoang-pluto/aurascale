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


class EnhancementMode(StrEnum):
    """What the user is asking the enhancement to prioritise.

    A mode is a named bundle of decisions - which network, how much denoising -
    so those decisions live in one planner rather than being spread across the
    API, the worker and the UI. It is not a quality tier and nothing is gated
    behind it.
    """

    #: Fidelity first. The pipeline that shipped, unchanged.
    STANDARD = "standard"
    #: Detail first. A different network and an explicit denoise setting.
    CREATIVE = "creative"


class TargetResolution(StrEnum):
    """A requested output size, named by its long edge.

    Deliberately *not* an alias for an upscale factor. "4K" is a destination -
    3840 pixels on the long edge whatever the input was - while "4x" is a
    multiplier. Conflating them is why a 500 px source and a 3000 px source
    would otherwise both be called 4K and land nowhere near it.

    16K makes that separation impossible to miss: it is 8x the 1920 base, so
    reaching it from a 1080p source is an **8x** pass, not a 16x one. The two
    names look alike and mean different things, which is exactly why they are
    planned by different rules.

    The values follow the 1920 family, so they are consistent with each other
    and with the displays people actually own. They are shown in the UI
    alongside the number, because "2K" means 2048 to a cinema and 2560 to a
    gamer and neither reading is wrong.
    """

    TWO_K = "2k"
    FOUR_K = "4k"
    SIX_K = "6k"
    EIGHT_K = "8k"
    SIXTEEN_K = "16k"

    @property
    def long_edge(self) -> int:
        return {
            TargetResolution.TWO_K: 1920,
            TargetResolution.FOUR_K: 3840,
            TargetResolution.SIX_K: 5760,
            TargetResolution.EIGHT_K: 7680,
            TargetResolution.SIXTEEN_K: 15360,
        }[self]

    @property
    def label(self) -> str:
        """How it is named to a user, long edge included to remove ambiguity."""
        return f"{self.value.upper()} ({self.long_edge} px)"


class OutputType(StrEnum):
    """How the output size was asked for.

    Recorded so a finished job can say which question it answered. An old job
    carries no target metadata at all, and reads as SCALE - which is what it
    was, since that was the only way to ask.
    """

    SCALE = "scale"
    TARGET = "target"
