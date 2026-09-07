"""Application error taxonomy.

Every failure the user can trigger maps to a `PixelForgeError` subclass with:

  * `code`      — a stable machine-readable enum the frontend maps to copy
  * `message`   — a human-readable sentence, safe to show verbatim
  * `technical` — optional detail for the collapsible "Technical details"
                  panel (§ 10). Never contains filesystem paths or user data.

Raw exceptions (CUDA OOM, PIL decode errors) are translated at the boundary so
the API never leaks a traceback as its primary message.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # -- input validation ---------------------------------------------
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPTED_IMAGE = "corrupted_image"
    FILE_TOO_LARGE = "file_too_large"
    IMAGE_TOO_LARGE = "image_too_large"
    IMAGE_TOO_SMALL = "image_too_small"
    OUTPUT_TOO_LARGE = "output_too_large"
    INVALID_PARAMETERS = "invalid_parameters"

    # -- resources -----------------------------------------------------
    OUT_OF_MEMORY = "out_of_memory"
    GPU_UNAVAILABLE = "gpu_unavailable"
    STORAGE_FULL = "storage_full"

    # -- model / inference ---------------------------------------------
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    INFERENCE_FAILED = "inference_failed"

    # -- jobs ----------------------------------------------------------
    JOB_NOT_FOUND = "job_not_found"
    JOB_NOT_COMPLETED = "job_not_completed"
    JOB_CANCELLED = "job_cancelled"
    QUEUE_FULL = "queue_full"
    TIMEOUT = "timeout"

    # -- catch-all ------------------------------------------------------
    INTERNAL_ERROR = "internal_error"


class PixelForgeError(Exception):
    """Base class for all errors surfaced through the API."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    title: str = "Something went wrong"

    def __init__(
        self,
        message: str,
        *,
        technical: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.technical = technical
        self.context = context or {}

    def to_problem(self) -> dict[str, Any]:
        """Serialise as RFC 9457 application/problem+json."""
        problem: dict[str, Any] = {
            "type": f"https://pixelforge.ai/errors/{self.code.value}",
            "title": self.title,
            "status": self.status_code,
            "code": self.code.value,
            "detail": self.message,
        }
        if self.technical:
            problem["technical"] = self.technical
        if self.context:
            problem["context"] = self.context
        return problem


# ---------------------------------------------------------------- 4xx


class ValidationError(PixelForgeError):
    code = ErrorCode.INVALID_PARAMETERS
    status_code = 422
    title = "Invalid request"


class UnsupportedFormatError(PixelForgeError):
    code = ErrorCode.UNSUPPORTED_FORMAT
    status_code = 415
    title = "Unsupported file type"


class CorruptedImageError(PixelForgeError):
    code = ErrorCode.CORRUPTED_IMAGE
    status_code = 422
    title = "Image could not be read"


class FileTooLargeError(PixelForgeError):
    code = ErrorCode.FILE_TOO_LARGE
    status_code = 413
    title = "File too large"


class ImageTooLargeError(PixelForgeError):
    code = ErrorCode.IMAGE_TOO_LARGE
    status_code = 413
    title = "Image too large"


class ImageTooSmallError(PixelForgeError):
    code = ErrorCode.IMAGE_TOO_SMALL
    status_code = 422
    title = "Image too small"


class OutputTooLargeError(PixelForgeError):
    code = ErrorCode.OUTPUT_TOO_LARGE
    status_code = 413
    title = "Result would be too large"


class JobNotFoundError(PixelForgeError):
    code = ErrorCode.JOB_NOT_FOUND
    status_code = 404
    title = "Job not found"


class JobNotCompletedError(PixelForgeError):
    code = ErrorCode.JOB_NOT_COMPLETED
    status_code = 409
    title = "Job is not finished"


class ModelNotFoundError(PixelForgeError):
    code = ErrorCode.MODEL_NOT_FOUND
    status_code = 404
    title = "Unknown model"


class QueueFullError(PixelForgeError):
    code = ErrorCode.QUEUE_FULL
    status_code = 503
    title = "Server busy"


# ---------------------------------------------------------------- 5xx


class InsufficientMemoryError(PixelForgeError):
    code = ErrorCode.OUT_OF_MEMORY
    status_code = 507
    title = "Not enough memory"


class GpuUnavailableError(PixelForgeError):
    code = ErrorCode.GPU_UNAVAILABLE
    status_code = 503
    title = "GPU unavailable"


class ModelLoadError(PixelForgeError):
    code = ErrorCode.MODEL_LOAD_FAILED
    status_code = 500
    title = "Model failed to load"


class ModelDownloadError(PixelForgeError):
    code = ErrorCode.MODEL_DOWNLOAD_FAILED
    status_code = 502
    title = "Model download failed"


class InferenceError(PixelForgeError):
    code = ErrorCode.INFERENCE_FAILED
    status_code = 500
    title = "Enhancement failed"


class StorageFullError(PixelForgeError):
    code = ErrorCode.STORAGE_FULL
    status_code = 507
    title = "Storage full"


class ProcessingTimeoutError(PixelForgeError):
    code = ErrorCode.TIMEOUT
    status_code = 504
    title = "Processing timed out"
