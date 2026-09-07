"""Torch device and precision selection.

Wraps the Phase 5 device resolution (`services.system_service.resolve_device`)
rather than re-deciding: there must be exactly one place that answers "where
does inference run", or `/api/system` and the worker can disagree and nobody
finds out until a job behaves strangely.

Precision is decided here too. fp16 halves activation memory and is faster on
every CUDA card that supports it; on CPU it is emulated and slower, so CPU is
always fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.enums import DeviceType
from app.services.system_service import resolve_device

logger = get_logger(__name__)

# Compute capability 5.3 is where CUDA half precision stops being emulated.
# Below it fp16 runs, but slowly enough that fp32 is the better default.
MIN_FP16_CAPABILITY = (5, 3)


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """Where a model will run and in what precision."""

    device_type: DeviceType
    reason: str
    fp16: bool
    index: int = 0

    @property
    def is_cuda(self) -> bool:
        return self.device_type is DeviceType.CUDA

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.index}" if self.is_cuda else "cpu"

    def dtype(self) -> Any:
        import torch

        return torch.float16 if self.fp16 else torch.float32

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.device_type.value,
            "reason": self.reason,
            "fp16": self.fp16,
        }


def supports_fp16(index: int = 0) -> bool:
    """Whether this CUDA device runs half precision natively.

    Asked of the device rather than assumed: on pre-Maxwell hardware fp16 is
    emulated, and choosing it there would make jobs slower while claiming an
    optimisation.
    """
    import torch

    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception as exc:  # pragma: no cover - depends on driver state
        logger.warning("could not read compute capability", extra={"error": str(exc)})
        return False

    return (major, minor) >= MIN_FP16_CAPABILITY


def select_target(settings: Settings) -> ExecutionTarget:
    """The device and precision inference should use.

    Raises GpuUnavailableError when DEVICE=cuda cannot be honoured — the
    Phase 5 rule that an explicit request never degrades silently.
    """
    device_type, reason = resolve_device(settings)

    fp16 = False
    if device_type is DeviceType.CUDA and settings.use_fp16:
        fp16 = supports_fp16()
        if not fp16:
            logger.info("fp16 disabled: device lacks native half precision support")

    return ExecutionTarget(device_type=device_type, reason=reason, fp16=fp16)


def cpu_target(reason: str) -> ExecutionTarget:
    """A CPU target, used when a CUDA job falls back after exhausting VRAM."""
    return ExecutionTarget(device_type=DeviceType.CPU, reason=reason, fp16=False)


def free_vram_mb(index: int = 0) -> int | None:
    """Free VRAM in megabytes, or None when there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return None

    try:
        free_bytes, _ = torch.cuda.mem_get_info(index)
    except Exception as exc:  # pragma: no cover - depends on driver state
        logger.warning("could not sample free VRAM", extra={"error": str(exc)})
        return None

    return int(free_bytes // (1024 * 1024))


def release_cuda_memory() -> None:
    """Return cached blocks to the driver.

    The allocator holds freed blocks for reuse, which is normally what you
    want. Between jobs it is not: fragmented cached blocks are why the second
    job on a 4 GB card can fail at a tile size the first one managed.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch missing is reported elsewhere
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Free VRAM (MB) -> the largest tile edge that comfortably fits alongside a
# model and its activations. Conservative on purpose: overshooting costs an OOM
# retry and a wasted partial pass, while undershooting costs some padding
# overhead. The steps are sized for the 23-block RRDBNet, the heaviest
# architecture here, so they are safe for the lighter ones too.
TILE_BY_FREE_VRAM_MB: tuple[tuple[int, int], ...] = (
    (1024, 128),
    (2048, 192),
    (3072, 256),
    (6144, 384),
)
LARGEST_AUTOMATIC_TILE = 512


def recommended_tile_size(configured: int, free_mb: int | None) -> int:
    """Clamp the configured tile to what the free VRAM will take.

    Sampled per job rather than once at startup: another application taking the
    GPU between jobs is the common case on a laptop, and the number that
    matters is the one at the moment work starts.

    `configured == 0` means the operator disabled tiling deliberately; that is
    left alone, and the out-of-memory ladder remains the safety net.
    """
    if configured <= 0 or free_mb is None:
        return configured

    ceiling = LARGEST_AUTOMATIC_TILE
    for threshold, tile in TILE_BY_FREE_VRAM_MB:
        if free_mb <= threshold:
            ceiling = tile
            break

    return min(configured, ceiling)
