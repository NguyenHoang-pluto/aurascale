"""The tiling inference loop.

This is the only place a GPU is used. It takes an RGB uint8 array and returns a
larger one, reporting progress as tiles complete and checking for cancellation
between them.

Why this exists rather than `RealESRGANer.enhance()`: that call is opaque, so
progress could only ever be faked. Here `tiles_done / tiles_total` is a genuine
measurement of work completed, and cancellation lands within one tile.

Memory discipline (docs/image-processing.md § 8):
  * `torch.inference_mode()` wraps every forward pass, so no autograd graph is
    built and activations are freed as soon as they are consumed;
  * one tile's tensors are alive at a time, never the whole padded image;
  * OOM is recoverable - free the cache, halve the tile, retry, then CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.core.exceptions import InferenceError, InsufficientMemoryError
from app.core.logging import get_logger
from app.inference.device import (
    ExecutionTarget,
    cpu_target,
    free_vram_mb,
    recommended_tile_size,
    release_cuda_memory,
)
from app.inference.tiler import plan_tiles

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn

logger = get_logger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

# How many times a tile may be halved before CUDA is abandoned for this job.
MAX_TILE_RETRIES = 3
# Below this a tile carries more padding overhead than real pixels.
MIN_TILE_SIZE = 32


class JobCancelledError(Exception):
    """Raised inside the loop when the caller asked to stop.

    Not a PixelForgeError: cancellation is a normal outcome the job layer turns
    into a cancelled record, not a failure to report to the user.
    """


@dataclass(frozen=True, slots=True)
class UpscaleReport:
    """What actually happened, for the job record and the UI."""

    device: str
    tile_size: int
    tiles: int
    fp16: bool
    fell_back_to_cpu: bool = False
    tile_size_reduced: bool = False


class Upscaler(Protocol):
    """The contract every model implementation satisfies.

    Any future architecture - SwinIR, HAT, a diffusion upscaler - becomes a new
    implementation of this plus a manifest entry, with no other layer changing.
    """

    id: str
    scale: int
    arch: str

    def upscale(
        self,
        image: np.ndarray[Any, Any],
        *,
        tile: int,
        tile_pad: int,
        on_progress: ProgressCallback | None = ...,
        should_cancel: CancelCheck | None = ...,
    ) -> np.ndarray[Any, Any]: ...


class _OutOfMemoryError(Exception):
    """Internal signal that a tile did not fit. Never leaves this module."""


@dataclass
class RealEsrganUpscaler:
    """A loaded Real-ESRGAN network, ready to run.

    Holds the module and the device it lives on. The ModelManager owns the
    lifetime; this class owns the arithmetic.
    """

    id: str
    scale: int
    arch: str
    module: nn.Module
    target: ExecutionTarget
    _last_report: UpscaleReport | None = field(default=None, init=False, repr=False)

    @property
    def last_report(self) -> UpscaleReport | None:
        """Provenance of the most recent call, for the job record."""
        return self._last_report

    def upscale(
        self,
        image: np.ndarray[Any, Any],
        *,
        tile: int,
        tile_pad: int,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> np.ndarray[Any, Any]:
        """Upscale an RGB image by this model's native factor.

        `image` is uint8 HxWx3. Alpha is the caller's problem - it must never
        reach a three-channel network (docs/image-processing.md § 3).
        """
        _validate_input(image)

        current_tile = self._starting_tile(tile)
        target = self.target
        reduced = False
        fell_back = False
        attempt = 0

        while True:
            try:
                result = self._run(
                    image,
                    tile=current_tile,
                    tile_pad=tile_pad,
                    target=target,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
            except _OutOfMemoryError as exc:
                attempt += 1
                release_cuda_memory()

                if target.is_cuda and current_tile > MIN_TILE_SIZE and attempt <= MAX_TILE_RETRIES:
                    current_tile = max(MIN_TILE_SIZE, _halved(current_tile, image))
                    reduced = True
                    logger.warning(
                        "out of GPU memory; retrying with a smaller tile",
                        extra={"model": self.id, "tile": current_tile, "attempt": attempt},
                    )
                    continue

                if target.is_cuda:
                    # Every tile size has been tried. CPU is slow but it
                    # finishes, which is better than failing the job.
                    logger.warning(
                        "out of GPU memory; falling back to CPU",
                        extra={"model": self.id, "attempts": attempt},
                    )
                    target = cpu_target("fell back to CPU after exhausting GPU memory")
                    self._move_to(target)
                    fell_back = True
                    current_tile = tile
                    attempt = 0
                    continue

                raise InsufficientMemoryError(
                    "There is not enough memory to enhance an image this large. "
                    "Try a smaller image or a lower upscale factor.",
                    technical=str(exc)[:400],
                    context={"model": self.id, "tile": current_tile},
                ) from exc

            self._last_report = UpscaleReport(
                device=target.device_type.value,
                tile_size=current_tile,
                tiles=_tile_count(image, current_tile, tile_pad),
                fp16=target.fp16,
                fell_back_to_cpu=fell_back,
                tile_size_reduced=reduced or current_tile != tile,
            )
            return result

    # ------------------------------------------------------------- internals

    def _starting_tile(self, configured: int) -> int:
        """The tile to begin with, given the VRAM free right now.

        Starting inside the budget is cheaper than discovering it through an
        OOM: a failed tile still costs the forward pass that raised.
        """
        if not self.target.is_cuda:
            return configured

        chosen = recommended_tile_size(configured, free_vram_mb(self.target.index))
        if chosen != configured:
            logger.info(
                "reduced tile size to fit available VRAM",
                extra={"model": self.id, "configured": configured, "tile": chosen},
            )
        return chosen

    def _run(
        self,
        image: np.ndarray[Any, Any],
        *,
        tile: int,
        tile_pad: int,
        target: ExecutionTarget,
        on_progress: ProgressCallback | None,
        should_cancel: CancelCheck | None,
    ) -> np.ndarray[Any, Any]:
        import torch

        height, width = image.shape[:2]
        grid = plan_tiles(width, height, tile_size=tile, tile_pad=tile_pad)
        dtype = target.dtype()

        # float32 canvas: tiles are written into it in fp32 regardless of the
        # compute precision, so rounding to uint8 happens once, at the end.
        canvas = np.zeros((height * self.scale, width * self.scale, 3), dtype=np.float32)

        for planned in grid.tiles():
            if should_cancel is not None and should_cancel():
                raise JobCancelledError

            rows, columns = planned.input_slice
            patch = image[rows, columns]

            with torch.inference_mode():
                tensor = _to_tensor(patch, device=target.torch_device, dtype=dtype)
                try:
                    output = self.module(tensor)
                except torch.cuda.OutOfMemoryError as exc:
                    del tensor
                    raise _OutOfMemoryError(str(exc)) from exc
                except RuntimeError as exc:
                    if _is_out_of_memory(exc):
                        del tensor
                        raise _OutOfMemoryError(str(exc)) from exc
                    raise InferenceError(
                        "The model failed while processing this image.",
                        technical=f"{type(exc).__name__}: {str(exc)[:400]}",
                        context={"model": self.id, "tile": tile},
                    ) from exc

                # Clamp before leaving the device: out-of-range values wrap
                # when they become uint8 and show up as speckle.
                block = output.squeeze(0).float().clamp_(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

            del tensor, output

            crop_rows, crop_columns = planned.crop_slice(self.scale)
            out_rows, out_columns = planned.output_slice(self.scale)
            canvas[out_rows, out_columns] = block[crop_rows, crop_columns]

            del block

            if on_progress is not None:
                on_progress((planned.index + 1) / grid.count)

        return (canvas * 255.0).round().astype(np.uint8)

    def _move_to(self, target: ExecutionTarget) -> None:
        """Relocate the network, e.g. to CPU after exhausting VRAM."""
        self.module = self.module.to(device=target.torch_device, dtype=target.dtype())
        self.target = target
        release_cuda_memory()


def _halved(tile: int, image: np.ndarray[Any, Any]) -> int:
    """Next tile size to try.

    A tile of 0 means "no tiling", so the first reduction has to start from the
    image itself rather than from zero.
    """
    if tile <= 0:
        height, width = image.shape[:2]
        return int(max(width, height)) // 2
    return tile // 2


def _tile_count(image: np.ndarray[Any, Any], tile: int, tile_pad: int) -> int:
    height, width = image.shape[:2]
    return plan_tiles(width, height, tile_size=tile, tile_pad=tile_pad).count


def _validate_input(image: np.ndarray[Any, Any]) -> None:
    """Reject anything the network cannot accept, with a developer-facing note.

    These are programming errors rather than user input - the pipeline splits
    alpha and converts colour before this point - so the technical detail
    matters more than the message.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError(
            "The image could not be prepared for the model.",
            technical=f"expected HxWx3 RGB, got shape {image.shape}",
        )
    if image.dtype != np.uint8:
        raise InferenceError(
            "The image could not be prepared for the model.",
            technical=f"expected uint8, got {image.dtype}",
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise InferenceError(
            "The image could not be prepared for the model.",
            technical=f"empty image, shape {image.shape}",
        )


def _to_tensor(patch: np.ndarray[Any, Any], *, device: str, dtype: Any) -> Any:
    """uint8 HxWx3 RGB into float NCHW on the target device, scaled to [0, 1]."""
    import torch

    array = np.ascontiguousarray(patch.transpose(2, 0, 1))
    tensor = torch.from_numpy(array).to(device=device, dtype=dtype, non_blocking=False)
    return tensor.div(255.0).unsqueeze(0)


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether a RuntimeError is really an allocation failure.

    CPU OOM arrives as a plain RuntimeError, and older CUDA paths raise
    RuntimeError rather than the dedicated class, so the message is the only
    signal available.
    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("out of memory", "cuda error: out of memory", "cannot allocate")
    )
