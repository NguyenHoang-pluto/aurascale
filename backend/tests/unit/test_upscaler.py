"""The tiling inference loop, exercised with real torch modules.

The networks here are tiny and deterministic rather than pretrained, which is
what makes the assertions exact: a model whose receptive field fits inside the
tile padding must produce the *same* image whether it ran in one piece or in
twelve. That equality is the seam test - if the crop offsets were wrong by a
pixel, or the padding were not cropped at `pad * scale`, the two would differ.

The real weights are exercised separately in tests/integration/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from app.core.exceptions import InferenceError, InsufficientMemoryError
from app.inference.device import ExecutionTarget, cpu_target
from app.inference.upscaler import JobCancelledError, RealEsrganUpscaler
from app.models.enums import DeviceType


class NearestUpscale(nn.Module):
    """Deterministic 1-pixel-receptive-field upscaler.

    Nearest-neighbour duplication is translation-equivariant, so tiling can be
    checked against the untiled result for exact equality.
    """

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class ConvUpscale(nn.Module):
    """A 3x3 convolution then a pixel shuffle: a real, if small, network.

    The convolution gives it a receptive field wider than one pixel, so it also
    checks that the padding supplies genuine neighbouring context.
    """

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 3 * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x)).clamp(0.0, 1.0)


class ExplodingModule(nn.Module):
    """Raises whatever it is given, to drive the failure paths."""

    def __init__(self, error: BaseException, *, fail_times: int | None = None) -> None:
        super().__init__()
        self.error = error
        self.fail_times = fail_times
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.fail_times is None or self.calls <= self.fail_times:
            raise self.error
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def make_upscaler(module: nn.Module, scale: int = 2) -> RealEsrganUpscaler:
    return RealEsrganUpscaler(
        id="test-model",
        scale=scale,
        arch="test",
        module=module,
        target=cpu_target("test"),
    )


def make_image(width: int, height: int, seed: int = 1) -> np.ndarray[Any, Any]:
    """A noisy image: flat colours would hide seam errors."""
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# ------------------------------------------------------------------- geometry


def test_the_result_has_the_models_scale() -> None:
    upscaler = make_upscaler(NearestUpscale(2), scale=2)

    result = upscaler.upscale(make_image(40, 30), tile=16, tile_pad=8)

    assert result.shape == (60, 80, 3)
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    ("width", "height", "tile"),
    [(64, 64, 16), (70, 50, 32), (129, 65, 32), (100, 100, 40)],
)
def test_tiling_produces_the_same_image_as_a_single_pass(
    width: int, height: int, tile: int
) -> None:
    """The seam test. Any crop or offset error shows up as a pixel difference."""
    image = make_image(width, height)
    module = NearestUpscale(2)

    tiled = make_upscaler(module).upscale(image, tile=tile, tile_pad=8)
    whole = make_upscaler(module).upscale(image, tile=0, tile_pad=0)

    assert np.array_equal(tiled, whole)


def test_tiling_matches_a_single_pass_for_a_model_with_a_wider_receptive_field() -> None:
    """A convolution reads its neighbours, which is what the padding is for.

    With no padding the tile edges would see zeros instead of real pixels and
    the two results would diverge along every boundary.
    """
    image = make_image(96, 96, seed=7)
    module = ConvUpscale(2)

    tiled = make_upscaler(module).upscale(image, tile=32, tile_pad=8)
    whole = make_upscaler(module).upscale(image, tile=0, tile_pad=0)

    # uint8 rounding at the boundary of two float paths can differ by one step.
    assert np.abs(tiled.astype(int) - whole.astype(int)).max() <= 1


def test_padding_is_what_prevents_the_seams() -> None:
    """Without padding a convolutional model does show tile boundaries.

    This is the control for the test above: it demonstrates the artefact the
    padding removes, so the previous assertion cannot pass vacuously.
    """
    image = make_image(96, 96, seed=7)
    module = ConvUpscale(2)

    unpadded = make_upscaler(module).upscale(image, tile=32, tile_pad=0)
    whole = make_upscaler(module).upscale(image, tile=0, tile_pad=0)

    assert not np.array_equal(unpadded, whole)


# ------------------------------------------------------------------- progress


def test_progress_is_reported_once_per_tile_and_ends_at_one() -> None:
    seen: list[float] = []
    upscaler = make_upscaler(NearestUpscale(2))

    upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8, on_progress=seen.append)

    assert len(seen) == 4  # a 2x2 grid, measured rather than estimated
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_progress_for_a_single_tile_is_a_single_report() -> None:
    seen: list[float] = []
    upscaler = make_upscaler(NearestUpscale(2))

    upscaler.upscale(make_image(32, 32), tile=0, tile_pad=0, on_progress=seen.append)

    assert seen == [pytest.approx(1.0)]


# --------------------------------------------------------------- cancellation


def test_cancellation_is_checked_between_tiles() -> None:
    calls = {"count": 0}

    def should_cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    upscaler = make_upscaler(NearestUpscale(2))

    with pytest.raises(JobCancelledError):
        upscaler.upscale(make_image(64, 64), tile=16, tile_pad=8, should_cancel=should_cancel)


def test_a_job_that_is_never_cancelled_runs_to_completion() -> None:
    upscaler = make_upscaler(NearestUpscale(2))

    result = upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8, should_cancel=lambda: False)

    assert result.shape == (128, 128, 3)


# ----------------------------------------------------------------- the OOM ladder


def test_out_of_memory_on_cpu_is_reported_as_insufficient_memory() -> None:
    """There is no smaller tile to fall back to once CPU RAM is the limit."""
    module = ExplodingModule(RuntimeError("[enforce fail] cannot allocate memory"))
    upscaler = make_upscaler(module)

    with pytest.raises(InsufficientMemoryError) as caught:
        upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8)

    assert caught.value.code.value == "out_of_memory"
    assert "not enough memory" in caught.value.message.lower()


def test_cpu_does_not_retry_because_there_is_nothing_to_retry_with() -> None:
    """The ladder only helps on CUDA: halving a tile does not create RAM."""
    module = ExplodingModule(RuntimeError("cannot allocate memory"))
    upscaler = make_upscaler(module)

    with pytest.raises(InsufficientMemoryError):
        upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8)

    assert module.calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_a_cuda_tile_that_does_not_fit_is_retried_at_half_the_size() -> None:
    """The real ladder: free the cache, halve the tile, carry on.

    The module fails once and succeeds afterwards, which is what a genuine
    per-tile OOM looks like, so the job finishes at a smaller tile instead of
    failing.
    """
    module = ExplodingModule(torch.cuda.OutOfMemoryError("CUDA out of memory"), fail_times=1)
    upscaler = RealEsrganUpscaler(
        id="test-model",
        scale=2,
        arch="test",
        module=module.to("cuda"),
        target=ExecutionTarget(DeviceType.CUDA, "test", fp16=False),
    )

    result = upscaler.upscale(make_image(256, 256), tile=128, tile_pad=8)

    assert result.shape == (512, 512, 3)
    report = upscaler.last_report
    assert report is not None
    assert report.tile_size == 64  # halved from 128
    assert report.tile_size_reduced is True
    assert report.fell_back_to_cpu is False
    assert report.device == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_cuda_falls_back_to_the_cpu_once_no_tile_size_fits() -> None:
    """A job that will not fit in VRAM at any tile size still finishes.

    The module refuses every CUDA tensor and accepts CPU ones, so the fallback
    is what actually produces the result rather than a lucky retry.
    """

    class CudaRefusing(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.is_cuda:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    upscaler = RealEsrganUpscaler(
        id="test-model",
        scale=2,
        arch="test",
        module=CudaRefusing().to("cuda"),
        target=ExecutionTarget(DeviceType.CUDA, "test", fp16=False),
    )

    result = upscaler.upscale(make_image(128, 128), tile=128, tile_pad=8)

    assert result.shape == (256, 256, 3)
    report = upscaler.last_report
    assert report is not None
    assert report.fell_back_to_cpu is True
    assert report.device == "cpu"
    assert report.fp16 is False  # CPU never uses half precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_the_tile_is_never_halved_below_the_floor() -> None:
    """Past a point the padding costs more than the tile carries, so the job
    moves to the CPU rather than grinding through thousands of tiny tiles."""
    module = ExplodingModule(torch.cuda.OutOfMemoryError("CUDA out of memory"), fail_times=None)
    upscaler = RealEsrganUpscaler(
        id="test-model",
        scale=2,
        arch="test",
        module=module.to("cuda"),
        target=ExecutionTarget(DeviceType.CUDA, "test", fp16=False),
    )

    with pytest.raises(InsufficientMemoryError):
        upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8)


def test_a_model_failure_that_is_not_memory_is_reported_as_an_inference_error() -> None:
    module = ExplodingModule(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    upscaler = make_upscaler(module)

    with pytest.raises(InferenceError) as caught:
        upscaler.upscale(make_image(32, 32), tile=0, tile_pad=0)

    assert caught.value.code.value == "inference_failed"


# ------------------------------------------------------------------ provenance


def test_the_report_records_what_actually_ran() -> None:
    upscaler = make_upscaler(NearestUpscale(2))

    upscaler.upscale(make_image(64, 64), tile=32, tile_pad=8)
    report = upscaler.last_report

    assert report is not None
    assert report.device == "cpu"
    assert report.tile_size == 32
    assert report.tiles == 4
    assert report.fp16 is False
    assert report.fell_back_to_cpu is False
    assert report.tile_size_reduced is False


# --------------------------------------------------------------------- input


def test_an_alpha_channel_never_reaches_the_model() -> None:
    """Four channels here means the pipeline failed to split the alpha off."""
    upscaler = make_upscaler(NearestUpscale(2))
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)

    with pytest.raises(InferenceError, match="could not be prepared"):
        upscaler.upscale(rgba, tile=0, tile_pad=0)


def test_a_float_image_is_rejected() -> None:
    upscaler = make_upscaler(NearestUpscale(2))
    floats = np.zeros((16, 16, 3), dtype=np.float32)

    with pytest.raises(InferenceError, match="could not be prepared"):
        upscaler.upscale(floats, tile=0, tile_pad=0)


def test_values_are_clamped_so_highlights_do_not_wrap() -> None:
    """Without the clamp, out-of-range values wrap and produce black speckle."""

    class Overshoot(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest") * 4.0

    result = make_upscaler(Overshoot()).upscale(
        np.full((8, 8, 3), 200, dtype=np.uint8), tile=0, tile_pad=0
    )

    assert result.max() == 255
    assert result.min() == 255
