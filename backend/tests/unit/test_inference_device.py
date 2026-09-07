"""Device and precision selection for inference.

These run against the real torch on this machine. Where a test needs the
opposite hardware from whatever is present, it drives the decision through the
settings rather than pretending torch reports something it does not - the
CUDA-absent path is covered for real in tests/integration/.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import GpuUnavailableError
from app.inference.device import (
    ExecutionTarget,
    cpu_target,
    free_vram_mb,
    recommended_tile_size,
    select_target,
)
from app.models.enums import DeviceType


def _settings(**overrides: object) -> Settings:
    return Settings(environment="test", **overrides)  # type: ignore[arg-type]


def cuda_is_present() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def test_cpu_is_honoured_regardless_of_the_hardware() -> None:
    target = select_target(_settings(device="cpu"))

    assert target.device_type is DeviceType.CPU
    assert target.torch_device == "cpu"
    assert target.reason == "DEVICE is set to cpu"


def test_cpu_never_uses_half_precision() -> None:
    """fp16 on CPU is emulated, so it would be slower while claiming to be faster."""
    target = select_target(_settings(device="cpu", use_fp16=True))

    assert target.fp16 is False


def test_dtype_follows_the_precision_decision() -> None:
    import torch

    assert cpu_target("test").dtype() is torch.float32
    assert ExecutionTarget(DeviceType.CUDA, "test", fp16=True).dtype() is torch.float16
    assert ExecutionTarget(DeviceType.CUDA, "test", fp16=False).dtype() is torch.float32


def test_a_cuda_target_names_its_device_index() -> None:
    assert ExecutionTarget(DeviceType.CUDA, "test", fp16=True).torch_device == "cuda:0"
    assert ExecutionTarget(DeviceType.CUDA, "test", fp16=True, index=1).torch_device == "cuda:1"


def test_the_cpu_fallback_target_records_why_it_happened() -> None:
    target = cpu_target("fell back to CPU after exhausting GPU memory")

    assert target.device_type is DeviceType.CPU
    assert target.fp16 is False
    assert "exhausting GPU memory" in target.reason


def test_auto_resolves_to_whatever_this_machine_actually_has() -> None:
    target = select_target(_settings(device="auto"))

    if cuda_is_present():
        assert target.device_type is DeviceType.CUDA
        assert target.reason == "CUDA device detected"
    else:
        assert target.device_type is DeviceType.CPU
        assert target.reason in {"no CUDA device available", "PyTorch is not available"}


@pytest.mark.skipif(not cuda_is_present(), reason="no CUDA device on this machine")
def test_fp16_is_used_on_a_capable_cuda_device() -> None:
    target = select_target(_settings(device="cuda", use_fp16=True))

    assert target.device_type is DeviceType.CUDA
    # Every card new enough to run this project is well past capability 5.3.
    assert target.fp16 is True


@pytest.mark.skipif(not cuda_is_present(), reason="no CUDA device on this machine")
def test_fp16_can_be_switched_off_without_changing_the_device() -> None:
    target = select_target(_settings(device="cuda", use_fp16=False))

    assert target.device_type is DeviceType.CUDA
    assert target.fp16 is False


@pytest.mark.skipif(cuda_is_present(), reason="CUDA is available on this machine")
def test_requesting_cuda_without_a_gpu_is_an_error_not_a_downgrade() -> None:
    with pytest.raises(GpuUnavailableError):
        select_target(_settings(device="cuda"))


def test_free_vram_is_reported_only_when_there_is_a_gpu() -> None:
    free = free_vram_mb()

    if cuda_is_present():
        assert free is not None and free > 0
    else:
        assert free is None


# --------------------------------------------------------- tile size budgeting


@pytest.mark.parametrize(
    ("free_mb", "expected"),
    [
        (800, 128),  # a nearly full 4 GB card
        (1500, 192),
        (2500, 256),
        (5000, 384),
        (11000, 512),
    ],
)
def test_the_tile_is_capped_by_the_vram_that_is_actually_free(free_mb: int, expected: int) -> None:
    """Starting inside the budget avoids paying for a forward pass that OOMs."""
    assert recommended_tile_size(1024, free_mb) == expected


def test_a_configured_tile_smaller_than_the_budget_is_left_alone() -> None:
    """The setting is a ceiling, never a target to grow into."""
    assert recommended_tile_size(128, 11000) == 128


def test_disabling_tiling_is_respected() -> None:
    """TILE_SIZE=0 is a deliberate choice; the OOM ladder is the safety net."""
    assert recommended_tile_size(0, 500) == 0


def test_without_a_gpu_there_is_no_budget_to_apply() -> None:
    assert recommended_tile_size(256, None) == 256
