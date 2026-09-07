"""System service tests.

These assert the shape and internal consistency of what the service reports
rather than hard-coding this machine's hardware, so they pass on a GPU box and
a CPU-only CI runner alike. The values themselves are read live from torch and
psutil — nothing here is mocked.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import GpuUnavailableError
from app.models.enums import DeviceType
from app.services.system_service import SystemService, _torch_info, resolve_device


def test_reports_a_real_device(settings: Settings) -> None:
    info = SystemService(settings).collect()

    assert info.device in {DeviceType.CUDA, DeviceType.CPU}
    assert info.device_reason


def test_reports_real_host_hardware(settings: Settings) -> None:
    info = SystemService(settings).collect()

    assert info.ram_total_mb > 0
    assert 0 <= info.ram_available_mb <= info.ram_total_mb
    assert info.cpu_name
    assert info.cpu_cores_logical is not None and info.cpu_cores_logical >= 1
    assert info.python_version.startswith("3.")
    assert info.platform_name


def test_cuda_availability_requires_an_actual_device(settings: Settings) -> None:
    """Regression: torch.cuda.is_available() returns True with
    CUDA_VISIBLE_DEVICES="" even though device_count() is 0."""
    info = SystemService(settings).collect()

    assert info.torch.cuda_available == (info.torch.device_count > 0)


def test_gpu_details_are_consistent_with_cuda_availability(settings: Settings) -> None:
    info = SystemService(settings).collect()

    if info.torch.cuda_available:
        assert info.gpu is not None
        assert info.gpu.name
        assert info.gpu.vram_total_mb > 0
        # Free VRAM cannot exceed the total, and some is always in use.
        assert 0 <= info.gpu.vram_free_mb <= info.gpu.vram_total_mb
        assert "." in info.gpu.capability
    else:
        # No GPU is a supported state, not an error.
        assert info.gpu is None
        assert info.device is DeviceType.CPU


def test_auto_selects_cuda_only_when_it_is_actually_available(settings: Settings) -> None:
    device, reason = resolve_device(settings)
    torch_info = _torch_info()

    if torch_info.cuda_available:
        assert device is DeviceType.CUDA
        assert reason == "CUDA device detected"
    else:
        assert device is DeviceType.CPU
        assert "CUDA" in reason or "PyTorch" in reason


def test_explicit_cpu_is_honoured_even_with_a_gpu_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    device, reason = resolve_device(Settings())

    assert device is DeviceType.CPU
    assert reason == "DEVICE is set to cpu"


def test_fp16_is_never_claimed_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half precision is emulated on CPU and slower, so it must not be reported
    as active there even when USE_FP16 is on."""
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("USE_FP16", "true")

    info = SystemService(Settings()).collect()

    assert info.device is DeviceType.CPU
    assert info.fp16 is False


def test_requesting_cuda_without_cuda_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent downgrade is how a job ends up taking forty minutes with nobody
    knowing why, so an impossible request is an error."""
    if _torch_info().cuda_available:
        pytest.skip("CUDA is available on this machine; the failure path cannot be exercised")

    monkeypatch.setenv("DEVICE", "cuda")

    with pytest.raises(GpuUnavailableError) as excinfo:
        resolve_device(Settings())

    assert excinfo.value.to_problem()["code"] == "gpu_unavailable"


def test_settings_are_reflected_in_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TILE_SIZE", "128")
    monkeypatch.setenv("TILE_PAD", "8")

    info = SystemService(Settings()).collect()

    assert info.tile_size == 128
    assert info.tile_pad == 8


def test_log_summary_contains_the_decisive_fields(settings: Settings) -> None:
    summary = SystemService(settings).describe_for_log()

    assert set(summary) == {
        "device",
        "reason",
        "torch",
        "cuda",
        "gpu",
        "vram_mb",
        "device_count",
    }
