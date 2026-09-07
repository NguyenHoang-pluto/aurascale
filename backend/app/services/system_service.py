"""Real hardware and runtime capability reporting.

Everything here is measured from the running process — torch's own view of
CUDA, and psutil's view of the machine. Nothing is assumed or hard-coded, so
`/api/system` reflects the environment the models will actually run in.

torch is imported lazily. It costs seconds and hundreds of megabytes, and the
health endpoint must stay answerable on a machine where torch is broken.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil

from app.core.config import Settings
from app.core.exceptions import GpuUnavailableError
from app.core.logging import get_logger
from app.models.enums import DeviceType

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TorchInfo:
    available: bool
    version: str | None
    cuda_version: str | None
    cuda_available: bool
    device_count: int = 0
    import_error: str | None = None


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int
    capability: str


@dataclass(frozen=True, slots=True)
class SystemInfo:
    device: DeviceType
    device_reason: str
    torch: TorchInfo
    gpu: GpuInfo | None
    cpu_name: str
    cpu_cores_physical: int | None
    cpu_cores_logical: int | None
    ram_total_mb: int
    ram_available_mb: int
    python_version: str
    platform_name: str
    fp16: bool
    tile_size: int
    tile_pad: int


@lru_cache(maxsize=1)
def _torch_info() -> TorchInfo:
    """Import torch once and record what it reports.

    Cached because importing torch is expensive; CUDA availability does not
    change during a process's lifetime, so caching costs no accuracy.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is installed in CI
        logger.warning("torch unavailable", extra={"error": str(exc)})
        return TorchInfo(
            available=False,
            version=None,
            cuda_version=None,
            cuda_available=False,
            import_error=str(exc),
        )

    # torch.cuda.is_available() alone is not sufficient. With
    # CUDA_VISIBLE_DEVICES="" it still reports True while device_count() is 0,
    # so trusting it would send inference to a device that does not exist and
    # fail with "Invalid device id" only once a job started.
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    return TorchInfo(
        available=True,
        version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cuda_available=device_count > 0,
        device_count=device_count,
    )


def _gpu_info() -> GpuInfo | None:
    """Live GPU properties, or None when CUDA is not usable.

    Free VRAM is sampled at call time rather than cached: it is the number that
    determines whether a job needs a smaller tile size.
    """
    info = _torch_info()
    if not info.available or not info.cuda_available:
        return None

    import torch

    try:
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # pragma: no cover - depends on driver state
        logger.warning("could not read GPU properties", extra={"error": str(exc)})
        return None

    return GpuInfo(
        name=properties.name,
        vram_total_mb=int(total_bytes // (1024 * 1024)),
        vram_free_mb=int(free_bytes // (1024 * 1024)),
        capability=f"{properties.major}.{properties.minor}",
    )


def _cpu_name_windows() -> str | None:
    """Marketing name from the registry, e.g. "AMD Ryzen 5 5600H".

    `platform.processor()` on Windows returns a family/model identifier such as
    "AMD64 Family 25 Model 80", which tells a user nothing useful.
    """
    try:
        import winreg

        key_path = "\\".join(["HARDWARE", "DESCRIPTION", "System", "CentralProcessor", "0"])
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except Exception:  # pragma: no cover - registry layout is stable but not guaranteed
        return None

    return str(value).strip() or None


def _cpu_name_linux() -> str | None:
    """Model name from /proc/cpuinfo."""
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    _, _, value = line.partition(":")
                    return value.strip() or None
    except OSError:  # pragma: no cover - not present off Linux
        return None
    return None


def _cpu_name() -> str:
    """Best available processor description, preferring a human-readable name."""
    system = platform.system()

    if system == "Windows":
        name = _cpu_name_windows()
    elif system == "Linux":
        name = _cpu_name_linux()
    else:
        # platform.processor() is already the marketing name on macOS.
        name = platform.processor() or None

    return name or platform.processor() or platform.machine() or "Unknown"


def resolve_device(settings: Settings) -> tuple[DeviceType, str]:
    """Decide which device inference will run on, and why.

    An explicit `DEVICE=cuda` on a machine without usable CUDA is an error, not
    a silent downgrade: silently falling back is how a job ends up taking forty
    minutes with nobody knowing why (`docs/architecture.md` § 8).
    """
    info = _torch_info()

    if settings.device == "cpu":
        return DeviceType.CPU, "DEVICE is set to cpu"

    if settings.device == "cuda":
        if not info.available:
            raise GpuUnavailableError(
                "GPU mode was requested but PyTorch could not be loaded.",
                technical=info.import_error,
            )
        if not info.cuda_available:
            raise GpuUnavailableError(
                "GPU mode was requested but no usable CUDA device was found. "
                "Check the driver version and that a CUDA build of PyTorch is installed.",
                technical=(
                    f"torch={info.version} cuda_build={info.cuda_version} "
                    f"device_count={info.device_count}"
                ),
            )
        return DeviceType.CUDA, "DEVICE is set to cuda"

    # auto
    if info.available and info.cuda_available:
        return DeviceType.CUDA, "CUDA device detected"
    if not info.available:
        return DeviceType.CPU, "PyTorch is not available"
    return DeviceType.CPU, "no CUDA device available"


class SystemService:
    """Assembles the system report served by `GET /api/system`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def collect(self) -> SystemInfo:
        device, reason = resolve_device(self._settings)
        memory = psutil.virtual_memory()

        return SystemInfo(
            device=device,
            device_reason=reason,
            torch=_torch_info(),
            gpu=_gpu_info(),
            cpu_name=_cpu_name(),
            cpu_cores_physical=psutil.cpu_count(logical=False),
            cpu_cores_logical=psutil.cpu_count(logical=True),
            ram_total_mb=int(memory.total // (1024 * 1024)),
            ram_available_mb=int(memory.available // (1024 * 1024)),
            python_version=platform.python_version(),
            platform_name=f"{platform.system()} {platform.release()}",
            # fp16 is only used on CUDA; on CPU it is emulated and slower.
            fp16=self._settings.use_fp16 and device is DeviceType.CUDA,
            tile_size=self._settings.tile_size,
            tile_pad=self._settings.tile_pad,
        )

    def describe_for_log(self) -> dict[str, Any]:
        """Compact startup log line describing the execution environment."""
        info = self.collect()
        return {
            "device": info.device.value,
            "reason": info.device_reason,
            "torch": info.torch.version,
            "cuda": info.torch.cuda_version,
            "gpu": info.gpu.name if info.gpu else None,
            "vram_mb": info.gpu.vram_total_mb if info.gpu else None,
            "device_count": info.torch.device_count,
        }
