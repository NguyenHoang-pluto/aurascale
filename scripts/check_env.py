#!/usr/bin/env python
"""Report the local execution environment.

Run this first when GPU acceleration is not behaving as expected:

    backend/.venv/Scripts/python scripts/check_env.py     # Windows
    backend/.venv/bin/python scripts/check_env.py         # Linux / macOS

It deliberately imports torch lazily so the script still reports something
useful on a machine where torch is not installed yet.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

MIN_DRIVER_FOR_CU12 = 527.41  # Windows; Linux needs >= 525.60


def _line(label: str, value: object) -> None:
    print(f"  {label:<22} {value}")


def report_system() -> None:
    print("System")
    _line("Platform", f"{platform.system()} {platform.release()}")
    _line("Python", sys.version.split()[0])
    _line("Executable", sys.executable)


def report_driver() -> float | None:
    print("\nNVIDIA driver")
    if shutil.which("nvidia-smi") is None:
        _line("nvidia-smi", "not found — no NVIDIA driver installed")
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        _line("nvidia-smi", f"failed: {exc}")
        return None

    driver_version: float | None = None
    for row in output.splitlines():
        name, memory, driver = (part.strip() for part in row.split(","))
        _line("GPU", f"{name} ({memory})")
        _line("Driver", driver)
        try:
            driver_version = float(".".join(driver.split(".")[:2]))
        except ValueError:
            driver_version = None
    return driver_version


def report_torch(driver_version: float | None) -> int:
    print("\nPyTorch")
    try:
        import torch
    except ImportError:
        _line("torch", "not installed")
        print("\n  Install one of:")
        print("    pip install -r backend/requirements-cuda.txt   (NVIDIA GPU)")
        print("    pip install -r backend/requirements-cpu.txt    (CPU only)")
        return 1

    _line("torch", torch.__version__)
    _line("Built against CUDA", torch.version.cuda or "none (CPU-only build)")
    _line("CUDA available", torch.cuda.is_available())

    if torch.cuda.is_available():
        _line("Device", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        _line("VRAM free / total", f"{free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
        if total < 6 * 1024**3:
            print("\n  Note: under 6 GiB of VRAM. Tiled inference is required;")
            print("        TILE_SIZE=256 is the recommended setting.")
        return 0

    print("\n  CUDA is NOT available. Likely causes:")
    if torch.version.cuda is None:
        print("    * This is a CPU-only torch build.")
        print("      Reinstall with: pip install -r backend/requirements-cuda.txt")
    elif driver_version is not None and driver_version < MIN_DRIVER_FOR_CU12:
        built = torch.version.cuda
        if built.startswith("12"):
            print(f"    * torch is built against CUDA {built}, which needs driver")
            print(f"      >= {MIN_DRIVER_FOR_CU12}, but this machine has {driver_version}.")
            print("      Either update the NVIDIA driver, or install the cu118 build:")
            print("        pip install -r backend/requirements-cuda.txt")
    else:
        print("    * No NVIDIA GPU detected, or the driver is not loaded.")
    print("\n  The application still runs in CPU mode, roughly 20-60x slower.")
    return 0


def main() -> int:
    report_system()
    driver = report_driver()
    return report_torch(driver)


if __name__ == "__main__":
    raise SystemExit(main())
