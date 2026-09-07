"""Device resolution exercised with CUDA genuinely absent.

`CUDA_VISIBLE_DEVICES=""` makes a real torch report no CUDA devices, so these
run the actual code path a CPU-only machine takes — no patching of torch, and
no assumption about the host this suite happens to run on.

A subprocess is required because torch caches CUDA availability on first use,
so the variable has to be set before the interpreter starts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from app.core.config import BACKEND_ROOT

pytestmark = pytest.mark.integration


def _run_with_no_cuda(script: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        # Hides every GPU from CUDA, which torch reports as "not available".
        "CUDA_VISIBLE_DEVICES": "",
        "ENVIRONMENT": "test",
        **env_overrides,
    }

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=BACKEND_ROOT,
        env=env,
        check=False,
    )


def test_auto_falls_back_to_cpu_when_no_cuda_device_exists() -> None:
    result = _run_with_no_cuda(
        """
        from app.core.config import Settings
        from app.services.system_service import SystemService, resolve_device

        settings = Settings()
        device, reason = resolve_device(settings)
        info = SystemService(settings).collect()
        print(f"DEVICE={device.value}")
        print(f"REASON={reason}")
        print(f"GPU={info.gpu}")
        print(f"FP16={info.fp16}")
        print(f"RAM_OK={info.ram_total_mb > 0}")
        """,
        DEVICE="auto",
    )

    assert result.returncode == 0, result.stderr
    assert "DEVICE=cpu" in result.stdout
    assert "REASON=no CUDA device available" in result.stdout
    # No GPU is a normal state, reported as absence rather than an error.
    assert "GPU=None" in result.stdout
    # Half precision is emulated and slower on CPU, so it must be off.
    assert "FP16=False" in result.stdout
    # The rest of the report still works without a GPU.
    assert "RAM_OK=True" in result.stdout


def test_explicitly_requesting_cuda_without_cuda_raises() -> None:
    result = _run_with_no_cuda(
        """
        from app.core.config import Settings
        from app.core.exceptions import GpuUnavailableError
        from app.services.system_service import resolve_device

        try:
            resolve_device(Settings())
        except GpuUnavailableError as exc:
            problem = exc.to_problem()
            print(f"CODE={problem['code']}")
            print(f"STATUS={problem['status']}")
            print(f"DETAIL={problem['detail']}")
        else:
            print("NO_ERROR_RAISED")
        """,
        DEVICE="cuda",
    )

    assert result.returncode == 0, result.stderr
    assert "CODE=gpu_unavailable" in result.stdout
    assert "STATUS=503" in result.stdout
    # The message must tell the user what to check, not just that it failed.
    assert "driver version" in result.stdout


def test_system_endpoint_serves_a_cpu_only_machine() -> None:
    """The API must answer normally with no GPU, since that is a supported
    deployment rather than a degraded one."""
    result = _run_with_no_cuda(
        """
        import asyncio, httpx
        from httpx import ASGITransport
        from app.main import create_app

        async def main():
            transport = ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                response = await c.get("/api/system")
            print(f"STATUS={response.status_code}")
            body = response.json()
            print(f"DEVICE={body['device']}")
            print(f"GPU={body['gpu']}")
            print(f"CUDA_AVAILABLE={body['torch']['cudaAvailable']}")
            print(f"TORCH_AVAILABLE={body['torch']['available']}")

        asyncio.run(main())
        """,
        DEVICE="auto",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS=200" in result.stdout
    assert "DEVICE=cpu" in result.stdout
    assert "GPU=None" in result.stdout
    assert "CUDA_AVAILABLE=False" in result.stdout
    # torch itself is still importable; only the device is absent.
    assert "TORCH_AVAILABLE=True" in result.stdout
