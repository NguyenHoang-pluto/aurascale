"""System capability endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import SystemServiceDep
from app.schemas.system import GpuStatus, SystemResponse, TorchStatus

router = APIRouter(tags=["system"])


@router.get(
    "/system",
    response_model=SystemResponse,
    summary="Hardware and runtime capabilities",
    description=(
        "Reports where inference will run and why. Every value is measured from "
        "the running process; free VRAM is sampled per request because it is the "
        "number that determines whether a job needs a smaller tile size.\n\n"
        "`gpu` being null is a normal, supported state, not an error — the "
        "application runs on CPU."
    ),
)
async def get_system(service: SystemServiceDep) -> SystemResponse:
    info = service.collect()

    return SystemResponse(
        device=info.device.value,
        device_reason=info.device_reason,
        torch=TorchStatus(
            available=info.torch.available,
            version=info.torch.version,
            cuda_version=info.torch.cuda_version,
            cuda_available=info.torch.cuda_available,
            import_error=info.torch.import_error,
        ),
        gpu=(
            GpuStatus(
                name=info.gpu.name,
                vram_total_mb=info.gpu.vram_total_mb,
                vram_free_mb=info.gpu.vram_free_mb,
                capability=info.gpu.capability,
            )
            if info.gpu is not None
            else None
        ),
        cpu_name=info.cpu_name,
        cpu_cores_physical=info.cpu_cores_physical,
        cpu_cores_logical=info.cpu_cores_logical,
        ram_total_mb=info.ram_total_mb,
        ram_available_mb=info.ram_available_mb,
        python_version=info.python_version,
        platform=info.platform_name,
        fp16=info.fp16,
        tile_size=info.tile_size,
        tile_pad=info.tile_pad,
    )
