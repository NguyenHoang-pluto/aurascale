"""Schemas for the system report."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import CamelModel


class TorchStatus(CamelModel):
    available: bool = Field(description="Whether PyTorch could be imported")
    version: str | None = Field(default=None, description="e.g. 2.7.1+cu118")
    cuda_version: str | None = Field(
        default=None, description="CUDA version torch was built against"
    )
    cuda_available: bool = Field(description="Whether a usable CUDA device was found")
    import_error: str | None = Field(
        default=None, description="Why torch failed to import, when it did"
    )


class GpuStatus(CamelModel):
    name: str
    vram_total_mb: int
    vram_free_mb: int = Field(description="Sampled at request time")
    capability: str = Field(description="CUDA compute capability, e.g. 8.6")


class SystemResponse(CamelModel):
    device: str = Field(description="'cuda' or 'cpu' — where inference will run")
    device_reason: str = Field(description="Why that device was selected")
    torch: TorchStatus
    gpu: GpuStatus | None = Field(
        default=None, description="Null when CUDA is unavailable, which is a normal state"
    )
    cpu_name: str
    cpu_cores_physical: int | None = None
    cpu_cores_logical: int | None = None
    ram_total_mb: int
    ram_available_mb: int
    python_version: str
    platform: str
    fp16: bool = Field(description="Whether half precision will be used")
    tile_size: int
    tile_pad: int
