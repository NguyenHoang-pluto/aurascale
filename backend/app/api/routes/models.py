"""Model registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CapabilitiesDep, ModelServiceDep
from app.schemas.model import ModelResponse

router = APIRouter(tags=["models"])

MB = 1024 * 1024


@router.get(
    "/models",
    response_model=list[ModelResponse],
    summary="Available super-resolution models",
    description=(
        "Models registered in models/manifest.json, annotated with whether their "
        "weights are present on disk, and with the upscale factors each one can "
        "actually produce. Weight sets that are not selectable on their own — "
        "such as the DNI denoise counterpart — are omitted."
    ),
)
async def list_models(
    service: ModelServiceDep, capabilities: CapabilitiesDep
) -> list[ModelResponse]:
    return [
        ModelResponse(
            id=status.entry.id,
            name=status.entry.name,
            description=status.entry.description,
            arch=status.entry.arch,
            scale=status.entry.scale,
            supports_denoise=status.entry.supports_denoise,
            supported_scales=capabilities.supported_scales(status.entry.id),
            downloaded=status.downloaded,
            size_mb=(round(status.size_bytes / MB, 1) if status.size_bytes is not None else None),
        )
        for status in service.list(selectable_only=True)
    ]
