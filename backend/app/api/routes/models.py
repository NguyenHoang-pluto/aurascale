"""Model registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ModelServiceDep
from app.schemas.model import ModelResponse

router = APIRouter(tags=["models"])

MB = 1024 * 1024


@router.get(
    "/models",
    response_model=list[ModelResponse],
    summary="Available super-resolution models",
    description=(
        "Models registered in models/manifest.json, annotated with whether their "
        "weights are present on disk. Weight sets that are not selectable on "
        "their own — such as the DNI denoise counterpart — are omitted."
    ),
)
async def list_models(service: ModelServiceDep) -> list[ModelResponse]:
    return [
        ModelResponse(
            id=status.entry.id,
            name=status.entry.name,
            description=status.entry.description,
            arch=status.entry.arch,
            scale=status.entry.scale,
            supports_denoise=status.entry.supports_denoise,
            downloaded=status.downloaded,
            size_mb=(round(status.size_bytes / MB, 1) if status.size_bytes is not None else None),
        )
        for status in service.list(selectable_only=True)
    ]
