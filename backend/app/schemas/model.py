"""Schemas for the model registry."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import CamelModel


class ModelResponse(CamelModel):
    id: str
    name: str
    description: str
    arch: str = Field(description="Network architecture, e.g. RRDBNet")
    scale: int = Field(description="Native upscale factor of these weights")
    supports_denoise: bool = Field(
        description="Whether a denoise strength control applies to this model"
    )
    downloaded: bool = Field(description="Whether the weights are present on disk")
    size_mb: float | None = Field(
        default=None, description="On-disk size, null when not downloaded"
    )
