"""Shared response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CamelModel(BaseModel):
    """Base model that serialises as camelCase for the TypeScript client.

    Population by field name stays enabled so Python code can construct these
    with snake_case keyword arguments.
    """

    model_config = ConfigDict(
        alias_generator=lambda field: "".join(
            part.capitalize() if index else part for index, part in enumerate(field.split("_"))
        ),
        populate_by_name=True,
        from_attributes=True,
    )


class ProblemDetail(CamelModel):
    """RFC 9457 problem document — the shape of every error response."""

    type: str = Field(description="Stable URI identifying the error class")
    title: str = Field(description="Short human-readable summary")
    status: int = Field(description="HTTP status code")
    code: str = Field(description="Machine-readable error code")
    detail: str = Field(description="Human-readable explanation, safe to display")
    technical: str | None = Field(
        default=None, description="Developer detail for the collapsible panel"
    )
    context: dict[str, Any] | None = Field(
        default=None, description="Structured hints, e.g. limits that were exceeded"
    )


class HealthResponse(CamelModel):
    status: str = Field(description="'ok' when the service is able to serve requests")
    version: str
    environment: str
    uptime_seconds: float
