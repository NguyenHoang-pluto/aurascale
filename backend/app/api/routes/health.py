"""Liveness endpoint. Intentionally free of database and GPU dependencies so it
still answers when those are degraded."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app import __version__
from app.core.config import get_settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["system"])

_STARTED_AT = time.monotonic()


@router.get("/health", response_model=HealthResponse, summary="Backend health")
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
    )
