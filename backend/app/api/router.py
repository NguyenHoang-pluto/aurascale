"""Aggregate API router.

Feature routers are mounted here as their phases land:
  Phase 9 — result previews and crops
  Phase 10 — job history and thumbnails
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, jobs, models, system

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(models.router)
api_router.include_router(jobs.router)
