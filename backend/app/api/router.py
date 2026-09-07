"""Aggregate API router.

Feature routers are mounted here as their phases land:
  Phase 7 — jobs, job events (SSE)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, models, system

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(models.router)
