"""Aggregate API router.

Feature routers are mounted here as their phases land:
  Phase 5  — system, models
  Phase 7  — jobs, job events (SSE)
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
