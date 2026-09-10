"""FastAPI application factory.

Business logic lives in `app.services`; this module only wires the app together
(§ 22: nothing substantial belongs in main.py).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging import configure_logging

DESCRIPTION = """
Backend for **PixelForge AI** — real image super-resolution with Real-ESRGAN.

Jobs are created with `POST /api/jobs`, progress is streamed over Server-Sent
Events, and the finished image is retrieved from `/api/jobs/{jobId}/result`.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    # The interactive docs are a development tool, and in production they are a
    # map of the API for anyone who finds the host. Passing `None` unmounts the
    # routes rather than hiding them, so there is nothing left to reach.
    #
    # Local runs are unaffected: `environment` defaults to `development`, which
    # is where the docs are worth having. What this closes is the case where a
    # tunnel or a reverse proxy is pointed straight at the API instead of at the
    # frontend, and `/docs` answers.
    served = not settings.is_production

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if served else None,
        redoc_url="/redoc" if served else None,
        openapi_url="/openapi.json" if served else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
