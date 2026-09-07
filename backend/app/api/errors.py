"""Translate exceptions into RFC 9457 problem responses."""

from __future__ import annotations

import traceback

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.exceptions import ErrorCode, PixelForgeError
from app.core.logging import get_logger

logger = get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"


def _problem_response(payload: dict[str, object], status_code: int) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, media_type=PROBLEM_MEDIA_TYPE)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so no endpoint ever returns a bare traceback."""

    @app.exception_handler(PixelForgeError)
    async def _handle_known(_: Request, exc: PixelForgeError) -> JSONResponse:
        logger.warning(
            "request failed",
            extra={"code": exc.code.value, "status": exc.status_code, **exc.context},
        )
        return _problem_response(exc.to_problem(), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _problem_response(
            {
                "type": f"https://pixelforge.ai/errors/{ErrorCode.INVALID_PARAMETERS.value}",
                "title": "Invalid request",
                "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "code": ErrorCode.INVALID_PARAMETERS.value,
                "detail": "One or more request fields are invalid.",
                "technical": str(exc.errors()),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem_response(
            {
                "type": "about:blank",
                "title": "Request failed",
                "status": exc.status_code,
                "code": ErrorCode.INTERNAL_ERROR.value,
                "detail": str(exc.detail),
            },
            exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Unexpected failures are logged in full but never echoed verbatim in
        # production, where the traceback could disclose filesystem layout.
        logger.exception("unhandled exception")
        include_trace = get_settings().environment != "production"
        payload: dict[str, object] = {
            "type": f"https://pixelforge.ai/errors/{ErrorCode.INTERNAL_ERROR.value}",
            "title": "Something went wrong",
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "code": ErrorCode.INTERNAL_ERROR.value,
            "detail": "An unexpected error occurred. Please try again.",
        }
        if include_trace:
            payload["technical"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        return _problem_response(payload, status.HTTP_500_INTERNAL_SERVER_ERROR)
