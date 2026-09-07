"""Structured logging.

Log records carry job/request context as extra fields so the text formatter can
render `job_id=... model=... status=...` (§ 25) and the JSON formatter can emit
the same data for log aggregation. No user file names or EXIF content is logged.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

from app.core.config import LogFormat

# Attributes present on every LogRecord; anything else was supplied by the
# caller via `extra=` and is therefore structured context worth emitting.
_RESERVED: frozenset[str] = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    # uvicorn attaches an ANSI-coloured duplicate of its own message; it is
    # noise in our formatters, which do their own rendering.
    "color_message",
}


def _safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Rename context keys that collide with LogRecord's own attributes.

    `logging` raises KeyError at emit time if `extra` contains a reserved name
    such as "name", "module" or "args". That turns a logging call into a crash,
    which is intolerable on error-handling paths where the log line is often
    the last thing standing. Colliding keys are prefixed instead of dropped, so
    no context is silently lost.
    """
    safe: dict[str, Any] = {}
    for key, value in context.items():
        safe[f"ctx_{key}" if key in _RESERVED else key] = value
    return safe


def _context_of(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class TextContextFormatter(logging.Formatter):
    """Human-readable formatter that appends `key=value` context."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = _context_of(record)
        if not context:
            return base
        rendered = " ".join(f"{k}={v}" for k, v in context.items())
        return f"{base} {rendered}"


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_context_of(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_format: LogFormat = "text") -> None:
    """Install the root handler. Idempotent — safe under uvicorn reload."""
    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = TextContextFormatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger that stamps `context` onto every record it emits."""
    return _ContextAdapter(logging.getLogger(name), context)


class _ContextAdapter(logging.LoggerAdapter[logging.Logger]):
    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra: dict[str, Any] = dict(self.extra or {})
        supplied = kwargs.get("extra")
        if isinstance(supplied, Mapping):
            extra.update(supplied)
        kwargs["extra"] = _safe_context(extra)
        return msg, kwargs
