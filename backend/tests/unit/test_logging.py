from __future__ import annotations

import json
import logging

from app.core.logging import JsonFormatter, TextContextFormatter, get_logger


def _record(**context: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="job finished",
        args=None,
        exc_info=None,
    )
    for key, value in context.items():
        setattr(record, key, value)
    return record


def test_text_formatter_appends_key_value_context() -> None:
    formatted = TextContextFormatter(fmt="%(message)s").format(
        _record(job_id="abc123", model="RealESRGAN_x4plus", scale=4)
    )

    assert formatted == "job finished job_id=abc123 model=RealESRGAN_x4plus scale=4"


def test_json_formatter_emits_context_as_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record(job_id="abc123", status="completed")))

    assert payload["message"] == "job finished"
    assert payload["level"] == "INFO"
    assert payload["job_id"] == "abc123"
    assert payload["status"] == "completed"


def test_context_adapter_merges_bound_and_call_site_context(caplog) -> None:  # type: ignore[no-untyped-def]
    logger = get_logger("app.test.adapter", job_id="abc123")

    with caplog.at_level(logging.INFO):
        logger.info("stage complete", extra={"stage": "inference"})

    record = caplog.records[-1]
    assert record.job_id == "abc123"  # type: ignore[attr-defined]
    assert record.stage == "inference"  # type: ignore[attr-defined]


def test_uvicorn_color_message_extra_is_not_rendered() -> None:
    """uvicorn passes an ANSI-coloured copy of its message via `extra`;
    rendering it would duplicate every uvicorn line in the log."""
    formatted = TextContextFormatter(fmt="%(message)s").format(
        _record(color_message="Started server process [\x1b[36m%d\x1b[0m]", port=8000)
    )

    assert formatted == "job finished port=8000"
