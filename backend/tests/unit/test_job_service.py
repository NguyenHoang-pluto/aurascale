"""Job service rules that are worth testing without an HTTP round trip.

The lifecycle itself is covered through the API in tests/api/test_jobs_api.py;
what is here is the option parsing and the naming rules, where the interesting
cases are hostile input rather than happy paths.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import ValidationError
from app.models.db import Job
from app.models.enums import JobStatus, OutputFormat
from app.services.job_service import (
    EnhanceOptions,
    JobService,
    _not_completed_message,
    _safe_original_name,
    parse_options,
)

# ------------------------------------------------------------------ options


def test_no_settings_means_defaults() -> None:
    options = parse_options(None)

    assert options == EnhanceOptions()
    assert options.denoise_strength is None


def test_settings_are_parsed() -> None:
    options = parse_options(
        '{"sharpenStrength": 0.4, "denoiseStrength": 0.6, "tileSize": 128, "tilePad": 8}'
    )

    assert options.sharpen_strength == 0.4
    assert options.denoise_strength == 0.6
    assert options.tile_size == 128
    assert options.tile_pad == 8


def test_malformed_json_is_refused() -> None:
    with pytest.raises(ValidationError, match="valid JSON"):
        parse_options("{oops")


def test_a_json_array_is_refused() -> None:
    with pytest.raises(ValidationError, match="JSON object"):
        parse_options("[1, 2, 3]")


@pytest.mark.parametrize("value", [-0.1, 1.1, 42])
def test_a_strength_outside_zero_to_one_is_refused(value: float) -> None:
    with pytest.raises(ValidationError, match="must be between"):
        parse_options(f'{{"sharpenStrength": {value}}}')


def test_a_non_numeric_strength_is_refused() -> None:
    with pytest.raises(ValidationError, match="number"):
        parse_options('{"denoiseStrength": "high"}')


def test_a_boolean_is_not_a_number() -> None:
    """`True` is an int in Python; accepting it would be an accident."""
    with pytest.raises(ValidationError, match="number"):
        parse_options('{"sharpenStrength": true}')


@pytest.mark.parametrize("value", [-1, 4096])
def test_a_tile_size_outside_the_bounds_is_refused(value: int) -> None:
    with pytest.raises(ValidationError, match="between"):
        parse_options(f'{{"tileSize": {value}}}')


def test_a_fractional_tile_size_is_refused() -> None:
    with pytest.raises(ValidationError, match="whole number"):
        parse_options('{"tileSize": 12.5}')


def test_options_serialise_only_what_was_set() -> None:
    """An absent option must stay absent, or "unset" becomes "zero"."""
    payload = EnhanceOptions(sharpen_strength=0.2).to_json()

    assert payload == {"sharpenStrength": 0.2}


def test_options_round_trip_through_json() -> None:
    original = EnhanceOptions(sharpen_strength=0.5, denoise_strength=0.25, tile_size=64, tile_pad=4)

    assert parse_options(__import__("json").dumps(original.to_json())) == original


# ----------------------------------------------------------------- filenames


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("holiday.png", "holiday.png"),
        ("../../etc/passwd", "passwd"),
        ("C:\\\\Windows\\\\system32\\\\evil.png", "evil.png"),
        ("  spaced.png  ", "spaced.png"),
        ("", None),
        (None, None),
    ],
)
def test_the_uploaded_name_is_stripped_of_anything_path_like(
    supplied: str | None, expected: str | None
) -> None:
    """It is kept for display only, and never reaches the filesystem."""
    assert _safe_original_name(supplied) == expected


def test_a_very_long_name_is_truncated() -> None:
    assert len(_safe_original_name("a" * 500) or "") == 255


def test_the_download_name_is_built_from_the_job_not_the_upload() -> None:
    job = Job(
        id="abcdef0123456789abcdef0123456789",
        status=JobStatus.COMPLETED,
        model_name="RealESRGAN_x4plus",
        scale=4,
        output_format=OutputFormat.PNG.value,
        input_path="x",
        input_width=100,
        input_height=100,
        input_bytes=1,
        input_format="PNG",
        original_filename="../../secret.png",
        output_width=400,
        output_height=400,
    )

    name = JobService.download_name(None, job)  # type: ignore[arg-type]

    assert name == "pixelforge-abcdef01-400x400.png"
    assert "secret" not in name


def test_the_download_name_survives_a_job_without_dimensions() -> None:
    job = Job(
        id="0" * 32,
        status=JobStatus.COMPLETED,
        model_name="m",
        scale=4,
        output_format=OutputFormat.JPEG.value,
        input_path="x",
        input_width=1,
        input_height=1,
        input_bytes=1,
        input_format="PNG",
    )

    assert JobService.download_name(None, job) == "pixelforge-00000000.jpg"  # type: ignore[arg-type]


# ------------------------------------------------------------------ messages


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (JobStatus.FAILED, "failed"),
        (JobStatus.CANCELLED, "cancelled"),
        (JobStatus.QUEUED, "not finished"),
        (JobStatus.PROCESSING, "not finished"),
    ],
)
def test_the_reason_a_result_is_unavailable_is_specific(status: JobStatus, fragment: str) -> None:
    """ "Not completed" covers three different situations; each gets its own."""
    assert fragment in _not_completed_message(status)
