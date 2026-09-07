"""Server-side validation and encoding.

Every image here is produced by a real encoder rather than assembled by hand:
the point of these checks is what a real file does, and a handwritten header
would pass a sniff test while proving nothing about the decoder.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import (
    CorruptedImageError,
    FileTooLargeError,
    ImageTooLargeError,
    ImageTooSmallError,
    OutputTooLargeError,
    UnsupportedFormatError,
    ValidationError,
)
from app.models.enums import OutputFormat
from app.services.image_service import ImageService, sniff_format


@pytest.fixture
def service(settings: Settings) -> ImageService:
    return ImageService(settings)


def make_image(width: int = 64, height: int = 48, mode: str = "RGB", seed: int = 1) -> Image.Image:
    generator = np.random.default_rng(seed)
    channels = len(mode)
    array = generator.integers(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(array.squeeze() if channels == 1 else array, mode=mode)


def write_image(
    path: Path, image: Image.Image, image_format: str = "PNG", **options: object
) -> Path:
    image.save(path, format=image_format, **options)
    return path


# ------------------------------------------------------------------ sniffing


def test_each_supported_format_is_recognised_from_real_encoder_output(tmp_path: Path) -> None:
    for image_format, extension in [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")]:
        path = write_image(tmp_path / f"real.{extension}", make_image(), image_format)
        assert sniff_format(path.read_bytes()[:12]) == image_format


def test_an_unsupported_format_is_not_recognised(tmp_path: Path) -> None:
    path = write_image(tmp_path / "real.gif", make_image(mode="P"), "GIF")

    assert sniff_format(path.read_bytes()[:12]) is None


def test_the_extension_is_never_consulted(service: ImageService, tmp_path: Path) -> None:
    """A renamed file is identified by its contents, not by what it claims."""
    disguised = tmp_path / "photo.jpg"
    write_image(disguised, make_image(), "PNG")

    declared, _, _ = service.inspect(disguised)

    assert declared == "PNG"


def test_a_renamed_non_image_is_refused(service: ImageService, tmp_path: Path) -> None:
    hostile = tmp_path / "payload.png"
    hostile.write_bytes(b"MZ\x90\x00\x03" + b"\x00" * 64)  # a PE header

    with pytest.raises(UnsupportedFormatError, match="not a JPEG, PNG or WEBP"):
        service.inspect(hostile)


# ---------------------------------------------------------------- structure


def test_a_truncated_file_is_rejected(service: ImageService, tmp_path: Path) -> None:
    """The signature survives truncation, so only a decode attempt finds this."""
    whole = tmp_path / "whole.png"
    write_image(whole, make_image(256, 256), "PNG")

    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(whole.read_bytes()[: len(whole.read_bytes()) // 2])

    with pytest.raises(CorruptedImageError):
        service.inspect(truncated)


def test_a_valid_image_reports_its_real_dimensions(service: ImageService, tmp_path: Path) -> None:
    path = write_image(tmp_path / "image.png", make_image(120, 90), "PNG")

    declared, width, height = service.inspect(path)

    assert (declared, width, height) == ("PNG", 120, 90)


# --------------------------------------------------------------- dimensions


def test_an_image_below_the_minimum_dimension_is_refused(
    service: ImageService, tmp_path: Path
) -> None:
    path = write_image(tmp_path / "tiny.png", make_image(16, 16), "PNG")

    with pytest.raises(ImageTooSmallError, match="at least"):
        service.inspect(path)


def test_an_image_beyond_the_pixel_limit_is_refused_before_decoding(
    tmp_path: Path, settings: Settings
) -> None:
    """The guard is the reason a decompression bomb never reaches the decoder."""
    tight = Settings(
        environment="test",
        storage_dir=settings.storage_dir,
        models_dir=settings.models_dir,
        max_input_pixels=1_000,
    )
    path = write_image(tmp_path / "large.png", make_image(200, 200), "PNG")

    with pytest.raises(ImageTooLargeError, match="MP"):
        ImageService(tight).inspect(path)


def test_a_projected_output_beyond_the_limit_is_refused(service: ImageService) -> None:
    with pytest.raises(OutputTooLargeError, match="Try a smaller upscale factor"):
        service.assert_output_fits(4000, 4000, 8)


def test_a_reasonable_output_is_allowed(service: ImageService) -> None:
    service.assert_output_fits(1280, 720, 4)


# ------------------------------------------------------------------ streaming


def test_an_upload_is_written_to_disk(service: ImageService, tmp_path: Path) -> None:
    payload = b"x" * 5000
    destination = tmp_path / "upload.bin"

    written = service.stream_to_file(io.BytesIO(payload), destination)

    assert written == len(payload)
    assert destination.read_bytes() == payload


def test_an_oversized_upload_is_aborted_and_leaves_nothing_behind(
    tmp_path: Path, settings: Settings
) -> None:
    """The cap is enforced against bytes received, not a declared length."""
    tight = Settings(
        environment="test",
        storage_dir=settings.storage_dir,
        models_dir=settings.models_dir,
        max_upload_size_mb=1,
    )
    destination = tmp_path / "big.bin"

    with pytest.raises(FileTooLargeError):
        ImageService(tight).stream_to_file(io.BytesIO(b"x" * (2 * 1024 * 1024)), destination)

    assert not destination.exists()


def test_an_empty_upload_is_refused(service: ImageService, tmp_path: Path) -> None:
    destination = tmp_path / "empty.bin"

    with pytest.raises(CorruptedImageError, match="empty"):
        service.stream_to_file(io.BytesIO(b""), destination)

    assert not destination.exists()


# -------------------------------------------------------------------- decode


def test_decoding_returns_rgb_pixels(service: ImageService, tmp_path: Path) -> None:
    path = write_image(tmp_path / "image.png", make_image(64, 48), "PNG")

    decoded = service.decode(path)

    assert decoded.pixels.shape == (48, 64, 3)
    assert decoded.pixels.dtype == np.uint8
    assert decoded.has_alpha is False
    assert decoded.source_format == "PNG"


def test_an_rgba_image_keeps_its_alpha_channel(service: ImageService, tmp_path: Path) -> None:
    path = write_image(tmp_path / "image.png", make_image(64, 48, mode="RGBA"), "PNG")

    decoded = service.decode(path)

    assert decoded.pixels.shape == (48, 64, 4)
    assert decoded.has_alpha is True


def test_greyscale_is_widened_to_three_channels(service: ImageService, tmp_path: Path) -> None:
    path = write_image(tmp_path / "grey.png", make_image(64, 48, mode="L"), "PNG")

    decoded = service.decode(path)

    assert decoded.pixels.shape[2] == 3


def test_exif_orientation_is_applied_to_the_pixels(service: ImageService, tmp_path: Path) -> None:
    """Orientation 6 means "rotate 90 degrees", which transposes the dimensions.

    Applying it here and clearing the tag is what stops a viewer that honours
    the tag rotating the result a second time.
    """
    image = make_image(80, 40)
    exif = Image.Exif()
    exif[0x0112] = 6

    path = tmp_path / "rotated.jpg"
    image.save(path, format="JPEG", exif=exif)

    decoded = service.decode(path)

    assert (decoded.width, decoded.height) == (40, 80)
    assert decoded.pixels.shape[:2] == (80, 40)


def test_an_icc_profile_is_carried_off_the_source(service: ImageService, tmp_path: Path) -> None:
    profile = b"fake-icc-profile"
    path = tmp_path / "tagged.png"
    make_image().save(path, format="PNG", icc_profile=profile)

    assert service.decode(path).icc_profile == profile


# -------------------------------------------------------------------- encode


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [(OutputFormat.PNG, "PNG"), (OutputFormat.JPEG, "JPEG"), (OutputFormat.WEBP, "WEBP")],
)
def test_each_output_format_is_written(
    service: ImageService, tmp_path: Path, output_format: OutputFormat, expected: str
) -> None:
    pixels = np.asarray(make_image(32, 32))
    destination = tmp_path / f"out.{output_format.extension}"

    size = service.encode(pixels, destination, output_format=output_format, quality=90)

    assert size > 0
    with Image.open(destination) as written:
        assert written.format == expected


def test_alpha_is_composited_onto_white_for_jpeg(service: ImageService, tmp_path: Path) -> None:
    """JPEG has no alpha channel, so the choice is visible rather than silent."""
    pixels = np.dstack([np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((16, 16), dtype=np.uint8)])
    destination = tmp_path / "out.jpg"

    service.encode(pixels, destination, output_format=OutputFormat.JPEG, quality=90)

    with Image.open(destination) as written:
        assert written.mode == "RGB"
        assert np.asarray(written).mean() > 200  # transparent became white


def test_metadata_is_carried_through_when_preserved(service: ImageService, tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    exif = Image.Exif()
    exif[0x010E] = "a description"
    make_image().save(source_path, format="JPEG", exif=exif)
    decoded = service.decode(source_path)

    destination = tmp_path / "out.jpg"
    service.encode(
        decoded.pixels,
        destination,
        output_format=OutputFormat.JPEG,
        quality=90,
        source=decoded,
        preserve_metadata=True,
    )

    with Image.open(destination) as written:
        assert written.getexif().get(0x010E) == "a description"


def test_metadata_is_dropped_when_not_preserved(service: ImageService, tmp_path: Path) -> None:
    """Stripping takes EXIF wholesale, which necessarily takes GPS with it."""
    source_path = tmp_path / "source.jpg"
    exif = Image.Exif()
    exif[0x010E] = "a description"
    exif[0x8825] = {1: "N"}  # GPS
    make_image().save(source_path, format="JPEG", exif=exif)
    decoded = service.decode(source_path)

    destination = tmp_path / "out.jpg"
    service.encode(
        decoded.pixels,
        destination,
        output_format=OutputFormat.JPEG,
        quality=90,
        source=decoded,
        preserve_metadata=False,
    )

    with Image.open(destination) as written:
        assert written.getexif().get(0x010E) is None
        assert written.getexif().get(0x8825) is None


def test_the_colour_profile_survives_stripping(service: ImageService, tmp_path: Path) -> None:
    """An ICC profile describes a colour space; losing it changes the colours."""
    source_path = tmp_path / "source.png"
    make_image().save(source_path, format="PNG", icc_profile=b"fake-icc")
    decoded = service.decode(source_path)

    destination = tmp_path / "out.png"
    service.encode(
        decoded.pixels,
        destination,
        output_format=OutputFormat.PNG,
        source=decoded,
        preserve_metadata=False,
    )

    with Image.open(destination) as written:
        assert written.info.get("icc_profile") == b"fake-icc"


@pytest.mark.parametrize("quality", [10, 49, 101, 200])
def test_an_out_of_range_quality_is_refused(service: ImageService, quality: int) -> None:
    with pytest.raises(ValidationError, match="Quality must be"):
        service.validate_quality(quality)


def test_quality_defaults_when_unspecified(service: ImageService) -> None:
    assert service.validate_quality(None) == 92
