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
from app.services.image_service import (
    PREVIEW_MAX_EDGE,
    THUMBNAIL_MAX_EDGE,
    ImageService,
    open_trusted,
    sniff_format,
)


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


# ------------------------------------------------- trusted result decoding


def test_a_result_larger_than_pillows_bomb_limit_can_be_opened(
    service: ImageService, tmp_path: Path
) -> None:
    """An 8x result legitimately exceeds Pillow's limit, and is ours to read.

    The limit is lowered rather than a 192 MP file being built, so the test
    costs milliseconds instead of half a gigabyte. What is being checked is the
    guard, and the guard does not know how big "too big" happens to be.
    """
    source = write_image(tmp_path / "result.jpg", make_image(320, 240), "JPEG")
    destination = tmp_path / "preview.jpg"

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 16  # anything real is now "a bomb"
    try:
        # Plain Image.open refuses at this limit...
        with pytest.raises(Image.DecompressionBombError), Image.open(source) as probe:
            probe.load()

        # ...but our own result is opened anyway.
        service.write_preview(source, destination)
    finally:
        Image.MAX_IMAGE_PIXELS = previous

    assert destination.is_file()
    with Image.open(destination) as written:
        assert written.size == (320, 240)


def test_the_bomb_guard_is_restored_after_a_trusted_open(
    service: ImageService, tmp_path: Path
) -> None:
    """The guard is global, so leaving it off would disarm every later open."""
    source = write_image(tmp_path / "result.png", make_image(64, 48))

    sentinel = 12_345_678
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = sentinel
    try:
        service.write_preview(source, tmp_path / "preview.jpg")
        assert sentinel == Image.MAX_IMAGE_PIXELS
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def test_the_bomb_guard_is_restored_even_when_the_open_fails(tmp_path: Path) -> None:
    """`finally`, not a happy-path restore: an exception must not disarm it."""
    source = write_image(tmp_path / "result.png", make_image(32, 32))

    sentinel = 999_999
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = sentinel
    try:
        with pytest.raises(RuntimeError, match="boom"), open_trusted(source):
            raise RuntimeError("boom")

        assert sentinel == Image.MAX_IMAGE_PIXELS
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def test_an_upload_above_the_pixel_limit_is_still_refused(
    settings: Settings, tmp_path: Path
) -> None:
    """The trusted path must not have loosened anything for user uploads.

    `MAX_INPUT_PIXELS` is the real upload defence - it is read from the header
    before any decode, and is stricter than Pillow's guard by an order of
    magnitude - so it is what this asserts.
    """
    tight = Settings(
        environment="test",
        storage_dir=settings.storage_dir,
        models_dir=settings.models_dir,
        max_input_pixels=1_000,
    )
    source = write_image(tmp_path / "upload.png", make_image(64, 48))  # 3072 px

    with pytest.raises(ImageTooLargeError):
        ImageService(tight).inspect(source)


def test_a_bomb_error_from_a_result_becomes_a_corrupted_image_error(
    service: ImageService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It should be unreachable now; if it ever fires it must not be a 500.

    `DecompressionBombError` inherits from Exception, not OSError, so before
    this it escaped the handler and left the route returning `internal_error`.
    """
    source = write_image(tmp_path / "result.png", make_image(64, 48))

    def explode(*_: object, **__: object) -> None:
        raise Image.DecompressionBombError("simulated")

    monkeypatch.setattr(Image, "open", explode)

    with pytest.raises(CorruptedImageError, match="could not be prepared for viewing"):
        service.write_preview(source, tmp_path / "preview.jpg")


def test_a_bomb_error_from_a_crop_becomes_a_corrupted_image_error(
    service: ImageService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_image(tmp_path / "result.png", make_image(64, 48))
    region = service.validate_crop(0, 0, 16, 16, bounds=(64, 48))

    def explode(*_: object, **__: object) -> None:
        raise Image.DecompressionBombError("simulated")

    monkeypatch.setattr(Image, "open", explode)

    with pytest.raises(CorruptedImageError, match="could not be read"):
        service.crop_to_jpeg(source, region)


@pytest.mark.slow
def test_a_real_16000x12000_result_produces_a_preview_and_a_thumbnail(
    service: ImageService, tmp_path: Path
) -> None:
    """The reported case, at full size: 2000x1500 at 8x is 192 MP.

    Marked slow because it genuinely encodes a 192 MP JPEG. The cheap tests
    above cover the same guard; this one proves the whole path holds at the
    size that actually failed in production, and that `draft()` still keeps the
    decode small.
    """
    source = tmp_path / "result-8x.jpg"

    # Built one strip at a time: a single 16000x12000x3 array would be 549 MiB.
    with Image.new("RGB", (16000, 12000)) as canvas:
        strip = make_image(16000, 500)
        for top in range(0, 12000, 500):
            canvas.paste(strip, (0, top))
        canvas.save(source, format="JPEG", quality=80)

    # `open_trusted`, because a plain open here would trip the very guard this
    # test exists to work around.
    with open_trusted(source) as written:
        assert written.size == (16000, 12000)
        assert written.size[0] * written.size[1] == 192_000_000

    preview = service.write_preview(source, tmp_path / "preview.jpg")
    thumbnail = service.write_thumbnail(source, tmp_path / "thumb.webp")

    assert preview.is_file()
    assert thumbnail.is_file()

    with Image.open(preview) as opened:
        assert max(opened.size) == PREVIEW_MAX_EDGE
    with Image.open(thumbnail) as opened:
        assert max(opened.size) == THUMBNAIL_MAX_EDGE


# ------------------------------------------------------------ jpeg chroma


def sampling_of(path: Path) -> int:
    """0 is 4:4:4, 2 is 4:2:0. Read back from the encoded file, not asserted
    from the options we passed - the point is what libjpeg actually wrote."""
    from PIL import JpegImagePlugin

    with Image.open(path) as opened:
        return int(JpegImagePlugin.get_sampling(opened))


def test_jpeg_is_written_with_full_chroma_resolution(service: ImageService, tmp_path: Path) -> None:
    """Pillow's default is 4:2:0 at every quality, 100 included.

    That halved chroma on the file a user downloads as their finished result.
    It is not luminance detail - subsampling never touched luma - but it is
    the colour detail in skin, foliage and saturated edges.
    """
    destination = tmp_path / "out.jpg"

    service.encode(
        np.asarray(make_image(64, 48)), destination, output_format=OutputFormat.JPEG, quality=92
    )

    assert sampling_of(destination) == 0


@pytest.mark.parametrize("quality", [50, 75, 92, 100])
def test_full_chroma_holds_at_every_quality(
    service: ImageService, tmp_path: Path, quality: int
) -> None:
    destination = tmp_path / f"out-{quality}.jpg"

    service.encode(
        np.asarray(make_image(64, 48)),
        destination,
        output_format=OutputFormat.JPEG,
        quality=quality,
    )

    assert sampling_of(destination) == 0


def test_full_chroma_preserves_colour_detail_a_subsampled_file_loses(
    service: ImageService, tmp_path: Path
) -> None:
    """The measurable point of the change, on chroma the eye can see.

    Alternating red and blue columns put all the detail in chroma and none in
    luma, which is exactly what 4:2:0 discards.
    """
    columns = np.zeros((64, 64, 3), dtype=np.uint8)
    columns[:, 0::2] = (200, 40, 40)
    columns[:, 1::2] = (40, 40, 200)
    source = Image.fromarray(columns)

    full = tmp_path / "full.jpg"
    service.encode(columns, full, output_format=OutputFormat.JPEG, quality=92)

    subsampled = tmp_path / "subsampled.jpg"
    source.save(subsampled, format="JPEG", quality=92, subsampling=2)

    def chroma_error(path: Path) -> float:
        with Image.open(path) as opened:
            decoded = np.asarray(opened.convert("RGB"), dtype=np.int16)
        return float(np.abs(decoded - columns.astype(np.int16)).mean())

    assert chroma_error(full) < chroma_error(subsampled)


def test_the_jpeg_still_decodes_at_the_right_size_and_mode(
    service: ImageService, tmp_path: Path
) -> None:
    destination = tmp_path / "out.jpg"

    service.encode(
        np.asarray(make_image(70, 50)), destination, output_format=OutputFormat.JPEG, quality=92
    )

    with Image.open(destination) as opened:
        assert opened.format == "JPEG"
        assert opened.size == (70, 50)
        assert opened.convert("RGB").size == (70, 50)


def test_png_is_untouched_by_the_chroma_change(service: ImageService, tmp_path: Path) -> None:
    """PNG has no chroma subsampling and must not have gained an option."""
    destination = tmp_path / "out.png"

    service.encode(np.asarray(make_image(32, 32)), destination, output_format=OutputFormat.PNG)

    with Image.open(destination) as opened:
        assert opened.format == "PNG"


def test_a_full_resolution_crop_also_uses_full_chroma(
    service: ImageService, tmp_path: Path
) -> None:
    """Since Phase 1 this is what the viewer shows from 100 % upward."""
    from PIL import JpegImagePlugin

    source = write_image(tmp_path / "result.png", make_image(64, 64))
    region = service.validate_crop(0, 0, 32, 32, bounds=(64, 64))

    encoded = service.crop_to_jpeg(source, region)

    with Image.open(io.BytesIO(encoded)) as opened:
        assert int(JpegImagePlugin.get_sampling(opened)) == 0
