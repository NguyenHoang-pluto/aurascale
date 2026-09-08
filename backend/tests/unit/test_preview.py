"""Preview generation and crop validation.

The preview is what stops a 200 MP result from being handed to a browser
whole, so what matters here is that it is genuinely capped, that it never
invents pixels for a small result, and that an invalid crop is refused rather
than quietly moved somewhere valid.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import CorruptedImageError, ValidationError
from app.services.image_service import (
    MAX_CROP_PIXELS,
    PREVIEW_MAX_EDGE,
    CropRegion,
    ImageService,
)


@pytest.fixture
def service(settings: Settings) -> ImageService:
    return ImageService(settings)


def write_image(path: Path, width: int, height: int, image_format: str = "PNG") -> Path:
    """A noisy image, so a resample cannot be mistaken for the original."""
    generator = np.random.default_rng(5)
    array = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path, format=image_format)
    return path


# ------------------------------------------------------------------ preview


def test_a_large_result_is_capped_on_its_long_edge(service: ImageService, tmp_path: Path) -> None:
    source = write_image(tmp_path / "big.png", 6000, 3000)

    preview = service.write_preview(source, tmp_path / "preview.jpg")

    with Image.open(preview) as image:
        assert max(image.size) == PREVIEW_MAX_EDGE
        # The aspect ratio survives the cap.
        assert image.size == (PREVIEW_MAX_EDGE, PREVIEW_MAX_EDGE // 2)


def test_a_tall_result_is_capped_on_its_own_long_edge(
    service: ImageService, tmp_path: Path
) -> None:
    source = write_image(tmp_path / "tall.png", 2000, 8000)

    preview = service.write_preview(source, tmp_path / "preview.jpg")

    with Image.open(preview) as image:
        assert image.height == PREVIEW_MAX_EDGE
        assert image.width == PREVIEW_MAX_EDGE // 4


def test_a_small_result_is_never_upscaled(service: ImageService, tmp_path: Path) -> None:
    """Inventing pixels for a preview would misrepresent what the model made."""
    source = write_image(tmp_path / "small.png", 640, 480)

    preview = service.write_preview(source, tmp_path / "preview.jpg")

    with Image.open(preview) as image:
        assert image.size == (640, 480)


def test_the_preview_is_a_jpeg_whatever_the_result_was(
    service: ImageService, tmp_path: Path
) -> None:
    for extension, image_format in [("png", "PNG"), ("webp", "WEBP"), ("jpg", "JPEG")]:
        source = write_image(tmp_path / f"result.{extension}", 800, 600, image_format)

        preview = service.write_preview(source, tmp_path / f"preview-{extension}.jpg")

        with Image.open(preview) as image:
            assert image.format == "JPEG"


def test_a_preview_is_smaller_than_the_result_it_came_from(
    service: ImageService, tmp_path: Path
) -> None:
    source = write_image(tmp_path / "big.png", 5000, 5000)

    preview = service.write_preview(source, tmp_path / "preview.jpg")

    assert preview.stat().st_size < source.stat().st_size


def test_writing_a_preview_leaves_no_partial_file_behind(
    service: ImageService, tmp_path: Path
) -> None:
    """The write is atomic, so a reader never sees a half-encoded preview."""
    source = write_image(tmp_path / "result.png", 900, 700)
    destination = tmp_path / "previews" / "preview.jpg"

    service.write_preview(source, destination)

    assert destination.is_file()
    assert list(destination.parent.glob("*.part")) == []


def test_overwriting_an_existing_preview_replaces_it(service: ImageService, tmp_path: Path) -> None:
    source = write_image(tmp_path / "result.png", 900, 700)
    destination = tmp_path / "preview.jpg"
    destination.write_bytes(b"stale")

    service.write_preview(source, destination)

    with Image.open(destination) as image:
        assert image.size == (900, 700)


def test_an_unreadable_result_is_reported_rather_than_crashing(
    service: ImageService, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n truncated")

    with pytest.raises(CorruptedImageError, match="prepared for viewing"):
        service.write_preview(broken, tmp_path / "preview.jpg")


# --------------------------------------------------------------------- crop


def test_a_crop_returns_the_requested_region_at_full_resolution(
    service: ImageService, tmp_path: Path
) -> None:
    source = write_image(tmp_path / "result.png", 2000, 1000)

    data = service.crop_to_jpeg(source, CropRegion(x=100, y=50, width=400, height=300))

    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (400, 300)
        assert image.format == "JPEG"


def test_a_crop_takes_the_pixels_that_were_asked_for(service: ImageService, tmp_path: Path) -> None:
    """A crop from the wrong place would look plausible and be wrong."""
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    canvas[0:100, 0:100] = 255  # a white square in the top-left quadrant
    source = tmp_path / "quadrants.png"
    Image.fromarray(canvas).save(source)

    top_left = service.crop_to_jpeg(source, CropRegion(x=0, y=0, width=50, height=50))
    bottom_right = service.crop_to_jpeg(source, CropRegion(x=150, y=150, width=50, height=50))

    with Image.open(io.BytesIO(top_left)) as image:
        assert np.asarray(image).mean() > 200
    with Image.open(io.BytesIO(bottom_right)) as image:
        assert np.asarray(image).mean() < 50


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [
        (0, 0, 0, 100),  # zero width
        (0, 0, 100, 0),  # zero height
        (0, 0, -10, 100),  # negative width
        (0, 0, 100, -10),  # negative height
    ],
)
def test_a_crop_without_positive_dimensions_is_refused(
    service: ImageService, x: int, y: int, width: int, height: int
) -> None:
    with pytest.raises(ValidationError, match="positive width and height"):
        service.validate_crop(x, y, width, height, bounds=(1000, 1000))


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [
        (-1, 0, 100, 100),  # before the left edge
        (0, -1, 100, 100),  # above the top edge
        (950, 0, 100, 100),  # past the right edge
        (0, 950, 100, 100),  # past the bottom edge
        (2000, 2000, 10, 10),  # nowhere near the image
    ],
)
def test_a_crop_outside_the_result_is_refused_not_clamped(
    service: ImageService, x: int, y: int, width: int, height: int
) -> None:
    """Clamping would return a different region than the one requested, and the
    viewer would draw those pixels in the wrong place."""
    with pytest.raises(ValidationError, match="outside the result"):
        service.validate_crop(x, y, width, height, bounds=(1000, 1000))


def test_a_crop_that_fills_the_result_exactly_is_allowed(service: ImageService) -> None:
    region = service.validate_crop(0, 0, 1000, 1000, bounds=(1000, 1000))

    assert region.box == (0, 0, 1000, 1000)


def test_an_enormous_crop_is_refused(service: ImageService) -> None:
    """A crop the size of the whole result is not a crop, and serving one would
    defeat the cap the preview exists to enforce."""
    edge = 8000

    with pytest.raises(ValidationError, match="too large"):
        service.validate_crop(0, 0, edge, edge, bounds=(edge, edge))


def test_the_crop_limit_is_stated_in_the_error(service: ImageService) -> None:
    with pytest.raises(ValidationError) as caught:
        service.validate_crop(0, 0, 8000, 8000, bounds=(8000, 8000))

    assert caught.value.context["limitPixels"] == MAX_CROP_PIXELS


def test_a_valid_crop_becomes_a_pillow_box(service: ImageService) -> None:
    region = service.validate_crop(10, 20, 30, 40, bounds=(100, 100))

    assert region.box == (10, 20, 40, 60)
