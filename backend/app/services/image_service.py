"""Decoding, validating and encoding images.

This is stages 1-5 and 12 of docs/image-processing.md. The browser runs the
same checks in the same order for immediate feedback, but those are a
convenience: the client is not trusted, so everything is repeated here.

The order is deliberate and each step protects the next:

  1. size, while streaming, before the body is buffered;
  2. magic bytes, because the extension and the declared MIME type are both
     attacker-controlled;
  3. `verify()`, which catches truncation and structural corruption;
  4. dimensions, checked from the header *before* a full decode, which is what
     makes it a decompression-bomb defence rather than a cosmetic limit;
  5. the decode itself, applying EXIF orientation to the pixels.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

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
from app.core.logging import get_logger
from app.models.enums import OutputFormat

logger = get_logger(__name__)

# Enough bytes to identify every format we accept. WEBP needs 12: "RIFF", four
# size bytes, then "WEBP".
SIGNATURE_BYTES = 12
UPLOAD_CHUNK_BYTES = 1024 * 64

JPEG_QUALITY_RANGE = (50, 100)

# The comparison viewer's base layer. Long edge capped so a 200 MP result is
# never handed to a browser whole; JPEG because this is a view, not a download.
PREVIEW_MAX_EDGE = 4096
PREVIEW_QUALITY = 92

# The largest region a crop request may ask for. A "crop" the size of the whole
# result is not a crop, and serving one would defeat the cap above.
MAX_CROP_PIXELS = 4096 * 4096

# Serialises the `MAX_IMAGE_PIXELS` save/restore in `open_trusted`. Previews and
# thumbnails are built on the thread pool, so two of them can overlap; without
# this, the second to enter would save the *disabled* value as its "previous"
# and leave the guard off for good when it restored.
_TRUSTED_OPEN_LOCK = threading.Lock()


@contextmanager
def open_trusted(path: Path) -> Iterator[Image.Image]:
    """Open a file this application produced, without the bomb guard.

    Pillow refuses to open anything above `MAX_IMAGE_PIXELS` (89.5 MP, or a
    hard error above twice that) because an image far larger than its
    compressed size is how a decompression bomb works. That reasoning applies
    to files a stranger uploads. It does not apply here: the only caller is the
    preview/thumbnail/crop path, and it is reading a result *this process just
    wrote* from pixels it had already decoded. An 8x job legitimately reaches
    192 MP, and refusing to open our own output means a completed job whose
    result downloads but cannot be viewed.

    Deliberately not a global change:

      * `MAX_INPUT_PIXELS` is what actually protects uploads, and it is
        untouched. `inspect()` reads the dimensions from the header and calls
        `_assert_dimensions` *before* any decode, so an oversized upload is
        refused at 16 MP - twelve times stricter than Pillow's own limit, and
        applied earlier. Upload safety has never depended on this setting.
      * the previous value is restored in `finally`, so an exception inside the
        block cannot leave the guard off;
      * the lock keeps two concurrent trusted opens from interleaving their
        save/restore.
    """
    with _TRUSTED_OPEN_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as opened:
                yield opened
        finally:
            Image.MAX_IMAGE_PIXELS = previous


# The history grid's tile. WEBP because a grid loads many at once and the
# format is a third of the bytes of JPEG at this size.
THUMBNAIL_MAX_EDGE = 256
THUMBNAIL_QUALITY = 82


@dataclass(frozen=True, slots=True)
class CropRegion:
    """A validated region of a result, in output pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Pillow's (left, upper, right, lower)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """A decoded image and the facts about the file it came from."""

    pixels: np.ndarray[Any, Any]
    width: int
    height: int
    source_format: str
    size_bytes: int
    has_alpha: bool
    # Carried through to the output when metadata is preserved.
    icc_profile: bytes | None = None
    exif: bytes | None = None

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def sniff_format(header: bytes) -> str | None:
    """Identify a format from its leading bytes, or None if unrecognised.

    The declared content type and the filename are never consulted: both are
    supplied by the client, and a renamed executable would sail past either.
    """
    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WEBP"
    return None


class ImageService:
    """Validation and codec work, with no knowledge of jobs or HTTP."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------ intake

    def stream_to_file(self, source: BinaryIO, destination: Path) -> int:
        """Write an upload to disk, aborting the moment it exceeds the cap.

        Streaming rather than reading the body into memory is the point: the
        limit has to be enforced before a hostile upload is buffered, and
        `Content-Length` cannot be trusted to do it.
        """
        limit = self._settings.max_upload_size_bytes
        written = 0

        try:
            with destination.open("wb") as sink:
                while chunk := source.read(UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > limit:
                        raise FileTooLargeError(
                            f"That file is larger than the {self._settings.max_upload_size_mb} MB "
                            "limit.",
                            technical=f"aborted after {written} bytes",
                            context={"limitBytes": limit},
                        )
                    sink.write(chunk)
        except FileTooLargeError:
            destination.unlink(missing_ok=True)
            raise
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise CorruptedImageError(
                "The upload could not be saved.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        if written == 0:
            destination.unlink(missing_ok=True)
            raise CorruptedImageError("The uploaded file is empty.", technical="0 bytes received")

        return written

    # -------------------------------------------------------------- validation

    def inspect(self, path: Path) -> tuple[str, int, int]:
        """Format and dimensions from the header, without decoding pixels.

        Returns `(format, width, height)`. This is the check that has to happen
        before the decode, or a 100 000 x 100 000 PNG would be expanded into
        memory just to discover it is too large.
        """
        header = path.read_bytes()[:SIGNATURE_BYTES]
        sniffed = sniff_format(header)

        if sniffed is None:
            raise UnsupportedFormatError(
                "That file is not a JPEG, PNG or WEBP image.",
                technical=f"unrecognised signature {header[:8]!r}",
                context={"allowed": self._settings.allowed_formats},
            )

        if sniffed not in self._settings.allowed_formats:
            raise UnsupportedFormatError(
                f"{sniffed} images are not accepted.",
                technical=f"allowed: {', '.join(self._settings.allowed_formats)}",
                context={"allowed": self._settings.allowed_formats},
            )

        # verify() detects truncation and structural damage, and invalidates
        # the file object as it goes - so the image is opened again below. That
        # reopen is not redundant; using the verified handle raises.
        try:
            with Image.open(path) as probe:
                probe.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise CorruptedImageError(
                "That image could not be read. It may be incomplete or damaged.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            with Image.open(path) as reopened:
                width, height = reopened.size
                declared = reopened.format or sniffed
        except (UnidentifiedImageError, OSError) as exc:
            raise CorruptedImageError(
                "That image could not be read. It may be incomplete or damaged.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._assert_dimensions(width, height)
        return declared, width, height

    def _assert_dimensions(self, width: int, height: int) -> None:
        minimum = self._settings.min_input_dimension

        if width < minimum or height < minimum:
            raise ImageTooSmallError(
                f"That image is {width}x{height}. Images need to be at least "
                f"{minimum}x{minimum} pixels to have something to reconstruct.",
                technical=f"input={width}x{height} minimum={minimum}",
                context={"minimumDimension": minimum},
            )

        pixels = width * height
        if pixels > self._settings.max_input_pixels:
            raise ImageTooLargeError(
                f"That image is {pixels / 1_000_000:.1f} MP, which is larger than the "
                f"{self._settings.max_input_pixels / 1_000_000:.0f} MP limit.",
                technical=f"input={width}x{height} pixels={pixels}",
                context={"limitPixels": self._settings.max_input_pixels, "actualPixels": pixels},
            )

    def assert_output_fits(self, width: int, height: int, scale: int) -> None:
        """Refuse a job whose result would be unmanageable, before any work.

        A 4000x4000 input at 8x is 1024 MP. Refusing at submission is far
        kinder than failing after four minutes of compute.
        """
        projected = width * height * scale * scale

        if projected > self._settings.max_output_pixels:
            raise OutputTooLargeError(
                f"A {scale}x enlargement of that image would be "
                f"{projected / 1_000_000:.0f} MP, which is beyond the "
                f"{self._settings.max_output_pixels / 1_000_000:.0f} MP limit. "
                "Try a smaller upscale factor.",
                technical=f"input={width}x{height} scale={scale} projected={projected}",
                context={
                    "limitPixels": self._settings.max_output_pixels,
                    "projectedPixels": projected,
                },
            )

    # ------------------------------------------------------------------ decode

    def decode(self, path: Path) -> DecodedImage:
        """Decode to an RGB or RGBA array, orientation already applied.

        EXIF orientation is baked into the pixels and the tag normalised, so a
        viewer that honours the tag does not rotate the result a second time.
        """
        declared, width, height = self.inspect(path)
        size_bytes = path.stat().st_size

        try:
            with Image.open(path) as opened:
                icc = opened.info.get("icc_profile")
                exif = opened.info.get("exif")
                # A copy with the rotation applied to the pixels and the
                # orientation tag cleared, so nothing rotates it twice.
                oriented = ImageOps.exif_transpose(opened)

                has_alpha = oriented.mode in {"RGBA", "LA", "PA"} or "transparency" in oriented.info
                converted = oriented.convert("RGBA" if has_alpha else "RGB")
                pixels = np.asarray(converted, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise CorruptedImageError(
                "That image could not be decoded. It may be incomplete or damaged.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        # Orientation can transpose the dimensions, so they are re-read rather
        # than carried over from the header.
        actual_height, actual_width = pixels.shape[:2]
        if (actual_width, actual_height) != (width, height):
            self._assert_dimensions(actual_width, actual_height)

        return DecodedImage(
            pixels=pixels,
            width=actual_width,
            height=actual_height,
            source_format=declared,
            size_bytes=size_bytes,
            has_alpha=has_alpha,
            icc_profile=icc if isinstance(icc, bytes) else None,
            exif=exif if isinstance(exif, bytes) else None,
        )

    # ------------------------------------------------------------------ encode

    def encode(
        self,
        pixels: np.ndarray[Any, Any],
        destination: Path,
        *,
        output_format: OutputFormat,
        quality: int | None = None,
        source: DecodedImage | None = None,
        preserve_metadata: bool = True,
    ) -> int:
        """Write the result. Returns the file size in bytes."""
        image = Image.fromarray(pixels)

        if output_format is not OutputFormat.PNG and image.mode == "RGBA":
            if output_format is OutputFormat.JPEG:
                # JPEG has no alpha channel; compositing onto white is the
                # conventional, visible choice rather than silently dropping it.
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
        elif output_format is OutputFormat.JPEG and image.mode != "RGB":
            image = image.convert("RGB")

        options = self._encode_options(
            output_format, quality=quality, source=source, preserve_metadata=preserve_metadata
        )

        try:
            image.save(destination, format=output_format.name, **options)
        except (OSError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise ValidationError(
                "The result could not be saved in the requested format.",
                technical=f"{type(exc).__name__}: {exc}",
                context={"format": output_format.value},
            ) from exc

        return destination.stat().st_size

    def _encode_options(
        self,
        output_format: OutputFormat,
        *,
        quality: int | None,
        source: DecodedImage | None,
        preserve_metadata: bool,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}

        if output_format is OutputFormat.PNG:
            options["compress_level"] = 6
        else:
            options["quality"] = self.validate_quality(quality)
            if output_format is OutputFormat.WEBP:
                options["method"] = 4

        if source is None:
            return options

        # An ICC profile is a colour-space description, not personal data, so
        # it survives stripping: a Display-P3 image that loses it silently
        # becomes sRGB.
        if source.icc_profile is not None:
            options["icc_profile"] = source.icc_profile

        # Stripping drops EXIF wholesale, which necessarily takes the GPS and
        # maker-note tags with it. Keeping "most" metadata would mean deciding
        # which tags are safe, and that list is never complete.
        if preserve_metadata and source.exif is not None:
            options["exif"] = source.exif

        return options

    # ----------------------------------------------------------------- preview

    def write_preview(self, source: Path, destination: Path) -> Path:
        """A resolution-capped JPEG of a result, written atomically.

        Never upscales: a result smaller than the cap is re-encoded at its own
        size rather than blown up, because inventing pixels for a preview would
        misrepresent what the model produced.

        The write goes to a temporary file and is moved into place, so two
        requests racing to build the same preview cannot leave a half-written
        file for a third to serve.
        """
        return self._write_capped_copy(
            source,
            destination,
            max_edge=PREVIEW_MAX_EDGE,
            image_format="JPEG",
            quality=PREVIEW_QUALITY,
        )

    def _write_capped_copy(
        self,
        source: Path,
        destination: Path,
        *,
        max_edge: int,
        image_format: str,
        quality: int,
    ) -> Path:
        """Shared body of the preview and thumbnail writers."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Our own result, so the decompression-bomb guard does not apply -
            # see `open_trusted`. Without this an 8x result (192 MP) cannot be
            # opened at all, and the job has no preview despite completing.
            with open_trusted(source) as opened:
                # draft() lets the JPEG decoder skip straight to a reduced
                # resolution, so a large JPEG result never fully decodes here.
                # It is a no-op for other formats.
                opened.draft("RGB", (max_edge, max_edge))
                image = opened.convert("RGB")
                # thumbnail() only ever shrinks, which is the never-upscale
                # rule enforced by Pillow rather than restated here.
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

                handle, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.stem}.", suffix=".part", dir=destination.parent
                )
                os.close(handle)
                temporary = Path(temporary_name)

                try:
                    image.save(temporary, format=image_format, quality=quality, optimize=True)
                    temporary.replace(destination)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
        # DecompressionBombError inherits straight from Exception, so it is not
        # covered by OSError/ValueError. Listed explicitly: with `open_trusted`
        # it should no longer be reachable here, and if it ever is, it must
        # surface as the application's corrupted-result error rather than
        # escaping as an HTTP 500.
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            raise CorruptedImageError(
                "The result could not be prepared for viewing.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        return destination

    def write_thumbnail(self, source: Path, destination: Path) -> Path:
        """A 256 px WEBP tile for the history grid, written atomically.

        Shares the preview's shape deliberately: same never-upscale rule, same
        temp-then-move write, same lazy cache. A result smaller than the tile
        is stored at its own size rather than blown up to fill the square.
        """
        return self._write_capped_copy(
            source,
            destination,
            max_edge=THUMBNAIL_MAX_EDGE,
            image_format="WEBP",
            quality=THUMBNAIL_QUALITY,
        )

    def crop_to_jpeg(self, source: Path, region: CropRegion) -> bytes:
        """A full-resolution slice of a result, encoded as JPEG.

        Not cached: a crop is specific to where the user happens to be looking,
        and caching every region a pan produces would fill the disk faster than
        the sweeper empties it.
        """
        buffer = io.BytesIO()

        try:
            # Same trusted result as the preview, and the same reason: a crop
            # of a 192 MP 8x output must not be refused as a bomb.
            with open_trusted(source) as opened:
                cropped = opened.convert("RGB").crop(region.box)
                cropped.save(buffer, format="JPEG", quality=PREVIEW_QUALITY)
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            raise CorruptedImageError(
                "That region of the result could not be read.",
                technical=f"{type(exc).__name__}: {exc}",
            ) from exc

        return buffer.getvalue()

    def validate_crop(
        self, x: int, y: int, width: int, height: int, *, bounds: tuple[int, int]
    ) -> CropRegion:
        """Check a requested region against the result it will be taken from.

        Invalid requests are refused rather than clamped. Silently returning a
        different region than the one asked for would put the wrong pixels
        under the viewer's crosshair, which is worse than an error.
        """
        limit_width, limit_height = bounds

        if width <= 0 or height <= 0:
            raise ValidationError(
                "A crop needs a positive width and height.",
                technical=f"w={width} h={height}",
                context={"width": width, "height": height},
            )

        if x < 0 or y < 0 or x + width > limit_width or y + height > limit_height:
            raise ValidationError(
                "That region lies outside the result.",
                technical=f"crop={x},{y},{width},{height} result={limit_width}x{limit_height}",
                context={
                    "resultWidth": limit_width,
                    "resultHeight": limit_height,
                    "requested": {"x": x, "y": y, "width": width, "height": height},
                },
            )

        if width * height > MAX_CROP_PIXELS:
            raise ValidationError(
                f"That region is too large to serve at full resolution. Ask for at most "
                f"{MAX_CROP_PIXELS // 1_000_000} MP.",
                technical=f"requested={width * height} limit={MAX_CROP_PIXELS}",
                context={"limitPixels": MAX_CROP_PIXELS},
            )

        return CropRegion(x=x, y=y, width=width, height=height)

    def validate_quality(self, quality: int | None) -> int:
        """Clamp-free validation: an out-of-range quality is a client error."""
        default = 92
        if quality is None:
            return default

        low, high = JPEG_QUALITY_RANGE
        if not low <= quality <= high:
            raise ValidationError(
                f"Quality must be between {low} and {high}.",
                technical=f"quality={quality}",
                context={"minimum": low, "maximum": high},
            )
        return quality
