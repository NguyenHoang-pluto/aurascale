"""Reference-free image metrics for the quality benchmark.

These describe an image; they do not judge it. There is deliberately no
aggregate "quality score", because the interesting failures all look like an
improvement in one number and a regression in another - sharpening raises edge
strength *and* halo, denoising lowers noise *and* texture. Collapsing that into
one figure would hide exactly what the benchmark exists to show.

Everything is computed on Rec.709 luma, in float64, from **deterministic
sampled tiles** rather than the whole image. A 192 MP output cannot be FFT'd on
a 4 GB machine, and sampling keeps the cost flat regardless of output size. The
seed is fixed, so two runs over the same file sample the same tiles and the
numbers are directly comparable.

No metric here has a ground truth. They are proxies, they are named as proxies,
and a delta between two arms is the only thing any of them is good for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Rec.709 luma. The eye reads detail mostly in luminance, and using it keeps a
# chroma-only change (4:2:0 subsampling, say) from masquerading as detail.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

TILE_SIZE = 512
TILE_COUNT = 16
TILE_SEED = 20260908

# Windows for the local-contrast and flat-region statistics.
CONTRAST_WINDOW = 16
# A window counts as "flat" when its variance is in the lowest fifth of the
# tile. Noise is only meaningful where there is no structure to confuse it.
FLAT_PERCENTILE = 20.0
# Radius, in normalised frequency, above which energy counts as "high".
HIGH_FREQUENCY_CUTOFF = 0.25


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    """What was measured. Every field is a raw quantity, never a verdict."""

    sobel_mean: float
    sobel_p95: float
    local_contrast: float
    high_frequency_ratio: float
    flat_noise: float
    edge_overshoot: float
    tiles_sampled: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_luma(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Rec.709 luma in [0, 1] from an HxWx3 uint8 array."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3, got {rgb.shape}")

    weights = np.array(LUMA_WEIGHTS, dtype=np.float64)
    return (rgb[:, :, :3].astype(np.float64) @ weights) / 255.0


def tile_boxes(
    width: int,
    height: int,
    *,
    count: int = TILE_COUNT,
    size: int = TILE_SIZE,
    seed: int = TILE_SEED,
) -> list[tuple[int, int, int, int]]:
    """Deterministic (left, top, right, bottom) boxes to measure.

    Seeded, so the same image yields the same boxes on every run and two arms
    are compared over the same regions. An image smaller than one tile gives a
    single box covering it, rather than a padded one - padding would invent
    flat area and drag every statistic toward zero.
    """
    if width <= size or height <= size:
        return [(0, 0, width, height)]

    generator = np.random.default_rng(seed)
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        left = int(generator.integers(0, width - size))
        top = int(generator.integers(0, height - size))
        boxes.append((left, top, left + size, top + size))

    return boxes


def sample_tiles(
    luma: np.ndarray[Any, Any],
    *,
    count: int = TILE_COUNT,
    size: int = TILE_SIZE,
    seed: int = TILE_SEED,
) -> list[np.ndarray[Any, Any]]:
    """The same boxes, taken from an array already in memory."""
    height, width = luma.shape[:2]
    boxes = tile_boxes(width, height, count=count, size=size, seed=seed)

    return [luma[top:bottom, left:right] for left, top, right, bottom in boxes]


def _gradient_magnitude(tile: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Sobel |grad|, computed directly so the harness needs no OpenCV."""
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = kernel_x.T

    gx = _convolve3(tile, kernel_x)
    gy = _convolve3(tile, kernel_y)
    magnitude: np.ndarray[Any, Any] = np.hypot(gx, gy)
    return magnitude


def _convolve3(tile: np.ndarray[Any, Any], kernel: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Valid-region 3x3 convolution by shifted views.

    No SciPy dependency, and no edge padding: padding would manufacture an
    artificial gradient at the tile border and inflate every edge statistic.
    """
    out = np.zeros((tile.shape[0] - 2, tile.shape[1] - 2), dtype=np.float64)

    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight != 0.0:
                out += weight * tile[dy : dy + out.shape[0], dx : dx + out.shape[1]]

    return out


def _box_blur3(tile: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A 3x3 mean, used as the smooth reference for the overshoot proxy."""
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    return _convolve3(tile, kernel)


def sobel_statistics(tile: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Mean and 95th-percentile gradient magnitude.

    The pair matters more than either alone: the mean moves with overall
    acutance, the 95th percentile with how hard the strongest edges are driven.
    Sharpening raises the second much faster than the first.
    """
    magnitude = _gradient_magnitude(tile)
    if magnitude.size == 0:
        return 0.0, 0.0

    return float(magnitude.mean()), float(np.percentile(magnitude, 95))


def local_contrast(tile: np.ndarray[Any, Any], window: int = CONTRAST_WINDOW) -> float:
    """Mean per-window standard deviation - micro-detail, not global contrast."""
    height, width = tile.shape[:2]
    rows, columns = height // window, width // window
    if rows == 0 or columns == 0:
        return float(tile.std())

    blocks = tile[: rows * window, : columns * window]
    blocks = blocks.reshape(rows, window, columns, window).swapaxes(1, 2)
    return float(blocks.reshape(rows * columns, -1).std(axis=1).mean())


def high_frequency_ratio(
    tile: np.ndarray[Any, Any], cutoff: float = HIGH_FREQUENCY_CUTOFF
) -> float:
    """Share of spectral energy above `cutoff` of Nyquist.

    Reads as "how much fine structure is present". Upscaling without adding
    detail lowers it: the same structure now spans more pixels, so its energy
    moves down the spectrum. That is precisely the "larger but not sharper"
    complaint, expressed as a number.
    """
    if tile.size == 0:
        return 0.0

    # Windowed, or the tile edges ring and put energy in every band.
    window = np.hanning(tile.shape[0])[:, None] * np.hanning(tile.shape[1])[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(tile * window))) ** 2

    height, width = spectrum.shape
    cy, cx = height / 2.0, width / 2.0
    yy = (np.arange(height) - cy) / max(cy, 1.0)
    xx = (np.arange(width) - cx) / max(cx, 1.0)
    radius = np.hypot(yy[:, None], xx[None, :])

    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0

    return float(spectrum[radius > cutoff].sum() / total)


def flat_region_noise(tile: np.ndarray[Any, Any], window: int = CONTRAST_WINDOW) -> float:
    """Median absolute deviation inside the flattest windows.

    Measured only where there is no structure, so real texture is not counted
    as noise. Denoising should lower this; over-denoising lowers it while also
    lowering `local_contrast` and `high_frequency_ratio`, which is how the
    benchmark tells the two apart.
    """
    height, width = tile.shape[:2]
    rows, columns = height // window, width // window
    if rows == 0 or columns == 0:
        return float(np.median(np.abs(tile - np.median(tile))))

    blocks = tile[: rows * window, : columns * window]
    blocks = blocks.reshape(rows, window, columns, window).swapaxes(1, 2)
    flat = blocks.reshape(rows * columns, -1)

    variance = flat.var(axis=1)
    threshold = np.percentile(variance, FLAT_PERCENTILE)
    selected = flat[variance <= threshold]
    if selected.size == 0:
        return 0.0

    centred = selected - np.median(selected, axis=1, keepdims=True)
    return float(np.median(np.abs(centred)))


def edge_overshoot(tile: np.ndarray[Any, Any]) -> float:
    """Halo proxy: excursion beyond a smoothed reference, next to strong edges.

    Unsharp masking leaves a bright rim on one side of an edge and a dark one
    on the other. With no ground truth available the smoothed tile stands in
    for "the signal without the rim", and this reports how far the image
    departs from it in the band around strong edges.

    A proxy, and named as one. It is comparable between two arms over the same
    image; it is not an absolute measure of anything.
    """
    magnitude = _gradient_magnitude(tile)
    if magnitude.size == 0:
        return 0.0

    smoothed = _box_blur3(tile)
    centre = tile[1:-1, 1:-1]

    strong = magnitude >= np.percentile(magnitude, 90)
    if not strong.any():
        return 0.0

    return float(np.abs(centre[strong] - smoothed[strong]).mean())


def measure(rgb: np.ndarray[Any, Any], *, seed: int = TILE_SEED) -> ImageMetrics:
    """Every metric, averaged over the sampled tiles."""
    return _reduce(sample_tiles(to_luma(rgb), seed=seed))


def _reduce(tiles: list[np.ndarray[Any, Any]]) -> ImageMetrics:
    """Average each metric over the tiles that were sampled."""
    sobel_means: list[float] = []
    sobel_p95s: list[float] = []
    contrasts: list[float] = []
    ratios: list[float] = []
    noises: list[float] = []
    overshoots: list[float] = []

    for tile in tiles:
        mean, p95 = sobel_statistics(tile)
        sobel_means.append(mean)
        sobel_p95s.append(p95)
        contrasts.append(local_contrast(tile))
        ratios.append(high_frequency_ratio(tile))
        noises.append(flat_region_noise(tile))
        overshoots.append(edge_overshoot(tile))

    return ImageMetrics(
        sobel_mean=float(np.mean(sobel_means)),
        sobel_p95=float(np.mean(sobel_p95s)),
        local_contrast=float(np.mean(contrasts)),
        high_frequency_ratio=float(np.mean(ratios)),
        flat_noise=float(np.mean(noises)),
        edge_overshoot=float(np.mean(overshoots)),
        tiles_sampled=len(tiles),
    )


def measure_file(path: Path, *, seed: int = TILE_SEED) -> ImageMetrics:
    """Measure a result file at its own full resolution.

    Read through `open_trusted`: an 8x result legitimately exceeds Pillow's
    decompression-bomb limit, and this is our own output.

    Tiles are cropped from the open image rather than converting the whole
    thing to an array first. Pillow still decodes the source once, but a 192 MP
    result does not additionally become a 549 MiB numpy copy - which on a
    machine that already failed once on a 558 MiB allocation is the difference
    between measuring and crashing.
    """
    from app.services.image_service import open_trusted

    with open_trusted(path) as opened:
        width, height = opened.size
        tiles = [
            to_luma(np.asarray(opened.crop(box).convert("RGB"), dtype=np.uint8))
            for box in tile_boxes(width, height, seed=seed)
        ]

    return _reduce(tiles)
