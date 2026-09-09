"""Benchmark helpers shared by more than one research phase.

Everything here was first written inside a phase script - `phase3a`, `phase3b`,
`phase3c` - because that is where it was first needed. None of it is specific to
those phases: reading a corpus image, blocking a channel, estimating noise sigma
and measuring chroma are generic, and a later phase that needs them should not
have to import from a research script that happens to predate it.

That is exactly what went wrong. Phase 4 F2 was committed importing `_load` from
`phase3a`, three chroma helpers from `phase3b` and a crop table from `phase3c`,
none of which are tracked - so a clean checkout could not run it. This module is
the tracked home for those helpers.

What is deliberately **not** here
---------------------------------

`metrics.py` keeps the luma statistics. It is shared with four earlier reports
and moving anything out of it would silently move their numbers.

`prefilter.py` keeps the Phase 3A candidate filters and the noise *estimator*
built on these primitives. Only the primitives move; the estimator, its
calibrated anchors and the refuted mapping stay where their evidence is.

The phase scripts themselves are unchanged. They still carry their own copies,
which is harmless while they are untracked and produced the reports they did; if
any of them is ever committed it should import from here instead.

Numerical equivalence with the originals was verified before the move: the
functions below reproduce `phase3b.chroma_metrics` exactly on the corpus, so
the numbers already published in `phase4_f2_denoise_report.md` still stand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.corpus import CorpusImage
from benchmarks.metrics import tile_boxes

# ------------------------------------------------------------------- reading


def load_rgb(image: CorpusImage) -> np.ndarray[Any, Any]:
    """One corpus image as an HxWx3 uint8 array.

    Was `phase3a._load`. A private name imported across three modules is a sign
    it was never really private.
    """
    from PIL import Image

    return np.asarray(Image.open(image.path).convert("RGB"), dtype=np.uint8)


# -------------------------------------------------------- block statistics
#
# Was `prefilter._blocks_of` / `_immerkaer_sigma`. Generic: neither knows
# anything about denoising, prefilters or any particular phase.

#: Blocks this size are scored for flatness. Large enough to hold noise
#: statistics, small enough that a 2 MP image yields thousands of them.
NOISE_BLOCK = 32

#: Only the flattest share of blocks is measured. This is the whole trick: a
#: photograph's texture lives in high-variance blocks, so restricting to the
#: low-variance tail is what keeps foliage from reading as noise. 10 % is
#: deliberately tighter than `metrics.FLAT_PERCENTILE` (20 %), because an
#: estimator that decides how hard to filter should err toward *under*
#: estimating.
NOISE_FLAT_PERCENTILE = 10.0

#: Immerkaer's Laplacian mask. Convolving with it annihilates locally linear
#: signal - which is what a smooth patch is - and leaves noise, so the mean
#: absolute response over flat blocks estimates the noise sigma. The 6 and the
#: sqrt(pi/2) are the normalisation from the original derivation.
IMMERKAER = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


def blocks_of(channel: np.ndarray[Any, Any], size: int = NOISE_BLOCK) -> np.ndarray[Any, Any]:
    """Reshape a channel into non-overlapping size x size blocks."""
    height, width = channel.shape[:2]
    rows, columns = height // size, width // size
    if rows == 0 or columns == 0:
        return channel.reshape(1, *channel.shape)

    trimmed = channel[: rows * size, : columns * size]
    blocks = trimmed.reshape(rows, size, columns, size).swapaxes(1, 2)
    return blocks.reshape(rows * columns, size, size)


def immerkaer_sigma(blocks: np.ndarray[Any, Any]) -> float:
    """Noise sigma over a stack of blocks, by the Laplacian-response method.

    Computed per block and then taken as a **median** across blocks rather than
    a mean. A median is what makes the estimator robust: a handful of blocks
    that are flat by variance but contain a hard edge would drag a mean upward,
    and there is always a handful.
    """
    if blocks.size == 0:
        return 0.0

    # Valid-region 3x3 convolution across the whole stack at once.
    response = np.zeros((blocks.shape[0], blocks.shape[1] - 2, blocks.shape[2] - 2))
    for dy in range(3):
        for dx in range(3):
            weight = IMMERKAER[dy, dx]
            if weight != 0.0:
                response += (
                    weight * blocks[:, dy : dy + response.shape[1], dx : dx + response.shape[2]]
                )

    if response.shape[1] == 0 or response.shape[2] == 0:
        return 0.0

    # sqrt(pi/2) / 6 converts the mean absolute Laplacian response to sigma.
    per_block = np.sqrt(np.pi / 2.0) / 6.0 * np.abs(response).mean(axis=(1, 2))
    return float(np.median(per_block))


# --------------------------------------------------------------- chroma
#
# Was `phase3b`. `metrics.py` is entirely Rec.709 luma and is left that way -
# it is shared with four earlier reports. What it cannot see lives here, and is
# sampled over the same `metrics.tile_boxes` so both families read identical
# regions.


def ycrcb_tiles(rgb: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    """The 16 deterministic tiles of an image, as float64 YCrCb.

    The same boxes `metrics.measure` uses, from the same seed, so the luma and
    chroma statistics describe identical regions. Sampling rather than whole-
    frame also keeps the cost flat: a 4x output is 32 MP and blocking three
    full channels of it costs more than the SR pass that produced it.
    """
    import cv2

    height, width = rgb.shape[:2]
    return [
        cv2.cvtColor(
            np.ascontiguousarray(rgb[top:bottom, left:right, :3]), cv2.COLOR_RGB2YCrCb
        ).astype(np.float64)
        for left, top, right, bottom in tile_boxes(width, height)
    ]


def flat_mask(tiles: list[np.ndarray[Any, Any]], percentile: float = NOISE_FLAT_PERCENTILE) -> Any:
    """Which 32 px blocks across the sampled tiles are flat, by luma variance.

    Computed once from a baseline arm and then reused for every other arm of
    that image. This is the Phase 3A correction: a filter that changes luma
    changes which blocks are flattest, so a mask recomputed per arm silently
    compares different regions.
    """
    variance = np.concatenate([blocks_of(tile[:, :, 0]).var(axis=(1, 2)) for tile in tiles])
    return variance <= np.percentile(variance, percentile)


@dataclass(frozen=True, slots=True)
class ChromaMetrics:
    """Colour-channel statistics the luma metrics are structurally blind to."""

    #: Noise sigma in Y over the shared mask. A chroma-only change must leave
    #: this untouched - it is the control.
    sigma_luma: float
    #: The same in Cr/Cb, averaged.
    sigma_chroma: float
    #: Mean Sobel magnitude on Cr/Cb. Colour-boundary preservation: chroma
    #: smearing and colour bleeding show up here as a fall, and it is what
    #: stops "less chroma noise" from being read as "better".
    chroma_gradient: float
    #: The strongest colour edges specifically.
    chroma_gradient_p95: float
    blocks_used: int
    blocks_total: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sobel_magnitude(channel: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Sobel |grad| on one channel, valid region only."""
    import cv2

    gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    magnitude: np.ndarray[Any, Any] = np.hypot(gx, gy)[1:-1, 1:-1]
    return magnitude


def chroma_metrics(tiles: list[np.ndarray[Any, Any]], mask: Any) -> ChromaMetrics:
    """Chroma statistics over a *given* mask, so arms are directly comparable."""
    luma_blocks = np.concatenate([blocks_of(tile[:, :, 0]) for tile in tiles])
    cr_blocks = np.concatenate([blocks_of(tile[:, :, 1]) for tile in tiles])
    cb_blocks = np.concatenate([blocks_of(tile[:, :, 2]) for tile in tiles])

    gradient = np.concatenate(
        [sobel_magnitude(tile[:, :, channel]).ravel() for tile in tiles for channel in (1, 2)]
    )

    return ChromaMetrics(
        sigma_luma=immerkaer_sigma(luma_blocks[mask]),
        sigma_chroma=(immerkaer_sigma(cr_blocks[mask]) + immerkaer_sigma(cb_blocks[mask])) / 2.0,
        chroma_gradient=float(gradient.mean()),
        chroma_gradient_p95=float(np.percentile(gradient, 95)),
        blocks_used=int(np.count_nonzero(mask)),
        blocks_total=int(mask.size),
    )


# -------------------------------------------------------------- crop regions

#: Where to crop for visual inspection, in **source** pixel coordinates, as
#: (centre_x, centre_y, size). Was `phase3c.REGIONS`.
#:
#: Verified by eye against the sources before first use: the portrait regions
#: land on the iris, the sclera, the lip surface, individual hair strands and
#: open cheek skin respectively. The portrait carries six of the ten because
#: that is where the product's detail complaint lives.
#:
#: Tied to this specific corpus - the coordinates only mean anything for the
#: images `corpus/manifest.json` names.
CROP_REGIONS: dict[str, dict[str, tuple[int, int, int]]] = {
    "portrait-skin": {
        "eye-left": (315, 510, 140),
        "eye-right": (929, 510, 140),
        "hair": (180, 120, 160),
        "lips": (645, 1185, 160),
        "skin-cheek": (1019, 900, 160),
        "identity-face": (620, 760, 640),
    },
    "landscape-detail": {"fine-texture": (760, 420, 180)},
    "foliage-texture": {"saturated-foliage": (500, 760, 180)},
    "text-signage": {"text": (600, 300, 180)},
    "low-light-noise": {"dark-noisy": (900, 450, 180)},
}
