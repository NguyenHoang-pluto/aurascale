"""Phase 4 F4: does the inference tile size change image quality?

    python -m benchmarks.phase4_f4 --determinism  # is inference repeatable at all?
    python -m benchmarks.phase4_f4 --sweep        # 5 photographs x 3 tile sizes
    python -m benchmarks.phase4_f4 --strips       # blind, shuffled comparison strips

Research only. Nothing here changes production behaviour: the configured tile
stays 256, `tile_pad` stays 16, `DEFAULT_DENOISE` stays 1.0, `CREATIVE_DENOISE`
stays 0.25, the sharpening default stays 0.0 and the out-of-memory ladder is
untouched. This module reads the production implementation and never writes to
it.

The question
------------

Every memory decision in the pipeline - the VRAM ladder in `device.py`, the OOM
retry in `upscaler.py`, the operator's `tile_size` setting - assumes a smaller
tile is a *performance* choice. If that assumption is wrong, then a laptop that
quietly drops to tile 128 under memory pressure is shipping a different picture
than a desktop, and nobody has ever checked.

So: is tile 128 the same picture as tile 256, only slower?

Why it might not be
-------------------

From `tiler.py`, the geometry is unusually strict. Each tile is given
`tile_pad = 16` pixels of real neighbouring context, runs through the network,
and then that padding is cropped back off at `pad * scale`. What lands in the
canvas is only the tile's own rectangle, and `_run` says so explicitly: "no two
tiles ever write the same pixel and nothing is blended or accumulated here".

That is a **butt joint**. There is no feathering, no overlap average, no
gradient blend. Continuity across a tile boundary is not enforced by the
stitcher at all - it is left entirely to the 16 pixels of context, on the
assumption that a convolutional network's receptive field is narrow enough that
16 source pixels is all the context a boundary pixel can see.

If that assumption holds, the output is *independent of tile size* and the whole
question is settled. If it does not - a 23-block RRDBNet has a receptive field
considerably wider than 16 - then every tile boundary is a place where the
network saw different context than it would have seen untiled, and the smaller
the tile the more such places there are.

Where the seams are
-------------------

Derived from the implementation, not assumed. `TileGrid.tiles()` lays tiles at
`x = column * tile_size` in **source** coordinates, and `output_slice(scale)`
writes them at `x * scale`. So an internal seam falls at every multiple of
`tile_size * scale` in output coordinates: 1024 px at tile 256, 512 px at tile
128, 256 px at tile 64. `tile_pad` shifts what the network *reads*; it does not
move where the joint *is*, because the pad is cropped away again.

The control that makes this measurable
--------------------------------------

Those three grids nest, and that is the whole experimental design. Split them
into exclusive sets:

===================  =========  =========  ========
output coordinate    tile 256   tile 128   tile 64
===================  =========  =========  ========
multiple of 1024     seam       seam       seam
512 but not 1024     *content*  seam       seam
256 but not 512      *content*  *content*  seam
===================  =========  =========  ========

The starred cells are the point. A column at, say, x = 1536 is a real tile joint
for tile 128 and tile 64, and is nothing at all for tile 256 - but it is the
same column of the same photograph in all three. So the image content underneath
is held exactly constant, and any excess discontinuity that appears there in the
finer arms and not in tile 256 is caused by tiling and by nothing else.

Without that control a "seam metric" is uninterpretable: photographs contain
strong vertical edges, and some of them land on multiples of 512 by luck.

Determinism first
-----------------

None of the above means anything until repeatability is established. `--sweep`
refuses to run until `--determinism` has recorded whether two identical passes
produce identical bytes, because a difference between tile 256 and tile 128 is
only evidence about tiling if tile 256 against *itself* is zero. cuDNN is free
to pick different convolution algorithms between runs, and fp16 accumulation is
not associative, so this is a real possibility rather than a formality.

What is deliberately not varied
-------------------------------

`RealESRGAN_x4plus` only, 4x, PNG, sharpening 0.0, identical input bytes,
identical weights, identical dtype. `x4plus` also has no `denoise_pair`, so
`_denoise_for` drops any denoise value and there is no denoising stage at all -
denoise is not held constant here, it is structurally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.corpus import DEFAULT_CORPUS_DIR, CorpusImage, load_corpus, load_manifest
from benchmarks.metrics import ImageMetrics, measure, to_luma
from benchmarks.research_utils import (
    CROP_REGIONS,
    chroma_metrics,
    flat_mask,
    load_rgb,
    ycrcb_tiles,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase4_f4_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase4-f4-crops"

MODEL = "RealESRGAN_x4plus"
SCALE = 4

#: The arms, largest first. 256 is the production default and therefore the
#: reference; 128 is what the VRAM ladder drops to on a small card; 64 is well
#: past anything production would choose, included as the stress case - if
#: tiling has an effect at all it must be visible here.
TILES: tuple[int, ...] = (256, 128, 64)
REFERENCE_TILE = TILES[0]

#: The image the repeatability control runs on. Low-light noise is the hardest
#: case for reproducibility: it has the least deterministic structure for the
#: network to lock onto, so if any arm drifts between runs it is this one.
DETERMINISM_CATEGORY = "low-light-noise"
DETERMINISM_REPEATS = 2

#: Boundaries either side of a seam that form its local baseline. 24 is chosen
#: to stay well inside the 256 px spacing of the densest grid, so a seam's
#: baseline can never include another seam.
SEAM_WINDOW = 24

#: Rows compared at a time when diffing two full-resolution arms, so a 32 MP
#: comparison does not need a 32 MP int16 temporary.
DIFF_CHUNK_ROWS = 512

#: One inspection region per photograph, at 1:1 on the 4x output. Deliberately
#: not the full `CROP_REGIONS` set: F2 and F3 needed many regions because they
#: changed the whole frame, whereas tiling - if it does anything - does it at
#: known coordinates, and the seam crops below go there directly.
DETAIL_REGIONS: dict[str, str] = {
    "portrait-skin": "skin-cheek",
    "landscape-detail": "fine-texture",
    "foliage-texture": "saturated-foliage",
    "text-signage": "text",
    "low-light-noise": "dark-noisy",
}

#: A seam crop is taller than it is wide: the joint is a vertical line, so the
#: crop needs enough length along it to judge continuity and only enough width
#: either side for context.
SEAM_CROP_W = 384
SEAM_CROP_H = 512

#: The grids the seam crops are placed on. `t128` is a joint for tile 128 and
#: tile 64 and plain content for tile 256; `t064` is a joint for tile 64 alone.
#: Between them the two crops carry both verdicts the phase has to reach.
SEAM_GRIDS = ("t128", "t064")

#: Texture crops are visual-only, so they are JPEG. Seam crops are **not**:
#: JPEG's 8x8 transform blocks land on multiples of 8, every seam coordinate
#: here is a multiple of 256, and block edges appearing exactly on the joint
#: under test is the one compression artifact that could be mistaken for the
#: effect being measured.
DETAIL_QUALITY = 95


def arm(tile: int) -> str:
    return f"t{tile:03d}"


def digest(array: np.ndarray[Any, Any]) -> str:
    """SHA-256 of the raw pixels of one arm.

    Hashed on the array rather than on an encoded PNG, so the answer is about
    the inference result and not about the encoder's choices.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ------------------------------------------------------------ seam geometry


def seam_boundaries(extent: int, tile: int, scale: int = SCALE) -> list[int]:
    """Indices into an edge profile where tiling puts a joint.

    A profile entry `i` is the boundary between output samples `i` and `i + 1`,
    so the joint before output coordinate `c` is entry `c - 1`.
    """
    step = tile * scale
    return [c - 1 for c in range(step, extent, step)]


def exclusive_grids(extent: int, tiles: tuple[int, ...] = TILES) -> dict[str, list[int]]:
    """The seam grids of every arm, split so no boundary appears twice.

    Keyed by the coarsest arm that owns the boundary. `t256` is the multiples of
    1024, `t128` the multiples of 512 that are not multiples of 1024, `t064` the
    multiples of 256 that are neither. Each set is a real seam for its own arm
    and every finer arm, and plain image content for every coarser one.
    """
    grids: dict[str, list[int]] = {}
    claimed: set[int] = set()
    for tile in sorted(tiles, reverse=True):
        boundaries = [b for b in seam_boundaries(extent, tile) if b not in claimed]
        grids[arm(tile)] = boundaries
        claimed.update(boundaries)
    return grids


def edge_profiles(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Mean absolute luma step across every column boundary, and every row one.

    Reduced to two 1-D profiles before any seam arithmetic happens. A tile joint
    is a full-height discontinuity, so averaging down the whole column is the
    matched filter for it: real image edges are local and average away, while a
    seam - which by construction runs the entire height of the tile row - does
    not.
    """
    # In 8-bit levels rather than the [0, 1] `to_luma` returns, so `step_mean`
    # and `baseline_mean` can be read directly as levels out of 255. The
    # z-scores are ratios and are unaffected either way.
    luma = to_luma(rgb) * LUMA_FULL_SCALE
    columns: np.ndarray[Any, Any] = np.abs(np.diff(luma, axis=1)).mean(axis=0)
    rows: np.ndarray[Any, Any] = np.abs(np.diff(luma, axis=0)).mean(axis=1)
    return columns, rows


@dataclass(frozen=True, slots=True)
class SeamStats:
    """How far the joints of one grid stand out from their surroundings."""

    #: Boundaries scored. Zero means this grid does not exist at this size.
    count: int
    #: Median robust z-score of the step at a seam against its neighbourhood.
    #: This is the headline: 0 means a seam is indistinguishable from ordinary
    #: image structure, and a large positive value means it is visible to the
    #: measurement whether or not it is visible to the eye.
    z_median: float
    #: The worst single joint, which is what a reader would notice first.
    z_max: float
    #: Share of joints standing more than 3 robust deviations above local.
    fraction_above_3: float
    #: Mean step at the seams, and the median local baseline they are measured
    #: against, both in luma units - so the z-scores can be read back into
    #: something physical.
    step_mean: float
    baseline_mean: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EMPTY_SEAM = SeamStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def score_grid(profile: np.ndarray[Any, Any], boundaries: list[int]) -> SeamStats:
    """Robust z-score of each boundary against the profile around it.

    Median and MAD rather than mean and standard deviation: the neighbourhood of
    a seam in a photograph routinely contains a genuine edge, and one edge is
    enough to inflate a standard deviation to the point where nothing looks
    significant.
    """
    if not boundaries:
        return EMPTY_SEAM

    zs: list[float] = []
    steps: list[float] = []
    baselines: list[float] = []

    for index in boundaries:
        low = max(0, index - SEAM_WINDOW)
        high = min(profile.size, index + SEAM_WINDOW + 1)
        neighbourhood = np.delete(profile[low:high], index - low)
        if neighbourhood.size < SEAM_WINDOW:
            continue

        median = float(np.median(neighbourhood))
        mad = float(np.median(np.abs(neighbourhood - median))) * 1.4826
        step = float(profile[index])

        steps.append(step)
        baselines.append(median)
        # A neighbourhood flat to the last bit gives mad == 0. Falling back to
        # the median keeps the ratio finite instead of reporting an infinity
        # that would dominate every aggregate downstream.
        zs.append((step - median) / mad if mad > 0.0 else (step - median) / max(median, 1e-6))

    if not zs:
        return EMPTY_SEAM

    array = np.asarray(zs)
    return SeamStats(
        count=array.size,
        z_median=float(np.median(array)),
        z_max=float(array.max()),
        fraction_above_3=float((array > 3.0).mean()),
        step_mean=float(np.mean(steps)),
        baseline_mean=float(np.mean(baselines)),
    )


def seam_report(rgb: np.ndarray[Any, Any]) -> dict[str, dict[str, Any]]:
    """Score one arm's output against all three exclusive grids, both axes.

    Every arm is scored on every grid, including the grids that are not its own
    seams. Those are the control: they measure what the metric reads on ordinary
    image content at coordinates that merely look like seam coordinates.
    """
    columns, rows = edge_profiles(rgb)
    height, width = rgb.shape[:2]

    report: dict[str, dict[str, Any]] = {}
    for axis, profile, extent in (("vertical", columns, width), ("horizontal", rows, height)):
        grids = exclusive_grids(extent)
        report[axis] = {
            name: score_grid(profile, boundaries).as_dict() for name, boundaries in grids.items()
        }
    return report


def worst_vertical_seam(profile: np.ndarray[Any, Any], extent: int, grid: str) -> tuple[int, float]:
    """Output column of the most conspicuous joint in one grid.

    Worst rather than typical on purpose: the visual question is "can a reader
    see it", and that is settled by the most conspicuous instance, not by the
    median one. Picking the crop by measurement also removes the obvious way to
    cheat at this - choosing a flat region where no stitcher could fail.
    """
    boundaries = exclusive_grids(extent).get(grid, [])
    best_index, best_z = -1, -np.inf
    for index in boundaries:
        stats = score_grid(profile, [index])
        if stats.count and stats.z_max > best_z:
            best_index, best_z = index, stats.z_max
    return best_index + 1, float(best_z)


def busiest_band(luma: np.ndarray[Any, Any], column: int, height: int) -> int:
    """Top row of the `height`-tall band where a seam column crosses most detail.

    A joint in a clear sky is invisible whatever the stitcher does, so the crop
    is placed where the seam runs through structure. Scored on the absolute
    luma step across the joint itself, summed down a sliding window, so the band
    chosen is simultaneously the busiest and the most discontinuous.
    """
    left = max(0, column - 1)
    step = np.abs(luma[:, min(column, luma.shape[1] - 1)] - luma[:, left])
    if step.size <= height:
        return 0

    window = np.convolve(step, np.ones(height), mode="valid")
    return int(np.argmax(window))


# ------------------------------------------------------------------- SSIM

#: Wang et al.'s stabilising constants for 8-bit data.
SSIM_C1 = (0.01 * 255.0) ** 2
SSIM_C2 = (0.03 * 255.0) ** 2
SSIM_SIGMA = 1.5
SSIM_WINDOW = 11

#: `metrics.to_luma` returns Rec.709 luma in **[0, 1]**, and SSIM's stabilising
#: constants are defined against the dynamic range. Feeding [0, 1] data to
#: constants sized for [0, 255] makes the denominator almost entirely constant
#: and pins the score at ~0.9999 no matter how different the inputs are - which
#: is exactly the answer this phase is trying to test, so it would have been an
#: easy result to believe. Luma is put back on the 8-bit scale here instead.
LUMA_FULL_SCALE = 255.0


def ssim_sampled(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    """Mean SSIM between two arms, over the 16 tiles the luma metrics sample.

    Built from the `cv2` Gaussian blur the project already depends on, not from
    scikit-image, which is not installed here: adding a heavyweight dependency
    for one metric was explicitly out of scope, and the Wang et al. formulation
    is four blurs and some arithmetic.

    Sampled on `metrics.tile_boxes` rather than the full frame for two reasons:
    it describes exactly the regions every other number in the table describes,
    and a full-frame pass would need several 32 MP float64 temporaries per
    comparison.

    SSIM is reported because it is the metric that separates "different bytes"
    from "different picture". Neither arm is ground truth, so this is a symmetric
    agreement score, not an accuracy score.
    """
    import cv2

    from benchmarks.metrics import tile_boxes

    height, width = left.shape[:2]
    scores: list[float] = []

    for box in tile_boxes(width, height):
        box_left, top, right_edge, bottom = box
        a = to_luma(left[top:bottom, box_left:right_edge]) * LUMA_FULL_SCALE
        b = to_luma(right[top:bottom, box_left:right_edge]) * LUMA_FULL_SCALE

        kernel = (SSIM_WINDOW, SSIM_WINDOW)
        mu_a = cv2.GaussianBlur(a, kernel, SSIM_SIGMA)
        mu_b = cv2.GaussianBlur(b, kernel, SSIM_SIGMA)
        mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

        sigma_aa = cv2.GaussianBlur(a * a, kernel, SSIM_SIGMA) - mu_aa
        sigma_bb = cv2.GaussianBlur(b * b, kernel, SSIM_SIGMA) - mu_bb
        sigma_ab = cv2.GaussianBlur(a * b, kernel, SSIM_SIGMA) - mu_ab

        numerator = (2.0 * mu_ab + SSIM_C1) * (2.0 * sigma_ab + SSIM_C2)
        denominator = (mu_aa + mu_bb + SSIM_C1) * (sigma_aa + sigma_bb + SSIM_C2)
        scores.append(float(np.mean(numerator / denominator)))

    return float(np.mean(scores)) if scores else 1.0


# ---------------------------------------------------------- array comparison


@dataclass(frozen=True, slots=True)
class Difference:
    """How two full-resolution arms differ, exactly."""

    identical: bool
    max_abs: int
    mean_abs: float
    #: Share of subpixels that differ at all. The one number that separates
    #: "a handful of pixels moved by 1" from "the whole frame is different".
    fraction_differing: float
    #: Root mean square difference, and the PSNR it implies. PSNR is reported
    #: because it is the number a reader will expect, not because it is the
    #: right tool: it is computed here between two arms, neither of which is
    #: ground truth.
    rmse: float
    psnr_db: float
    #: Mean SSIM over the sampled tiles. 1.0 is pixel agreement; this is the
    #: number that decides whether a nonzero byte difference is a *picture*
    #: difference.
    ssim: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> Difference:
    """Exact uint8 comparison, in row chunks so nothing large is materialised."""
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")

    total = left.size
    differing = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum = 0

    for start in range(0, left.shape[0], DIFF_CHUNK_ROWS):
        stop = start + DIFF_CHUNK_ROWS
        delta = left[start:stop].astype(np.int16) - right[start:stop].astype(np.int16)
        np.abs(delta, out=delta)
        differing += int(np.count_nonzero(delta))
        absolute_sum += float(delta.sum())
        squared_sum += float(np.square(delta, dtype=np.float64).sum())
        maximum = max(maximum, int(delta.max()))

    rmse = float(np.sqrt(squared_sum / total))
    identical = differing == 0
    return Difference(
        identical=identical,
        max_abs=maximum,
        mean_abs=absolute_sum / total,
        fraction_differing=differing / total,
        rmse=rmse,
        psnr_db=float("inf") if rmse == 0.0 else float(20.0 * np.log10(255.0 / rmse)),
        # Skipped when the arrays are equal: SSIM of an array with itself is
        # 1.0 by construction and the blurs are not free.
        ssim=1.0 if identical else ssim_sampled(left, right),
    )


# ------------------------------------------------------------- the SR passes


@dataclass(frozen=True, slots=True)
class PassRecord:
    """Provenance of one neural pass, as the production path reported it."""

    requested_tile: int
    effective_tile: int
    tiles: int
    tile_reduced: bool
    fell_back_to_cpu: bool
    device: str
    fp16: bool
    seconds: float
    peak_vram_mb: float
    #: What `_starting_tile` had to work with. Recorded because it, not the
    #: image, is what decides whether the requested tile is granted.
    free_vram_before_mb: int | None
    #: False when the production path silently chose a different tile than the
    #: one asked for - which `_starting_tile` will do if VRAM is tight. Such a
    #: cell is not the arm it claims to be and must not be compared.
    honoured: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_manager() -> Any:
    from app.core.config import Settings
    from app.inference.model_manager import ModelManager
    from app.services.model_service import ModelService

    settings = Settings()
    return ModelManager(settings, ModelService(settings))


def run_pass(manager: Any, source: np.ndarray[Any, Any], tile: int) -> tuple[Any, PassRecord]:
    """One production upscale at a requested tile size.

    Goes through `RealEsrganUpscaler.upscale` exactly as `enhancement_service`
    does, tile and pad included, so the OOM ladder, the VRAM clamp and the
    stitcher are the real ones. `tile_pad` stays at the production 16 for every
    arm: the experiment varies tile size, and varying the pad with it would
    confound the two.
    """
    import torch

    from app.core.config import Settings
    from app.inference.device import free_vram_mb, release_cuda_memory

    settings = Settings()
    upscaler = manager.get(MODEL, denoise_strength=None)

    # Returns the allocator's cached blocks to the driver before `_starting_tile`
    # samples free VRAM. This is not cosmetic. The first run of this control
    # exposed why: pass 1 was granted tile 256, and pass 2 - same image, same
    # request - was silently clamped to 128, because the caching allocator was
    # still holding pass 1's blocks and `free_vram_mb` cannot see them as free.
    # Two cells that asked for the same tile then ran at different tile sizes,
    # which would have been recorded as a repeatability failure.
    #
    # `release_cuda_memory` is production's own helper, called here rather than
    # changed: nothing in `app/` is modified by this benchmark. It puts the card
    # in the state a freshly started process would see, so every cell competes
    # for the tile it asked for on equal terms.
    release_cuda_memory()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    free_before = free_vram_mb(upscaler.target.index) if torch.cuda.is_available() else None

    started = time.perf_counter()
    output = upscaler.upscale(source, tile=tile, tile_pad=settings.tile_pad)
    seconds = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    report = upscaler.last_report
    assert report is not None

    record = PassRecord(
        requested_tile=tile,
        effective_tile=report.tile_size,
        tiles=report.tiles,
        tile_reduced=report.tile_size_reduced,
        fell_back_to_cpu=report.fell_back_to_cpu,
        device=report.device,
        fp16=report.fp16,
        seconds=seconds,
        peak_vram_mb=peak,
        free_vram_before_mb=free_before,
        honoured=report.tile_size == tile and not report.fell_back_to_cpu,
    )
    return output, record


# --------------------------------------------------------- the determinism run


def run_determinism(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Is a pass repeatable at all?

    Runs before anything else and answers the only question that makes the rest
    interpretable. Every arm is repeated, not just the reference: a tiled pass
    and an untiled-ish one exercise different kernel shapes, and cuDNN chooses
    algorithms per shape, so repeatability at tile 256 would not by itself imply
    repeatability at tile 64.
    """
    chosen = [image for image in images if image.category == DETERMINISM_CATEGORY]
    if not chosen:
        raise SystemExit(f"corpus has no {DETERMINISM_CATEGORY} image")

    image = chosen[0]
    source = load_rgb(image)
    manager = _make_manager()
    rows: list[dict[str, Any]] = []

    print(f"\n  repeatability control on {image.category} ({source.shape[1]}x{source.shape[0]})")

    try:
        for tile in TILES:
            first: np.ndarray[Any, Any] | None = None
            first_digest = ""
            records: list[PassRecord] = []

            for repeat in range(DETERMINISM_REPEATS):
                output, record = run_pass(manager, source, tile)
                records.append(record)
                print(
                    f"    {arm(tile)} run {repeat + 1}  {record.seconds:6.1f}s  "
                    f"tiles={record.tiles:3d} effective={record.effective_tile} "
                    f"peak={record.peak_vram_mb:6.0f}MB",
                    flush=True,
                )
                if first is None:
                    first = output
                    first_digest = digest(output)
                    continue

                difference = compare(first, output)
                rows.append(
                    {
                        "experiment": "determinism",
                        "category": image.category,
                        "arm": arm(tile),
                        "repeat": repeat + 1,
                        "dimensions": [int(output.shape[1]), int(output.shape[0])],
                        "dimensions_match": output.shape == first.shape,
                        "sha256_first": first_digest,
                        "sha256_repeat": digest(output),
                        "difference": difference.as_dict(),
                        "passes": [record.as_dict() for record in records],
                    }
                )
                verdict = "IDENTICAL" if difference.identical else "DIFFERS"
                print(
                    f"      vs run 1: {verdict}  max={difference.max_abs} "
                    f"differing={difference.fraction_differing:.6%}",
                    flush=True,
                )
                del output

            del first

    finally:
        manager.release()

    return rows


# ---------------------------------------------------------------- the sweep


def run_sweep(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Three tile sizes over five photographs.

    Image-major, and unlike F2 and F3 the neural pass cannot be cached - the
    tile size *is* the variable, so every cell is its own pass. Fifteen passes.

    All three arms of one photograph are held in memory together, because the
    exact pairwise comparison is the primary result and it needs them
    simultaneously. Three 4x arms of the largest corpus image is about 290 MB,
    which is affordable; the arms are dropped before the next photograph.
    """
    manager = _make_manager()
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    try:
        for image in images:
            source = load_rgb(image)
            print(
                f"\n  {image.category}  {source.shape[1]}x{source.shape[0]} "
                f"-> {source.shape[1] * SCALE}x{source.shape[0] * SCALE}",
                flush=True,
            )

            outputs: dict[str, np.ndarray[Any, Any]] = {}
            shared_mask: Any = None

            for tile in TILES:
                name = arm(tile)
                output, record = run_pass(manager, source, tile)
                outputs[name] = output

                if not record.honoured:
                    print(
                        f"    {name}  NOT HONOURED: asked {tile}, ran "
                        f"{record.effective_tile} on {record.device} - cell excluded",
                        flush=True,
                    )

                luma = measure(output)
                tiles = ycrcb_tiles(output)
                if shared_mask is None:
                    shared_mask = flat_mask(tiles)
                chroma = chroma_metrics(tiles, shared_mask)
                del tiles

                seams = seam_report(output)
                difference = (
                    compare(outputs[arm(REFERENCE_TILE)], output)
                    if tile != REFERENCE_TILE
                    else None
                )

                rows.append(
                    {
                        "experiment": "sweep",
                        "category": image.category,
                        "arm": name,
                        "tile": tile,
                        "dimensions": [int(output.shape[1]), int(output.shape[0])],
                        "sha256": digest(output),
                        "pass": record.as_dict(),
                        "luma": _metrics_dict(luma),
                        "chroma": chroma.as_dict(),
                        "seams": seams,
                        "vs_reference": difference.as_dict() if difference else None,
                    }
                )

                vertical = seams["vertical"]
                print(
                    f"    {name}  {record.seconds:6.1f}s  tiles={record.tiles:3d}  "
                    f"peak={record.peak_vram_mb:6.0f}MB  "
                    f"seam z: 1024={vertical['t256']['z_median']:+.2f} "
                    f"512={vertical['t128']['z_median']:+.2f} "
                    f"256={vertical['t064']['z_median']:+.2f}",
                    flush=True,
                )
                if difference is not None:
                    print(
                        f"           vs {arm(REFERENCE_TILE)}: "
                        f"{'IDENTICAL' if difference.identical else 'differs'} "
                        f"max={difference.max_abs} "
                        f"differing={difference.fraction_differing:.4%} "
                        f"psnr={difference.psnr_db:.1f}dB",
                        flush=True,
                    )

            rows.extend(_write_crops(image.category, outputs))
            outputs.clear()
            del source

    finally:
        freed = manager.release()
        print(f"\n  released {freed} model(s)", flush=True)

    return rows


def _metrics_dict(metrics: ImageMetrics) -> dict[str, Any]:
    return asdict(metrics)


# ------------------------------------------------------------------- crops


def _write_crops(category: str, outputs: dict[str, np.ndarray[Any, Any]]) -> list[dict[str, Any]]:
    """One texture region and two seam regions per photograph, per arm.

    Nine images per photograph rather than F3's thirty. Tiling, if it does
    anything, does it at coordinates the geometry already names, so there is no
    reason to carpet the frame with crops: the seam crops go straight there and
    are placed by measurement rather than by taste.
    """
    from PIL import Image as PILImage

    rows: list[dict[str, Any]] = []
    finest = outputs[arm(TILES[-1])]
    height, width = finest.shape[:2]

    # ------------------------------------------------------------ texture
    region = DETAIL_REGIONS[category]
    cx, cy, size = CROP_REGIONS[category][region]
    span = size * SCALE

    for name, output in outputs.items():
        left = min(max(0, cx * SCALE - span // 2), max(0, output.shape[1] - span))
        top = min(max(0, cy * SCALE - span // 2), max(0, output.shape[0] - span))
        PILImage.fromarray(output[top : top + span, left : left + span]).save(
            CROPS_DIR / f"{category}__{region}__{name}.jpg",
            quality=DETAIL_QUALITY,
            subsampling=0,
        )

    # --------------------------------------------------------------- seams
    #
    # Located on the finest arm, which is where a joint is most likely to exist
    # at all, and then cropped at the same coordinates from every arm - so the
    # three panels differ only in whether tiling put a joint there.
    columns, _ = edge_profiles(finest)
    luma = to_luma(finest) * LUMA_FULL_SCALE

    for grid in SEAM_GRIDS:
        # A boundary on this grid is a joint for the arm that owns it and for
        # every finer arm; for the coarser ones it is ordinary image content.
        grid_tile = next(tile for tile in TILES if arm(tile) == grid)
        column, seam_z = worst_vertical_seam(columns, width, grid)
        if column <= 0:
            continue

        left = min(max(0, column - SEAM_CROP_W // 2), max(0, width - SEAM_CROP_W))
        top = min(busiest_band(luma, column, SEAM_CROP_H), max(0, height - SEAM_CROP_H))

        for name, output in outputs.items():
            PILImage.fromarray(output[top : top + SEAM_CROP_H, left : left + SEAM_CROP_W]).save(
                CROPS_DIR / f"{category}__seam-{grid}__{name}.png"
            )

        rows.append(
            {
                "experiment": "seam-crop",
                "category": category,
                "grid": grid,
                "seams_for": [arm(t) for t in TILES if arm(t) in outputs and t <= grid_tile],
                "seam_column": column,
                "seam_offset_in_crop": column - left,
                "worst_z_on_grid": seam_z,
                "crop": [left, top, SEAM_CROP_W, SEAM_CROP_H],
            }
        )

    return rows


# -------------------------------------------------------------- blind strips


def build_strips(seed: int = 20260911) -> list[dict[str, Any]]:
    """Comparison strips whose panel order is shuffled per region.

    Three panels, not six, and the shuffle matters more here than it did in F3.
    A tile-size ladder has an obvious expected direction - smaller tile, worse
    picture - and F3 established, at my own expense, that knowing the direction
    is enough to make a reader describe a progression the pixels do not contain.
    Panel order is recorded in the measurements file, never drawn on the image.
    """
    from PIL import Image as PILImage

    generator = np.random.default_rng(seed)
    names = [arm(tile) for tile in TILES]
    rows: list[dict[str, Any]] = []

    for category, region in sorted(DETAIL_REGIONS.items()):
        kinds = [(region, "jpg")] + [(f"seam-{grid}", "png") for grid in SEAM_GRIDS]

        for kind, extension in kinds:
            available = [
                name
                for name in names
                if (CROPS_DIR / f"{category}__{kind}__{name}.{extension}").exists()
            ]
            if len(available) < 2:
                continue

            order = list(available)
            generator.shuffle(order)

            panels = []
            for name in order:
                panel = np.asarray(
                    PILImage.open(CROPS_DIR / f"{category}__{kind}__{name}.{extension}").convert(
                        "RGB"
                    )
                ).copy()
                panel[:, -4:] = 255
                panels.append(panel)

            out_name = f"blind__{category}__{kind}.{extension}"
            strip = PILImage.fromarray(np.concatenate(panels, axis=1))
            if extension == "jpg":
                strip.save(CROPS_DIR / out_name, quality=DETAIL_QUALITY, subsampling=0)
            else:
                strip.save(CROPS_DIR / out_name)

            rows.append(
                {
                    "experiment": "strips",
                    "category": category,
                    "region": kind,
                    "file": out_name,
                    "panel_order": order,
                }
            )
            print(f"  {out_name}", flush=True)

    return rows


# ------------------------------------------------------------------ plumbing


def provenance() -> dict[str, Any]:
    import torch

    from app.core.config import Settings
    from app.schemas.job import EnhanceSettings
    from app.services.mode_planner import CREATIVE_DENOISE

    settings = Settings()
    defaults = EnhanceSettings()
    entries = load_manifest(DEFAULT_CORPUS_DIR)

    return {
        "phase": "4-F4",
        "generated": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "model": MODEL,
        "scale": SCALE,
        "tiles": list(TILES),
        "reference_tile": REFERENCE_TILE,
        "sharpen_strength": 0.0,
        "denoise_strength": None,
        "production_tile_size": settings.tile_size,
        "production_tile_pad": settings.tile_pad,
        # Recorded so the report can show the production defaults were untouched
        # while this ran. There is no importable `DEFAULT_DENOISE`: the shipped
        # default is the schema's `None`, which an earlier phase established is
        # byte-identical to an explicit 1.0.
        "schema_default_denoise": defaults.denoise_strength,
        "schema_default_sharpen": defaults.sharpen_strength,
        "creative_denoise_unchanged": CREATIVE_DENOISE,
        "seam_window": SEAM_WINDOW,
        "corpus": [
            {
                "category": entry.category,
                "filename": entry.filename,
                "sha256": hashlib.sha256(
                    (DEFAULT_CORPUS_DIR / entry.filename).read_bytes()
                ).hexdigest(),
            }
            for entry in entries
        ],
    }


def _load_existing() -> dict[str, Any]:
    if MEASUREMENTS.exists():
        loaded: dict[str, Any] = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
        return loaded
    return {"provenance": provenance(), "rows": []}


def _save(document: dict[str, Any], rows: list[dict[str, Any]], experiments: set[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kept = [row for row in document.get("rows", []) if row.get("experiment") not in experiments]
    document["rows"] = kept + rows
    document["provenance"] = provenance()
    MEASUREMENTS.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\n  wrote {MEASUREMENTS.relative_to(Path.cwd())}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism", action="store_true", help="repeatability control")
    parser.add_argument("--sweep", action="store_true", help="5 photographs x 3 tile sizes")
    parser.add_argument("--strips", action="store_true", help="blind comparison strips")
    arguments = parser.parse_args()

    if not (arguments.determinism or arguments.sweep or arguments.strips):
        parser.print_help()
        return 2

    document = _load_existing()
    images = load_corpus()
    if (arguments.sweep or arguments.determinism) and not images:
        raise SystemExit("corpus is empty - run benchmarks/fetch_corpus.py first")

    if arguments.determinism:
        _save(document, run_determinism(images), {"determinism"})

    if arguments.sweep:
        # Refused rather than warned. A tile-to-tile difference is only evidence
        # about tiling once the same tile against itself is known to be zero.
        if not any(row.get("experiment") == "determinism" for row in document.get("rows", [])):
            raise SystemExit("run --determinism first: the sweep is uninterpretable without it")
        _save(document, run_sweep(images), {"sweep", "seam-crop"})

    if arguments.strips:
        _save(document, build_strips(), {"strips"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
