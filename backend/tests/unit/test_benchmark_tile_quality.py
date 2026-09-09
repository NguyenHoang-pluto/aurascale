"""The Phase 4 F4 seam geometry, checked against the production tiler.

The whole phase rests on one claim: that a tile joint lands at every multiple of
`tile_size * scale` in output coordinates. If that is wrong then every seam
measurement is reading ordinary image content at coordinates chosen by mistake,
and the report would be confidently describing nothing.

The brief was explicit that the geometry had to be *derived* from the
implementation rather than assumed, so these tests derive it the other way
round - from `plan_tiles`, the real thing - and require the two to agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.inference.tiler import plan_tiles
from benchmarks.phase4_f4 import (
    SCALE,
    TILES,
    busiest_band,
    compare,
    digest,
    edge_profiles,
    exclusive_grids,
    score_grid,
    seam_boundaries,
    ssim_sampled,
)

# The five corpus images, as (width, height). Real sizes rather than round
# numbers: a tile grid that only works on multiples of the tile size is not a
# tile grid that works.
CORPUS_SHAPES = [
    (1732, 1154),
    (1307, 1530),
    (1632, 1224),
    (1224, 1632),
    (1892, 1057),
]


def tiler_seam_columns(width: int, height: int, tile: int, pad: int = 16) -> set[int]:
    """Where the production tiler actually starts a new tile, horizontally."""
    grid = plan_tiles(width, height, tile_size=tile, tile_pad=pad)
    return {
        columns.start for _, columns in (t.output_slice(SCALE) for t in grid.tiles()) if columns.start
    }


def tiler_seam_rows(width: int, height: int, tile: int, pad: int = 16) -> set[int]:
    """The same for the vertical direction."""
    grid = plan_tiles(width, height, tile_size=tile, tile_pad=pad)
    return {rows.start for rows, _ in (t.output_slice(SCALE) for t in grid.tiles()) if rows.start}


@pytest.mark.parametrize(("width", "height"), CORPUS_SHAPES)
@pytest.mark.parametrize("tile", TILES)
def test_derived_seams_match_the_production_tiler(width: int, height: int, tile: int) -> None:
    """The seam coordinates F4 measures are the ones the tiler writes at."""
    derived_columns = {boundary + 1 for boundary in seam_boundaries(width * SCALE, tile)}
    derived_rows = {boundary + 1 for boundary in seam_boundaries(height * SCALE, tile)}

    assert derived_columns == tiler_seam_columns(width, height, tile)
    assert derived_rows == tiler_seam_rows(width, height, tile)


@pytest.mark.parametrize(("width", "height"), CORPUS_SHAPES)
@pytest.mark.parametrize("tile", TILES)
def test_padding_is_cropped_back_off(width: int, height: int, tile: int) -> None:
    """Every tile writes exactly its own rectangle, pad included in neither.

    This is what makes a seam a butt joint rather than a blend, and therefore
    what makes the phase's question meaningful.
    """
    grid = plan_tiles(width, height, tile_size=tile, tile_pad=16)
    for planned in grid.tiles():
        crop_rows, crop_columns = planned.crop_slice(SCALE)
        out_rows, out_columns = planned.output_slice(SCALE)
        assert crop_columns.stop - crop_columns.start == out_columns.stop - out_columns.start
        assert crop_rows.stop - crop_rows.start == out_rows.stop - out_rows.start


@pytest.mark.parametrize(("width", "height"), CORPUS_SHAPES)
def test_exclusive_grids_partition_the_finest_grid(width: int, height: int) -> None:
    """The three grids are disjoint and together are exactly the tile-64 seams.

    The control depends on it: if a boundary appeared in two grids, a coordinate
    would be scored as both a seam and a non-seam.
    """
    extent = width * SCALE
    grids = exclusive_grids(extent)

    combined = [boundary for boundaries in grids.values() for boundary in boundaries]
    assert len(combined) == len(set(combined))
    assert set(combined) == set(seam_boundaries(extent, min(TILES)))


@pytest.mark.parametrize(("width", "height"), CORPUS_SHAPES)
def test_each_grid_holds_only_boundaries_of_its_own_arm(width: int, height: int) -> None:
    """`t128` is the multiples of 512 that are not multiples of 1024, and so on.

    Stated as a property rather than a table, so it stays true if the arms
    change.
    """
    extent = width * SCALE
    grids = exclusive_grids(extent)
    coarser: set[int] = set()

    for tile in sorted(TILES, reverse=True):
        own = set(seam_boundaries(extent, tile))
        assert set(grids[f"t{tile:03d}"]) == own - coarser
        coarser |= own


def test_a_grid_is_empty_when_the_image_is_smaller_than_one_tile() -> None:
    """No internal joints to score, and no crash while finding that out."""
    assert seam_boundaries(400, 256) == []
    assert score_grid(np.zeros(100), []).count == 0


def test_score_grid_finds_a_planted_step() -> None:
    """A discontinuity at a seam reads as a large z; flat neighbours read as 0."""
    profile = np.full(600, 2.0)
    profile[255] = 20.0

    planted = score_grid(profile, [255])
    assert planted.count == 1
    assert planted.z_median > 3.0

    quiet = score_grid(profile, [300])
    assert quiet.z_median == pytest.approx(0.0)


def test_score_grid_survives_a_perfectly_flat_neighbourhood() -> None:
    """A zero MAD must not produce an infinity that swamps every aggregate."""
    profile = np.zeros(600)
    profile[255] = 5.0
    assert np.isfinite(score_grid(profile, [255]).z_median)


def test_edge_profiles_have_one_entry_per_boundary() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    columns, rows = edge_profiles(image)
    assert columns.shape == (59,)
    assert rows.shape == (39,)


def test_busiest_band_finds_the_structured_half() -> None:
    """The crop goes where the seam crosses detail, not where the frame is flat."""
    luma = np.zeros((400, 100), dtype=np.float64)
    luma[300:, 50:] = 255.0  # a hard step across column 50, lower half only
    assert busiest_band(luma, 50, 64) >= 236


def test_compare_reports_identity_exactly() -> None:
    image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    same = compare(image, image)

    assert same.identical
    assert same.max_abs == 0
    assert same.fraction_differing == 0.0
    assert same.ssim == 1.0
    assert same.psnr_db == float("inf")


def test_compare_counts_a_single_changed_subpixel() -> None:
    """The one number that separates "a pixel moved" from "the frame changed"."""
    left = np.zeros((64, 64, 3), dtype=np.uint8)
    right = left.copy()
    right[10, 10, 0] = 40

    difference = compare(left, right)
    assert not difference.identical
    assert difference.max_abs == 40
    assert difference.fraction_differing == pytest.approx(1 / left.size)


def test_compare_spans_more_than_one_chunk() -> None:
    """The chunked loop must not lose rows past the first block."""
    left = np.zeros((1200, 8, 3), dtype=np.uint8)
    right = left.copy()
    right[1100, 4, 1] = 7
    assert compare(left, right).max_abs == 7


def test_compare_refuses_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


def smooth_image(width: int = 1200, height: int = 1200) -> np.ndarray:
    """A gradient with a couple of hard edges, as a stand-in for a photograph.

    Deliberately *not* uniform noise. SSIM divides by the local variance of both
    inputs, so on a field that is already maximally noisy, added noise barely
    moves the score - which is correct behaviour and a useless test.
    """
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    base = np.repeat(ramp[None, :], height, axis=0)
    base[height // 3 : 2 * height // 3, width // 4 : width // 2] = 20.0
    return np.repeat(base.astype(np.uint8)[:, :, None], 3, axis=2)


def test_ssim_is_one_for_identical_images_and_falls_for_noise() -> None:
    image = smooth_image()
    generator = np.random.default_rng(1)
    noisy = np.clip(
        image.astype(np.int16) + generator.integers(-30, 31, image.shape), 0, 255
    ).astype(np.uint8)

    assert ssim_sampled(image, image) == pytest.approx(1.0)
    assert ssim_sampled(image, noisy) < 0.5


def test_ssim_is_symmetric_and_penalises_a_brightness_shift() -> None:
    """Neither arm is ground truth here, so the score has to be symmetric."""
    image = smooth_image()
    shifted = np.clip(image.astype(np.int16) + 12, 0, 255).astype(np.uint8)

    assert ssim_sampled(image, shifted) == pytest.approx(ssim_sampled(shifted, image))
    assert ssim_sampled(image, shifted) < 1.0


def test_digest_is_stable_and_shape_sensitive() -> None:
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert digest(image) == digest(image.copy())
    assert digest(image) != digest(image[::-1])
