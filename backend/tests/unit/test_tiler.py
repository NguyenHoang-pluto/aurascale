"""Tiling geometry.

The grid is where seam bugs and off-by-ones live, so these tests check the two
properties that matter for correctness rather than a handful of examples:

  * the output regions exactly partition the image - no gap, no overlap;
  * every model input contains its own region plus real context around it.

Both are checked over a range of awkward sizes, including images that do not
divide evenly and tiles larger than the image.
"""

from __future__ import annotations

import pytest

from app.inference.tiler import plan_tiles


@pytest.mark.parametrize(
    ("width", "height", "tile"),
    [
        (64, 64, 32),
        (100, 100, 32),  # does not divide evenly
        (300, 200, 128),
        (257, 129, 64),  # one pixel past a boundary in both axes
        (1280, 720, 256),
        (33, 17, 16),
    ],
)
def test_tiles_partition_the_image_exactly(width: int, height: int, tile: int) -> None:
    grid = plan_tiles(width, height, tile_size=tile, tile_pad=16)

    covered = set()
    for planned in grid.tiles():
        for y in range(planned.y, planned.y + planned.height):
            for x in range(planned.x, planned.x + planned.width):
                # A pixel written twice is a seam waiting to happen.
                assert (x, y) not in covered
                covered.add((x, y))

    assert len(covered) == width * height


@pytest.mark.parametrize(("width", "height", "tile", "pad"), [(300, 200, 128, 16), (65, 65, 32, 8)])
def test_padded_input_contains_the_tile_and_stays_inside_the_image(
    width: int, height: int, tile: int, pad: int
) -> None:
    grid = plan_tiles(width, height, tile_size=tile, tile_pad=pad)

    for planned in grid.tiles():
        assert planned.input_x <= planned.x
        assert planned.input_y <= planned.y
        assert planned.input_x + planned.input_width >= planned.x + planned.width
        assert planned.input_y + planned.input_height >= planned.y + planned.height

        # Clamped at the edges: padding never reads outside the image, which is
        # what would otherwise need zero-fill and produce a visible border.
        assert planned.input_x >= 0
        assert planned.input_y >= 0
        assert planned.input_x + planned.input_width <= width
        assert planned.input_y + planned.input_height <= height


def test_interior_tiles_get_the_full_padding() -> None:
    grid = plan_tiles(300, 300, tile_size=100, tile_pad=16)

    middle = next(t for t in grid.tiles() if t.row == 1 and t.column == 1)

    assert middle.input_x == middle.x - 16
    assert middle.input_y == middle.y - 16
    assert middle.input_width == middle.width + 32
    assert middle.input_height == middle.height + 32


def test_crop_offsets_follow_the_padding_that_was_actually_applied() -> None:
    """At the image edge less padding fits, so the crop cannot be a constant.

    Getting this wrong shifts every edge tile by `tile_pad * scale` pixels,
    which is the classic tiled-inference artefact.
    """
    grid = plan_tiles(300, 300, tile_size=100, tile_pad=16)
    scale = 4

    first = grid.tiles()[0]
    rows, columns = first.crop_slice(scale)
    assert rows.start == 0  # nothing to crop at the top edge
    assert columns.start == 0
    assert rows.stop - rows.start == first.height * scale

    middle = next(t for t in grid.tiles() if t.row == 1 and t.column == 1)
    rows, columns = middle.crop_slice(scale)
    assert rows.start == 16 * scale
    assert columns.start == 16 * scale


def test_output_slices_scale_with_the_model_factor() -> None:
    grid = plan_tiles(200, 100, tile_size=100, tile_pad=8)

    for planned in grid.tiles():
        rows, columns = planned.output_slice(4)
        assert rows.start == planned.y * 4
        assert columns.stop == (planned.x + planned.width) * 4


@pytest.mark.parametrize("tile", [0, -1, 512])
def test_a_tile_that_covers_the_image_collapses_to_a_single_pass(tile: int) -> None:
    """No tiling means no padding overhead: a small image runs in one piece."""
    grid = plan_tiles(256, 256, tile_size=tile, tile_pad=16)

    assert grid.is_single_tile
    assert grid.count == 1

    only = grid.tiles()[0]
    assert (only.width, only.height) == (256, 256)
    assert (only.input_width, only.input_height) == (256, 256)


def test_a_tile_larger_than_one_axis_still_tiles_the_other() -> None:
    """A wide, short image needs columns but only one row."""
    grid = plan_tiles(1000, 100, tile_size=256, tile_pad=16)

    assert grid.rows == 1
    assert grid.columns == 4
    assert not grid.is_single_tile


def test_grid_counts_are_ceiling_divisions() -> None:
    grid = plan_tiles(257, 129, tile_size=128, tile_pad=16)

    assert (grid.columns, grid.rows) == (3, 2)
    assert grid.count == 6


def test_iterating_the_grid_yields_its_tiles_in_row_major_order() -> None:
    grid = plan_tiles(300, 200, tile_size=128, tile_pad=8)

    positions = [(tile.row, tile.column) for tile in grid]

    assert positions == sorted(positions)
    assert [tile.index for tile in grid] == list(range(grid.count))


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-1, 5)])
def test_an_empty_image_is_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        plan_tiles(width, height, tile_size=64, tile_pad=8)


def test_negative_padding_is_rejected() -> None:
    with pytest.raises(ValueError, match="tile_pad"):
        plan_tiles(64, 64, tile_size=32, tile_pad=-1)
