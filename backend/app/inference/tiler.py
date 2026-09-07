"""Tiling geometry.

Deliberately free of torch and numpy: the grid arithmetic is where seam and
off-by-one bugs live, and keeping it pure means it can be tested exhaustively
in milliseconds rather than through a model.

A tile has two rectangles. The *output* rectangle is the region this tile is
responsible for writing. The *input* rectangle is that region grown by
`tile_pad` on every side and clamped to the image, which is what the model
actually sees. Convolutions near a boundary would otherwise see zero-padding
instead of real neighbouring pixels, which is exactly what a visible tile grid
is made of.

After inference the padding is removed at `pad * scale`, so what lands in the
canvas is only the tile's own region, reconstructed with real context around it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tile:
    """One cell of the grid, in input-image pixel coordinates."""

    index: int
    row: int
    column: int

    # The region this tile owns.
    x: int
    y: int
    width: int
    height: int

    # The padded region fed to the model, clamped at the image edge.
    input_x: int
    input_y: int
    input_width: int
    input_height: int

    @property
    def input_slice(self) -> tuple[slice, slice]:
        """Rows and columns to read from the input image."""
        return (
            slice(self.input_y, self.input_y + self.input_height),
            slice(self.input_x, self.input_x + self.input_width),
        )

    def output_slice(self, scale: int) -> tuple[slice, slice]:
        """Rows and columns this tile writes in the output canvas."""
        return (
            slice(self.y * scale, (self.y + self.height) * scale),
            slice(self.x * scale, (self.x + self.width) * scale),
        )

    def crop_slice(self, scale: int) -> tuple[slice, slice]:
        """Rows and columns to keep from the model's output for this tile.

        The offsets are how much padding was actually applied on the top and
        left, which is less than `tile_pad` at the image edge — the clamp is
        why this cannot simply be `pad * scale`.
        """
        left = (self.x - self.input_x) * scale
        top = (self.y - self.input_y) * scale
        return (
            slice(top, top + self.height * scale),
            slice(left, left + self.width * scale),
        )


@dataclass(frozen=True, slots=True)
class TileGrid:
    """The full plan for one inference pass."""

    width: int
    height: int
    tile_size: int
    tile_pad: int
    columns: int
    rows: int

    @property
    def count(self) -> int:
        return self.columns * self.rows

    @property
    def is_single_tile(self) -> bool:
        """True when the whole image goes through the model in one piece."""
        return self.count == 1

    def __iter__(self) -> Iterator[Tile]:
        return iter(self.tiles())

    def tiles(self) -> list[Tile]:
        """Every tile, in row-major order."""
        planned: list[Tile] = []

        for row in range(self.rows):
            for column in range(self.columns):
                x = column * self.tile_size
                y = row * self.tile_size
                # The last tile in a row or column is short unless the image
                # divides exactly; taking the remainder here is what keeps the
                # grid covering the image without running past its edge.
                width = min(self.tile_size, self.width - x)
                height = min(self.tile_size, self.height - y)

                input_x = max(0, x - self.tile_pad)
                input_y = max(0, y - self.tile_pad)
                input_right = min(self.width, x + width + self.tile_pad)
                input_bottom = min(self.height, y + height + self.tile_pad)

                planned.append(
                    Tile(
                        index=len(planned),
                        row=row,
                        column=column,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        input_x=input_x,
                        input_y=input_y,
                        input_width=input_right - input_x,
                        input_height=input_bottom - input_y,
                    )
                )

        return planned


def plan_tiles(width: int, height: int, *, tile_size: int, tile_pad: int) -> TileGrid:
    """Divide an image into tiles.

    `tile_size <= 0` means "no tiling": the image is one tile, which is what a
    small image or an explicitly disabled TILE_SIZE should do. A tile larger
    than the image is also collapsed to a single tile, so a 64x64 thumbnail
    does not pay for a 256-pixel grid.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image must have positive dimensions, got {width}x{height}")
    if tile_pad < 0:
        raise ValueError(f"tile_pad must not be negative, got {tile_pad}")

    if tile_size <= 0 or (tile_size >= width and tile_size >= height):
        return TileGrid(
            width=width,
            height=height,
            tile_size=max(width, height),
            tile_pad=0,
            columns=1,
            rows=1,
        )

    columns = -(-width // tile_size)  # ceiling division
    rows = -(-height // tile_size)

    return TileGrid(
        width=width,
        height=height,
        tile_size=tile_size,
        tile_pad=tile_pad,
        columns=columns,
        rows=rows,
    )
