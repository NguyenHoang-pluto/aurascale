"""Turning a requested output size into a neural pass plus an exact resize.

A target resolution and an upscale factor are different questions. "4x" asks
how much bigger; "4K" asks how big. This module answers the second by planning
the first, because the models only produce fixed factors and the gap has to be
closed somewhere visible rather than somewhere convenient.

The plan is always the same shape:

  1. work out the factor needed to reach the target's long edge;
  2. pick the **smallest** neural factor the model actually supports that
     reaches it, so nothing is enlarged further than it has to be;
  3. run that pass, then resample down to the exact target.

Downscaling at the end is a resample and is described as one. It is not the
thing the neural-only rule forbids: that rule exists so a result is never part
model and part interpolation *enlargement*. Reducing a 4x result to a 3.84x
target is supersampling - it discards pixels the model produced rather than
inventing pixels it did not.

The reverse is never done. If the source already meets the target, there is no
enhancement to perform and the request is refused rather than quietly answered
with a downscale.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import OutputTooLargeError, ValidationError
from app.models.enums import TargetResolution


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """How to reach a requested output size."""

    #: The factor the model is asked for. Always one it supports.
    neural_scale: int
    #: Size after the neural pass, before any resize.
    neural_width: int
    neural_height: int
    #: The size the user gets. Equal to the neural size when it already lands
    #: exactly on the target, which is what `needs_resize` reports.
    target_width: int
    target_height: int
    preset: TargetResolution

    @property
    def needs_resize(self) -> bool:
        """Whether a final resample is required at all."""
        return (self.neural_width, self.neural_height) != (self.target_width, self.target_height)

    @property
    def neural_pixels(self) -> int:
        return self.neural_width * self.neural_height

    @property
    def target_pixels(self) -> int:
        return self.target_width * self.target_height


def target_dimensions(width: int, height: int, preset: TargetResolution) -> tuple[int, int]:
    """The exact output size for a preset, preserving aspect ratio.

    The preset fixes the **long** edge; the short edge follows from the source
    proportions. A 16:9 photograph and a square one both become "4K" and
    neither is stretched to a fixed 3840x2160 - which is the whole reason a
    preset is a long edge rather than a pair of numbers.
    """
    if width <= 0 or height <= 0:
        raise ValidationError(
            "The image dimensions are not usable.",
            technical=f"source={width}x{height}",
        )

    long_edge = preset.long_edge
    if width >= height:
        scaled_height = max(1, round(height * long_edge / width))
        return (long_edge, scaled_height)

    scaled_width = max(1, round(width * long_edge / height))
    return (scaled_width, long_edge)


def plan_resolution(
    width: int,
    height: int,
    preset: TargetResolution,
    *,
    supported_scales: list[int],
    max_output_pixels: int,
) -> ResolutionPlan:
    """Plan the passes needed to reach `preset`, or refuse and say why.

    Every refusal happens here, before a worker is handed anything, because a
    request that cannot be satisfied should cost the user a message rather than
    four minutes of compute.
    """
    target_width, target_height = target_dimensions(width, height, preset)

    source_long = max(width, height)
    target_long = preset.long_edge

    # Already there. Enhancement cannot make an image smaller, and quietly
    # returning a downscale would answer a question nobody asked.
    if source_long >= target_long:
        raise ValidationError(
            f"That image is already {width}x{height}, which is at or beyond "
            f"{preset.value.upper()} ({target_long} px on the long edge). "
            "Choose a larger target, or use an upscale factor instead.",
            technical=f"source_long={source_long} target_long={target_long}",
            context={
                "sourceWidth": width,
                "sourceHeight": height,
                "targetLongEdge": target_long,
                "preset": preset.value,
            },
        )

    if not supported_scales:
        raise ValidationError(
            "That model cannot produce any upscale factor.",
            technical="supported_scales is empty",
        )

    required = target_long / source_long
    usable = sorted(scale for scale in supported_scales if scale >= required)

    if not usable:
        largest = max(supported_scales)
        reachable = source_long * largest
        raise ValidationError(
            f"Reaching {preset.value.upper()} from a {width}x{height} image needs "
            f"{required:.2f}x, and the largest factor available is {largest}x. "
            f"The most this image can reach is about {reachable} px on the long edge.",
            technical=f"required={required:.4f} supported={supported_scales}",
            context={
                "requiredScale": round(required, 4),
                "maximumScale": largest,
                "preset": preset.value,
                "reachableLongEdge": reachable,
            },
        )

    # The smallest factor that reaches the target: anything larger is work
    # thrown away by the resize, and more memory held while doing it.
    neural_scale = usable[0]
    neural_width = width * neural_scale
    neural_height = height * neural_scale

    # The intermediate is the largest thing this job will hold, so it is what
    # the limit has to be checked against - not the final, smaller target.
    if neural_width * neural_height > max_output_pixels:
        raise OutputTooLargeError(
            f"Reaching {preset.value.upper()} from that image needs a {neural_scale}x pass, "
            f"which is {neural_width * neural_height / 1_000_000:.0f} MP - beyond the "
            f"{max_output_pixels / 1_000_000:.0f} MP limit. Try a smaller target.",
            technical=(
                f"neural={neural_width}x{neural_height} scale={neural_scale} "
                f"limit={max_output_pixels}"
            ),
            context={
                "limitPixels": max_output_pixels,
                "projectedPixels": neural_width * neural_height,
                "neuralScale": neural_scale,
                "preset": preset.value,
            },
        )

    return ResolutionPlan(
        neural_scale=neural_scale,
        neural_width=neural_width,
        neural_height=neural_height,
        target_width=target_width,
        target_height=target_height,
        preset=preset,
    )
