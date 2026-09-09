"""The enhancement pipeline, from a decoded image to an enhanced one.

This is stages 6-11 of docs/image-processing.md: alpha split, the neural
pass or passes, the optional post-process, and alpha recomposition. Decoding,
encoding and persistence are the caller's business, and nothing here knows what
a job or an HTTP request is.

Everything a user can switch on either runs a trained network or is labelled as
a post-process. There is no control here that pretends to be AI.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import OutputTooLargeError, ValidationError
from app.core.logging import get_logger
from app.inference.model_manager import ModelManager
from app.inference.upscaler import JobCancelledError, UpscaleReport
from app.services.model_service import ModelService, ModelStatus

logger = get_logger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

# The model that supplies the second pass of an 8x job. Two neural passes,
# so every pixel of the result is still model-generated (docs § 5).
SECOND_PASS_MODEL = "RealESRGAN_x2plus"

# Every factor the product offers. 2x and 4x are native weights; 8x and 16x
# are compositions of two neural passes, planned in `plan` below.
SUPPORTED_SCALES = (2, 4, 8, 16)

# Unsharp masking parameters.
#
# The radius is a Gaussian sigma in *output* pixels, and it has to follow the
# upscale factor. After an Nx enlargement the finest real structure - what the
# source resolved - spans roughly N output pixels, so a fixed radius sharpens
# a different thing at every scale: at 8x a sigma of 3 works almost entirely on
# frequencies the model interpolated, which carry the least real information.
# Scaling the radius keeps the filter pointed at the detail that came from the
# image rather than from the upsampling.
#
# 0.75 is chosen so that 4x - the default scale - lands on sigma 3.0, exactly
# what shipped before. 2x and 8x move relative to that anchor rather than to a
# new guess.
SHARPEN_SIGMA_PER_SCALE = 0.75
SHARPEN_SIGMA_RANGE = (1.0, 6.0)
SHARPEN_MAX_AMOUNT = 1.5

# Shaping applied to the high-frequency component before it is added back, in
# 0-255 units. Both ends matter and they do different jobs:
#
#   * below the floor the detail is discarded, so sensor noise and compression
#     mottle in flat regions are not amplified into visible grain;
#   * above the ceiling it is clamped, which is what limits the bright and dark
#     rim - the halo - that unsharp masking leaves along a strong edge.
#
# Between the two the response is linear, which is where texture lives. That is
# the whole design: little change in flat areas, most of the effect on
# mid-amplitude micro-detail, and a bounded amount on hard edges.
SHARPEN_NOISE_FLOOR = 2.0
SHARPEN_DETAIL_CEILING = 10.0

# Rec.709 luma. The sharpening correction is computed on this and added equally
# to R, G and B, which preserves colour relationships.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

# How much of the image the sharpener holds in float at once, in bytes of the
# widest intermediate. An 8x result can reach 195 MP, where converting the
# whole frame to float32 is 2.2 GiB and the blur of its luma another 744 MiB -
# more than the machine has, and the allocation OpenCV fails on. Working a
# strip at a time makes the cost a function of this budget and the image width
# rather than of the output size, so a 195 MP job costs the same as a 12 MP one.
SHARPEN_STRIP_BYTES = 64 * 1024 * 1024
# Never go below this many rows, or a very wide image would be processed in
# slivers and pay the per-strip overhead many times over.
SHARPEN_MIN_STRIP_ROWS = 64


@dataclass(frozen=True, slots=True)
class EnhancementRequest:
    """What the user asked for."""

    model_id: str
    scale: int
    # Weight on the standard checkpoint when the model has a denoise pair;
    # 1.0 denoises most, 0.0 preserves noise. None leaves the weights alone.
    denoise_strength: float | None = None
    # 0 disables the post-process entirely.
    sharpen_strength: float = 0.0
    # Per-job tiling overrides. None means "use the configured default", which
    # is then clamped to the free VRAM as usual.
    tile_size: int | None = None
    tile_pad: int | None = None
    # Exact output size, when the job asked for a target resolution rather than
    # a factor. Both None means the neural result is the result.
    target_width: int | None = None
    target_height: int | None = None

    @property
    def has_target(self) -> bool:
        return self.target_width is not None and self.target_height is not None


@dataclass(frozen=True, slots=True)
class CascadeStage:
    """One neural pass of a composed scale, with the size it operates on.

    8x and 16x are two passes, and the second is by far the expensive one: it
    sees an image the first pass already enlarged, sixteen times larger at 16x.
    Naming the stages with their dimensions is what lets each be checked
    against the output limit *before* it runs, rather than the problem arriving
    as an allocation failure partway through - or, worse, as a half-finished
    cascade returning a 4x image as though it were the answer.
    """

    #: 0-based position in the cascade.
    index: int
    #: How many passes there are in total.
    total: int
    model_id: str
    #: The model's native factor, not the job's.
    scale: int
    input_width: int
    input_height: int

    @property
    def output_width(self) -> int:
        return self.input_width * self.scale

    @property
    def output_height(self) -> int:
        return self.input_height * self.scale

    @property
    def output_pixels(self) -> int:
        return self.output_width * self.output_height

    @property
    def is_final(self) -> bool:
        return self.index == self.total - 1


@dataclass(frozen=True, slots=True)
class EnhancementResult:
    """The enhanced image plus what it took to produce it."""

    image: np.ndarray[Any, Any]
    scale: int
    passes: list[str]
    reports: list[UpscaleReport] = field(default_factory=list)
    sharpened: bool = False
    #: Whether a final resample to an exact target size was applied.
    resized: bool = False

    @property
    def device(self) -> str:
        """Where the work actually ran.

        Reports the last pass: if an earlier pass fell back to CPU, later ones
        follow it, so the final answer is the honest one.
        """
        return self.reports[-1].device if self.reports else "unknown"

    @property
    def tile_size(self) -> int | None:
        return self.reports[-1].tile_size if self.reports else None

    @property
    def fell_back_to_cpu(self) -> bool:
        return any(report.fell_back_to_cpu for report in self.reports)

    @property
    def tile_size_reduced(self) -> bool:
        return any(report.tile_size_reduced for report in self.reports)


class EnhancementService:
    """Runs an image through one or two models and any post-processing."""

    def __init__(
        self,
        settings: Settings,
        manager: ModelManager | None = None,
        models: ModelService | None = None,
    ) -> None:
        self._settings = settings
        self._models = models or ModelService(settings)
        self._manager = manager or ModelManager(settings, self._models)

    # ------------------------------------------------------------------ planning

    def plan(self, model_id: str, scale: int) -> list[str]:
        """Which models run, in order, to reach `scale`.

        Scales are composed from native model factors rather than resampled
        afterwards, so the whole result is model-generated. A request the
        available weights cannot satisfy neurally is refused here, before any
        work starts, with a message that names what would work.
        """
        if scale not in SUPPORTED_SCALES:
            raise ValidationError(
                f"{scale}x is not one of the available upscale factors.",
                technical=f"scale={scale}, supported={SUPPORTED_SCALES}",
                context={"supportedScales": list(SUPPORTED_SCALES)},
            )

        status = self._models.get(model_id)
        native = status.entry.scale

        if scale == native:
            return [model_id]

        if scale == 8 and native == 4:
            # 4x then 2x. There is no official 8x weight, and a bicubic second
            # step would make half the result non-neural (docs § 5).
            return [model_id, SECOND_PASS_MODEL]

        if scale == 16 and native == 4:
            # 4x then 4x - the same weights twice, so the second pass carries
            # the character of the first rather than handing the image to a
            # different network halfway through.
            #
            # Not 4x -> 2x -> 2x: that reaches the same size with a third
            # forward pass over an image four times larger than this one's
            # second pass sees, which is strictly more work and more memory
            # for no gain. And not one 16x allocation either - each pass goes
            # through the ordinary tiling loop, so the OOM ladder, the
            # cancellation checks and the progress counting all still apply.
            return [model_id, model_id]

        raise ValidationError(
            f"{status.entry.name} upscales by {native}x, so it cannot produce a {scale}x result.",
            technical=f"model={model_id} native_scale={native} requested={scale}",
            context={
                "model": model_id,
                "nativeScale": native,
                "requestedScale": scale,
                "suggestion": self._suggest_model_for(scale),
            },
        )

    def supported_scales(self, model_id: str) -> list[int]:
        """Every scale this model can reach with neural passes only.

        Derived by asking `plan` rather than restating its rule, so the answer
        the API publishes and the answer a job is validated against can never
        drift apart. This is what lets the UI grey out an impossible
        combination instead of duplicating the routing logic.
        """
        available: list[int] = []

        for scale in SUPPORTED_SCALES:
            try:
                self.plan(model_id, scale)
            except ValidationError:
                continue
            available.append(scale)

        return available

    def plan_cascade(
        self, model_id: str, scale: int, width: int, height: int
    ) -> list[CascadeStage]:
        """The passes `plan` chose, each with the size it will work on.

        Routing is not restated here - `plan` is called - so a cascade and the
        list of models it runs can never describe different journeys. What this
        adds is the arithmetic: every stage knows how large its own result will
        be, which is what the memory guard below needs and what makes a
        two-pass job inspectable without running it.
        """
        models = self.plan(model_id, scale)

        stages: list[CascadeStage] = []
        current_width, current_height = width, height

        for index, identifier in enumerate(models):
            stage = CascadeStage(
                index=index,
                total=len(models),
                model_id=identifier,
                scale=self._models.get(identifier).entry.scale,
                input_width=current_width,
                input_height=current_height,
            )
            stages.append(stage)
            current_width, current_height = stage.output_width, stage.output_height

        return stages

    def _suggest_model_for(self, scale: int) -> str | None:
        """A model whose native scale matches, so the error can name one."""
        for status in self._models.list(selectable_only=True):
            if status.entry.scale == scale:
                return status.entry.id
        return None

    def required_models(self, model_id: str, scale: int) -> list[ModelStatus]:
        """Every weight file a request needs, including the denoise pair."""
        required: list[ModelStatus] = []

        for identifier in self.plan(model_id, scale):
            status = self._models.get(identifier)
            required.append(status)
            if status.entry.denoise_pair is not None:
                required.append(self._models.get(status.entry.denoise_pair))

        return required

    # ------------------------------------------------------------------- running

    def enhance(
        self,
        image: np.ndarray[Any, Any],
        request: EnhancementRequest,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> EnhancementResult:
        """Enhance a decoded RGB or RGBA image.

        Progress runs 0..1 across every pass, so a two-pass 8x job reports one
        continuous measurement rather than restarting at the halfway point.
        """
        colour, alpha = split_alpha(image)
        height, width = colour.shape[:2]
        stages = self.plan_cascade(request.model_id, request.scale, width, height)
        plan = [stage.model_id for stage in stages]
        reports: list[UpscaleReport] = []

        current = colour
        for stage in stages:
            # Checked before every pass, not only the first. The alternative is
            # a cancelled 16x job loading weights and then running a whole
            # second pass before anyone notices.
            if should_cancel is not None and should_cancel():
                raise JobCancelledError

            # And refused before every pass, so a cascade that cannot finish
            # safely says so instead of returning what it managed.
            self._assert_stage_fits(stage)

            upscaler = self._manager.get(
                stage.model_id,
                denoise_strength=self._denoise_for(stage.model_id, request.denoise_strength),
            )

            current = upscaler.upscale(
                current,
                tile=(
                    request.tile_size if request.tile_size is not None else self._settings.tile_size
                ),
                tile_pad=(
                    request.tile_pad if request.tile_pad is not None else self._settings.tile_pad
                ),
                on_progress=_scaled_progress(on_progress, stage.index, stage.total),
                should_cancel=should_cancel,
            )

            report = upscaler.last_report
            if report is not None:
                reports.append(report)

            logger.info(
                "pass complete",
                extra={
                    "model": stage.model_id,
                    "pass": stage.index + 1,
                    "passes": stage.total,
                    "output": f"{stage.output_width}x{stage.output_height}",
                    "device": report.device if report else None,
                    "tile": report.tile_size if report else None,
                },
            )

        # Resize before sharpening, so the post-process works at the size the
        # user will actually look at.
        resized = False
        if request.has_target:
            current, resized = resize_to_target(
                current, request.target_width or 0, request.target_height or 0
            )

        sharpened = request.sharpen_strength > 0
        if sharpened:
            current = unsharp_mask(current, request.sharpen_strength, scale=request.scale)

        if alpha is not None:
            current = attach_alpha(current, alpha)

        return EnhancementResult(
            image=current,
            scale=request.scale,
            passes=plan,
            reports=reports,
            sharpened=sharpened,
            resized=resized,
        )

    def _assert_stage_fits(self, stage: CascadeStage) -> None:
        """Refuse a pass whose result would be past the output limit.

        The submission path already checks the final size, and a target
        resolution has its neural intermediate checked by `plan_resolution`.
        This is the same limit applied at the moment the memory is about to be
        committed, and it is what makes a cascade safe to compose: if the
        second pass of a 16x job cannot run, the job fails with a typed error
        naming the stage rather than quietly handing back the 4x intermediate.

        It never fires for a request that passed submission, because every
        earlier stage is smaller than the last one, which is what was checked.
        """
        limit = self._settings.max_output_pixels
        if stage.output_pixels <= limit:
            return

        raise OutputTooLargeError(
            f"Pass {stage.index + 1} of {stage.total} of this enhancement would produce "
            f"{stage.output_width}x{stage.output_height} "
            f"({stage.output_pixels / 1_000_000:.0f} MP), which is beyond the "
            f"{limit / 1_000_000:.0f} MP limit. Try a smaller upscale factor.",
            technical=(
                f"stage={stage.index + 1}/{stage.total} model={stage.model_id} "
                f"input={stage.input_width}x{stage.input_height} "
                f"output={stage.output_width}x{stage.output_height} limit={limit}"
            ),
            context={
                "limitPixels": limit,
                "projectedPixels": stage.output_pixels,
                "stage": stage.index + 1,
                "stages": stage.total,
            },
        )

    def _denoise_for(self, model_id: str, strength: float | None) -> float | None:
        """Apply the denoise setting only to the model that supports it.

        The second pass of an 8x job is a different network with no denoise
        pair, so passing the strength on would fail the load.

        A 16x cascade runs the same weights twice, so a model with a pair gets
        the same blend on both passes. That is the existing rule applied
        unchanged rather than a new one - whether a second pass should denoise
        an already-denoised image is a quality question, and answering it here
        without evidence would be a guess.
        """
        if strength is None:
            return None

        entry = self._models.get(model_id).entry
        return strength if entry.supports_denoise and entry.denoise_pair else None

    def release(self) -> int:
        """Free every resident model. Called between jobs and on shutdown."""
        return self._manager.release()


# ---------------------------------------------------------------------- helpers


def split_alpha(
    image: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Separate an alpha channel, if there is one.

    Alpha never enters the model. RRDBNet has three input channels, and feeding
    a mask through a network trained on photographs invents plausible-looking
    edges in the transparency, which shows up as halos around cut-outs
    (docs/image-processing.md § 3).
    """
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1), None

    if image.ndim != 3:
        raise ValidationError(
            "The image could not be prepared for enhancement.",
            technical=f"unexpected array shape {image.shape}",
        )

    channels = image.shape[2]

    if channels == 4:
        return np.ascontiguousarray(image[:, :, :3]), np.ascontiguousarray(image[:, :, 3])
    if channels == 3:
        return image, None
    if channels == 1:
        return np.repeat(image, 3, axis=2), None

    raise ValidationError(
        "The image could not be prepared for enhancement.",
        technical=f"unsupported channel count {channels}",
    )


def attach_alpha(colour: np.ndarray[Any, Any], alpha: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Reattach an alpha channel, resampled to the enhanced size.

    Lanczos, not the model: this is a resample and the UI does not claim
    otherwise. Interpolating a mask is honest; hallucinating one is not.
    """
    import cv2

    height, width = colour.shape[:2]
    if alpha.shape[:2] != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return np.dstack([colour, alpha])


def sharpen_sigma(scale: int) -> float:
    """The Gaussian sigma to sharpen an `scale`x result with, in output pixels.

    Proportional to the upscale factor, then clamped. Deliberately *not* a
    strength multiplier: turning the amount up at 8x would drive the same
    interpolated frequencies harder and produce halos, where moving the radius
    changes which frequencies are touched at all.
    """
    low, high = SHARPEN_SIGMA_RANGE
    return float(min(high, max(low, SHARPEN_SIGMA_PER_SCALE * scale)))


def blur_margin(sigma: float) -> int:
    """Rows a strip must overlap its neighbour by for the blur to be exact.

    `cv2.GaussianBlur` with `ksize=(0, 0)` derives the kernel from sigma, and
    for a float image its support reaches exactly 4 sigma - measured, not
    assumed. A couple of rows are added on top so the margin cannot be short
    if OpenCV ever rounds differently.
    """
    return math.ceil(4.0 * sigma) + 2


def strip_rows(width: int) -> int:
    """How many rows to sharpen at once, from a fixed byte budget.

    Derived from the width so the working set is bounded by
    `SHARPEN_STRIP_BYTES` however wide the result is, rather than growing with
    it. The widest intermediate is the strip in float32 RGB, at 12 bytes a
    pixel.
    """
    if width <= 0:
        return SHARPEN_MIN_STRIP_ROWS
    return max(SHARPEN_MIN_STRIP_ROWS, SHARPEN_STRIP_BYTES // (width * 12))


def resize_to_target(
    image: np.ndarray[Any, Any], width: int, height: int
) -> tuple[np.ndarray[Any, Any], bool]:
    """Resample a neural result down to an exact requested size.

    Only ever downwards. The planner picks the smallest supported factor that
    reaches the target, so the neural result is always at least as large - and
    enlarging here would put interpolated pixels into a result that is supposed
    to be model-generated, which is the one thing the neural-only rule forbids.
    An upward request is a planning bug and is refused rather than performed.

    `INTER_AREA`, not Lanczos: it averages the pixels being discarded, which is
    what makes a downscale look clean rather than aliased. Lanczos is the right
    choice for the opposite direction, which never happens here.
    """
    import cv2

    current_height, current_width = image.shape[:2]
    if (current_width, current_height) == (width, height):
        return image, False

    if width > current_width or height > current_height:
        raise ValidationError(
            "The result could not be resized to the requested size.",
            technical=(
                f"refusing to enlarge {current_width}x{current_height} "
                f"to {width}x{height}; the planner should have chosen a larger factor"
            ),
        )

    resized: np.ndarray[Any, Any] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized, True


def unsharp_mask(
    image: np.ndarray[Any, Any], strength: float, *, scale: int = 4
) -> np.ndarray[Any, Any]:
    """Post-process sharpening. Not a model, and labelled as such in the UI.

    An unsharp mask with three changes from the plain form, each aimed at one
    of the ways plain unsharp masking looks artificial:

      * the radius follows the upscale factor (`sharpen_sigma`), so the filter
        works on structure the source actually resolved rather than on whatever
        the upsampling invented;
      * the high-frequency component is taken on **luma** and added equally to
        R, G and B. That moves each pixel along the neutral axis, so hue and
        saturation are preserved and no coloured fringe appears on an edge -
        which is what sharpening the three channels independently produces;
      * that component is shaped by a dead zone and a ceiling before it is
        added back, which is what keeps flat regions quiet and edges free of a
        pronounced rim.

    `strength` is 0..1 and keeps its existing meaning and range - the UI is
    unchanged. `scale` is the job's upscale factor.

    Deterministic: same input, same output, no randomness anywhere.

    Alpha is not sharpened. The pipeline splits it off before this is reached,
    and a four-channel array is handled here as well only so the function is
    correct on its own terms.
    """
    import cv2

    if not 0.0 <= strength <= 1.0:
        raise ValidationError(
            "Sharpening strength must be between 0 and 1.",
            technical=f"sharpen_strength={strength}",
        )

    # A true no-op, and the same array back rather than a copy: sharpening off
    # must not cost a round trip through float or change a single byte.
    if strength == 0:
        return image

    amount = strength * SHARPEN_MAX_AMOUNT
    sigma = sharpen_sigma(scale)
    height, width = image.shape[:2]

    # A copy, so alpha and the caller's array both survive untouched. Only the
    # three colour channels are written below.
    result = image.copy()
    weights = np.array(LUMA_WEIGHTS, dtype=np.float32)
    margin = blur_margin(sigma)
    rows = strip_rows(width)

    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        # Read the neighbouring rows the blur kernel reaches into, so a strip
        # boundary is not a discontinuity. Where this clamps, it clamps at the
        # real image edge and OpenCV's border handling is the same one the
        # whole-frame blur would have applied - which is what makes the result
        # identical to processing the image in one piece.
        read_top = max(0, top - margin)
        read_bottom = min(height, bottom + margin)

        block = image[read_top:read_bottom, :, :3].astype(np.float32)
        detail = block @ weights
        del block

        blurred = cv2.GaussianBlur(detail, (0, 0), sigma)
        detail -= blurred

        # Dead zone, then ceiling, written as two clips so the whole shaping
        # runs in place. `d - clip(d, -floor, floor)` is a soft threshold: it
        # is exactly zero inside the dead zone and shrinks everything outside
        # it by `floor`, sign intact and with no separate magnitude array.
        np.clip(detail, -SHARPEN_NOISE_FLOOR, SHARPEN_NOISE_FLOOR, out=blurred)
        detail -= blurred
        del blurred
        np.clip(detail, -SHARPEN_DETAIL_CEILING, SHARPEN_DETAIL_CEILING, out=detail)
        detail *= amount

        # Discard the margin: those rows exist only to give the kernel real
        # neighbours, and the strip below owns them.
        correction = detail[top - read_top : top - read_top + (bottom - top)]

        patch = image[top:bottom, :, :3].astype(np.float32)
        patch += correction[:, :, None]
        np.clip(patch, 0.0, 255.0, out=patch)
        result[top:bottom, :, :3] = patch.astype(np.uint8)

    return result


def _scaled_progress(
    on_progress: ProgressCallback | None, index: int, total: int
) -> ProgressCallback | None:
    """Map one pass's 0..1 onto its slice of the overall run."""
    if on_progress is None:
        return None

    def report(fraction: float) -> None:
        on_progress((index + fraction) / total)

    return report
