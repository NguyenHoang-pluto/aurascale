"""The enhancement pipeline, from a decoded image to an enhanced one.

This is stages 6-11 of docs/image-processing.md: alpha split, the neural
pass or passes, the optional post-process, and alpha recomposition. Decoding,
encoding and persistence are the caller's business, and nothing here knows what
a job or an HTTP request is.

Everything a user can switch on either runs a trained network or is labelled as
a post-process. There is no control here that pretends to be AI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.inference.model_manager import ModelManager
from app.inference.upscaler import UpscaleReport
from app.services.model_service import ModelService, ModelStatus

logger = get_logger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

# The model that supplies the second pass of an 8x job. Two neural passes,
# so every pixel of the result is still model-generated (docs § 5).
SECOND_PASS_MODEL = "RealESRGAN_x2plus"

SUPPORTED_SCALES = (2, 4, 8)

# Unsharp masking parameters. Radius is in output pixels; the strength the user
# sets scales the amount of the high-frequency component added back.
SHARPEN_RADIUS = 3
SHARPEN_MAX_AMOUNT = 1.5


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


@dataclass(frozen=True, slots=True)
class EnhancementResult:
    """The enhanced image plus what it took to produce it."""

    image: np.ndarray[Any, Any]
    scale: int
    passes: list[str]
    reports: list[UpscaleReport] = field(default_factory=list)
    sharpened: bool = False

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
        plan = self.plan(request.model_id, request.scale)
        reports: list[UpscaleReport] = []

        current = colour
        for index, model_id in enumerate(plan):
            upscaler = self._manager.get(
                model_id,
                denoise_strength=self._denoise_for(model_id, request.denoise_strength),
            )

            current = upscaler.upscale(
                current,
                tile=(
                    request.tile_size if request.tile_size is not None else self._settings.tile_size
                ),
                tile_pad=(
                    request.tile_pad if request.tile_pad is not None else self._settings.tile_pad
                ),
                on_progress=_scaled_progress(on_progress, index, len(plan)),
                should_cancel=should_cancel,
            )

            report = upscaler.last_report
            if report is not None:
                reports.append(report)

            logger.info(
                "pass complete",
                extra={
                    "model": model_id,
                    "pass": index + 1,
                    "passes": len(plan),
                    "device": report.device if report else None,
                    "tile": report.tile_size if report else None,
                },
            )

        sharpened = request.sharpen_strength > 0
        if sharpened:
            current = unsharp_mask(current, request.sharpen_strength)

        if alpha is not None:
            current = attach_alpha(current, alpha)

        return EnhancementResult(
            image=current,
            scale=request.scale,
            passes=plan,
            reports=reports,
            sharpened=sharpened,
        )

    def _denoise_for(self, model_id: str, strength: float | None) -> float | None:
        """Apply the denoise setting only to the model that supports it.

        The second pass of an 8x job is a different network with no denoise
        pair, so passing the strength on would fail the load.
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


def unsharp_mask(image: np.ndarray[Any, Any], strength: float) -> np.ndarray[Any, Any]:
    """Post-process sharpening. Not a model, and labelled as such in the UI.

    `strength` is 0..1 and scales the amount of the high-frequency component
    added back. Everything is done in float and clipped once, so the highlights
    do not wrap to black at high strengths.
    """
    import cv2

    if not 0.0 <= strength <= 1.0:
        raise ValidationError(
            "Sharpening strength must be between 0 and 1.",
            technical=f"sharpen_strength={strength}",
        )

    if strength == 0:
        return image

    amount = strength * SHARPEN_MAX_AMOUNT
    source = image.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), SHARPEN_RADIUS)
    sharpened = source * (1.0 + amount) - blurred * amount

    return np.clip(sharpened, 0.0, 255.0).astype(np.uint8)


def _scaled_progress(
    on_progress: ProgressCallback | None, index: int, total: int
) -> ProgressCallback | None:
    """Map one pass's 0..1 onto its slice of the overall run."""
    if on_progress is None:
        return None

    def report(fraction: float) -> None:
        on_progress((index + fraction) / total)

    return report
