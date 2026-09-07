"""The enhancement pipeline: pass planning, alpha handling and post-processing.

Pass planning and the image helpers are pure, so they are tested directly. The
running of the pipeline is checked against a stub manager, which keeps these
fast and makes the multi-pass wiring visible; the real weights run in
tests/integration/test_real_inference.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.core.exceptions import ValidationError
from app.inference.upscaler import UpscaleReport
from app.services.enhancement_service import (
    EnhancementRequest,
    EnhancementService,
    attach_alpha,
    split_alpha,
    unsharp_mask,
)
from app.services.model_service import ModelService


class StubUpscaler:
    """Doubles or quadruples an image without loading any weights."""

    def __init__(self, model_id: str, scale: int) -> None:
        self.id = model_id
        self.scale = scale
        self.arch = "stub"
        self.calls: list[dict[str, Any]] = []
        self.last_report: UpscaleReport | None = None

    def upscale(
        self,
        image: np.ndarray[Any, Any],
        *,
        tile: int,
        tile_pad: int,
        on_progress: Any = None,
        should_cancel: Any = None,
    ) -> np.ndarray[Any, Any]:
        self.calls.append({"shape": image.shape, "tile": tile, "tile_pad": tile_pad})
        if on_progress is not None:
            on_progress(0.5)
            on_progress(1.0)
        self.last_report = UpscaleReport(device="cpu", tile_size=tile, tiles=1, fp16=False)
        return np.repeat(np.repeat(image, self.scale, axis=0), self.scale, axis=1)


class StubManager:
    """Hands out stub upscalers and records what was asked for."""

    def __init__(self, models: ModelService) -> None:
        self._models = models
        self.requested: list[tuple[str, float | None]] = []
        self.issued: dict[str, StubUpscaler] = {}
        self.released = 0

    def get(self, model_id: str, *, denoise_strength: float | None = None) -> StubUpscaler:
        self.requested.append((model_id, denoise_strength))
        scale = self._models.get(model_id).entry.scale
        upscaler = self.issued.setdefault(model_id, StubUpscaler(model_id, scale))
        return upscaler

    def release(self, model_id: str | None = None) -> int:
        self.released += 1
        return len(self.issued)


@pytest.fixture
def service() -> EnhancementService:
    """A service on the real manifest with a stubbed model manager."""
    settings = get_settings()
    models = ModelService(settings)
    return EnhancementService(settings, StubManager(models), models)  # type: ignore[arg-type]


def image(width: int = 16, height: int = 12, channels: int = 3) -> np.ndarray[Any, Any]:
    generator = np.random.default_rng(2)
    return generator.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


# ------------------------------------------------------------------- planning


def test_a_native_scale_is_a_single_pass(service: EnhancementService) -> None:
    assert service.plan("RealESRGAN_x4plus", 4) == ["RealESRGAN_x4plus"]
    assert service.plan("RealESRGAN_x2plus", 2) == ["RealESRGAN_x2plus"]


def test_eight_times_is_two_neural_passes(service: EnhancementService) -> None:
    """There is no 8x weight, and a resampled second step would make half the
    result non-neural."""
    assert service.plan("RealESRGAN_x4plus", 8) == ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]


def test_every_four_times_model_can_reach_eight(service: EnhancementService) -> None:
    for model_id in ("RealESRGAN_x4plus_anime_6B", "realesr-general-x4v3"):
        assert service.plan(model_id, 8)[0] == model_id


def test_asking_a_four_times_model_for_two_times_is_refused_with_a_suggestion(
    service: EnhancementService,
) -> None:
    """Silently swapping in a different model would change the look of the
    result without saying so; resampling afterwards would not be neural."""
    with pytest.raises(ValidationError) as caught:
        service.plan("RealESRGAN_x4plus", 2)

    assert "cannot produce a 2x result" in caught.value.message
    assert caught.value.context["suggestion"] == "RealESRGAN_x2plus"


@pytest.mark.parametrize("scale", [1, 3, 6, 16])
def test_an_unavailable_scale_is_refused(service: EnhancementService, scale: int) -> None:
    with pytest.raises(ValidationError, match="not one of the available"):
        service.plan("RealESRGAN_x4plus", scale)


def test_required_models_include_the_denoise_pair(service: EnhancementService) -> None:
    """Both halves of a DNI pair have to be on disk before the job starts."""
    required = [status.entry.id for status in service.required_models("realesr-general-x4v3", 4)]

    assert required == ["realesr-general-x4v3", "realesr-general-wdn-x4v3"]


def test_required_models_cover_every_pass(service: EnhancementService) -> None:
    required = [status.entry.id for status in service.required_models("RealESRGAN_x4plus", 8)]

    assert required == ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]


# -------------------------------------------------------------------- running


def test_a_single_pass_runs_one_model(service: EnhancementService) -> None:
    result = service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 4))

    assert result.image.shape == (48, 64, 3)
    assert result.passes == ["RealESRGAN_x4plus"]


def test_eight_times_chains_the_output_of_the_first_pass_into_the_second(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    result = service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 8))

    assert result.image.shape == (96, 128, 3)
    second = manager.issued["RealESRGAN_x2plus"]
    # The 2x model saw the 4x result, not the original.
    assert second.calls[0]["shape"] == (48, 64, 3)


def test_progress_is_continuous_across_two_passes(service: EnhancementService) -> None:
    """Restarting the bar at the halfway point would misreport the work left."""
    seen: list[float] = []

    service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 8), on_progress=seen.append)

    assert seen == sorted(seen)
    assert seen[0] < 0.5 < seen[-1]
    assert seen[-1] == pytest.approx(1.0)


def test_the_configured_tile_settings_reach_the_upscaler(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 4))

    call = manager.issued["RealESRGAN_x4plus"].calls[0]
    settings = get_settings()
    assert call["tile"] == settings.tile_size
    assert call["tile_pad"] == settings.tile_pad


def test_denoise_is_passed_only_to_the_model_that_supports_it(
    service: EnhancementService,
) -> None:
    """The 2x second pass has no denoise pair, so forwarding the setting there
    would fail the load."""
    manager: StubManager = service._manager  # type: ignore[assignment]

    service.enhance(image(), EnhancementRequest("realesr-general-x4v3", 8, denoise_strength=0.4))

    assert manager.requested == [("realesr-general-x4v3", 0.4), ("RealESRGAN_x2plus", None)]


def test_no_denoise_setting_leaves_every_model_unblended(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    service.enhance(image(), EnhancementRequest("realesr-general-x4v3", 4))

    assert manager.requested == [("realesr-general-x4v3", None)]


def test_the_result_reports_where_it_ran(service: EnhancementService) -> None:
    result = service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 4))

    assert result.device == "cpu"
    assert result.fell_back_to_cpu is False
    assert result.tile_size_reduced is False


# ---------------------------------------------------------------------- alpha


def test_alpha_is_split_off_before_inference() -> None:
    rgba = image(channels=4)

    colour, alpha = split_alpha(rgba)

    assert colour.shape == (12, 16, 3)
    assert alpha is not None and alpha.shape == (12, 16)


def test_an_rgb_image_has_no_alpha_to_split() -> None:
    colour, alpha = split_alpha(image())

    assert alpha is None
    assert colour.shape == (12, 16, 3)


def test_greyscale_is_expanded_to_three_channels() -> None:
    """A three-channel network cannot take a single-channel image."""
    grey = np.zeros((8, 8), dtype=np.uint8)

    colour, alpha = split_alpha(grey)

    assert colour.shape == (8, 8, 3)
    assert alpha is None


def test_an_unsupported_channel_count_is_refused() -> None:
    with pytest.raises(ValidationError, match="could not be prepared"):
        split_alpha(np.zeros((8, 8, 5), dtype=np.uint8))


def test_alpha_is_resampled_to_the_enhanced_size() -> None:
    colour = np.zeros((32, 32, 3), dtype=np.uint8)
    alpha = np.full((8, 8), 128, dtype=np.uint8)

    combined = attach_alpha(colour, alpha)

    assert combined.shape == (32, 32, 4)
    # A flat mask stays flat: Lanczos on a constant is that constant.
    assert set(np.unique(combined[..., 3])) == {128}


def test_an_rgba_image_comes_back_as_rgba(service: EnhancementService) -> None:
    result = service.enhance(image(channels=4), EnhancementRequest("RealESRGAN_x4plus", 4))

    assert result.image.shape == (48, 64, 4)


# ----------------------------------------------------------------- sharpening


def test_sharpening_is_off_by_default(service: EnhancementService) -> None:
    result = service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 4))

    assert result.sharpened is False


def test_sharpening_runs_when_asked_for(service: EnhancementService) -> None:
    result = service.enhance(
        image(), EnhancementRequest("RealESRGAN_x4plus", 4, sharpen_strength=0.5)
    )

    assert result.sharpened is True


def test_zero_strength_returns_the_image_untouched() -> None:
    source = image()

    assert np.array_equal(unsharp_mask(source, 0.0), source)


def test_sharpening_increases_local_contrast() -> None:
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[:, 16:] = 200  # one hard edge

    sharpened = unsharp_mask(source, 1.0)

    edge_before = int(source[16, 17, 0]) - int(source[16, 14, 0])
    edge_after = int(sharpened[16, 17, 0]) - int(sharpened[16, 14, 0])
    assert edge_after > edge_before


def test_sharpening_does_not_wrap_the_highlights() -> None:
    """Clipping in float before the cast is what stops white going black."""
    source = np.full((16, 16, 3), 250, dtype=np.uint8)
    source[8, 8] = 255

    sharpened = unsharp_mask(source, 1.0)

    assert sharpened.max() <= 255
    assert sharpened.min() >= 0


@pytest.mark.parametrize("strength", [-0.5, 1.5])
def test_an_out_of_range_sharpen_strength_is_refused(strength: float) -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        unsharp_mask(image(), strength)


# --------------------------------------------------------------------- misc


def test_release_frees_the_models(service: EnhancementService) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    service.release()

    assert manager.released == 1


def test_a_service_can_be_constructed_from_settings_alone() -> None:
    """The default wiring must work, or nothing outside the tests can use it."""
    service = EnhancementService(Settings(environment="test"))

    assert service.plan("RealESRGAN_x4plus", 4) == ["RealESRGAN_x4plus"]
