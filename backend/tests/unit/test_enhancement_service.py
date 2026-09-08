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
    SHARPEN_DETAIL_CEILING,
    SHARPEN_MAX_AMOUNT,
    SHARPEN_SIGMA_RANGE,
    SUPPORTED_SCALES,
    EnhancementRequest,
    EnhancementService,
    attach_alpha,
    sharpen_sigma,
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


# ------------------------------------------------------ adaptive sharpening


def grey(value: float, size: int = 128) -> np.ndarray[Any, Any]:
    """A flat RGB field."""
    return np.repeat(np.full((size, size), value, np.uint8)[:, :, None], 3, axis=2)


def textured(sigma: float, size: int = 128, seed: int = 0) -> np.ndarray[Any, Any]:
    """Mid-grey plus gaussian variation of a known amplitude."""
    generator = np.random.default_rng(seed)
    field = np.clip(128 + generator.normal(0, sigma, (size, size)), 0, 255).astype(np.uint8)
    return np.repeat(field[:, :, None], 3, axis=2)


def hard_edge(size: int = 128) -> np.ndarray[Any, Any]:
    """One strong vertical step - where halos appear if they are going to."""
    field = np.full((size, size), 40, np.uint8)
    field[:, size // 2 :] = 200
    return np.repeat(field[:, :, None], 3, axis=2)


def mean_change(before: np.ndarray[Any, Any], after: np.ndarray[Any, Any]) -> float:
    return float(np.abs(after.astype(int) - before.astype(int)).mean())


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_zero_strength_is_a_true_no_op_at_every_scale(scale: int) -> None:
    """The most important property: off means untouched, not "almost"."""
    source = textured(12.0)

    result = unsharp_mask(source, 0.0, scale=scale)

    assert np.array_equal(result, source)
    # The same array, not a copy that happens to be equal.
    assert result is source


def test_the_radius_follows_the_upscale_factor() -> None:
    """A fixed radius sharpens a different thing at every scale."""
    assert sharpen_sigma(2) == 1.5
    # 4x is the anchor: exactly the sigma that shipped before.
    assert sharpen_sigma(4) == 3.0
    assert sharpen_sigma(8) == 6.0


def test_the_radius_is_clamped_at_both_ends() -> None:
    low, high = SHARPEN_SIGMA_RANGE

    assert sharpen_sigma(1) == low
    assert sharpen_sigma(64) == high


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_a_flat_region_is_left_alone_at_every_scale(scale: int) -> None:
    """Nothing to sharpen means nothing done - no drift, no grain."""
    source = grey(128)

    assert np.array_equal(unsharp_mask(source, 1.0, scale=scale), source)


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_noise_in_a_flat_region_is_not_amplified_like_texture(scale: int) -> None:
    """The dead zone is what separates sensor noise from real detail.

    Both images are stochastic; only the amplitude differs. Sharpening that
    treated them alike would turn a clean sky into visible grain.
    """
    noise = textured(3.0)
    texture = textured(14.0)

    noise_change = mean_change(noise, unsharp_mask(noise, 1.0, scale=scale))
    texture_change = mean_change(texture, unsharp_mask(texture, 1.0, scale=scale))

    assert texture_change > noise_change * 3


def test_fine_texture_gains_local_contrast() -> None:
    source = textured(14.0)

    sharpened = unsharp_mask(source, 1.0, scale=4)

    assert float(sharpened.std()) > float(source.std())


def test_the_halo_on_a_strong_edge_is_bounded() -> None:
    """The ceiling exists precisely to make this bound provable.

    Overshoot cannot exceed `ceiling * max_amount` whatever the edge does, so
    a hard step cannot be driven to black and white the way plain unsharp
    masking drives it.
    """
    source = hard_edge()

    row_before = source[64, :, 0].astype(int)
    row_after = unsharp_mask(source, 1.0, scale=4)[64, :, 0].astype(int)

    overshoot = max(
        row_after.max() - row_before.max(),
        row_before.min() - row_after.min(),
    )
    assert 0 < overshoot <= SHARPEN_DETAIL_CEILING * SHARPEN_MAX_AMOUNT


def test_a_strong_edge_is_not_driven_to_the_ends_of_the_range() -> None:
    """Plain unsharp masking clips a 40/200 step to 0 and 255. This must not."""
    sharpened = unsharp_mask(hard_edge(), 1.0, scale=4)

    assert sharpened.min() > 0
    assert sharpened.max() < 255


def test_colour_relationships_are_preserved() -> None:
    """The correction is luma-based and added equally to R, G and B.

    Sharpening the channels independently pulls them apart at an edge, which
    is what a coloured fringe is.
    """
    source = np.zeros((64, 64, 3), np.uint8)
    source[:, :32] = (60, 90, 140)
    source[:, 32:] = (180, 150, 100)

    sharpened = unsharp_mask(source, 1.0, scale=4)

    before = source.astype(int)
    after = sharpened.astype(int)
    # Every channel moved by the same amount, so hue is untouched.
    delta = after - before
    assert np.array_equal(delta[:, :, 0], delta[:, :, 1])
    assert np.array_equal(delta[:, :, 1], delta[:, :, 2])


def test_an_rgba_image_keeps_its_alpha_untouched() -> None:
    """Alpha is never sharpened - the pipeline splits it, and so does this."""
    colour = textured(14.0, size=64)
    alpha = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
    source = np.dstack([colour, alpha])

    sharpened = unsharp_mask(source, 1.0, scale=4)

    assert sharpened.shape == source.shape
    assert np.array_equal(sharpened[:, :, 3], alpha)
    assert not np.array_equal(sharpened[:, :, :3], colour)


def test_rgb_input_returns_three_channels() -> None:
    sharpened = unsharp_mask(textured(14.0, size=64), 1.0, scale=4)

    assert sharpened.shape[2] == 3
    assert sharpened.dtype == np.uint8


def test_sharpening_is_deterministic() -> None:
    source = textured(14.0)

    first = unsharp_mask(source, 0.7, scale=4)
    second = unsharp_mask(source, 0.7, scale=4)

    assert np.array_equal(first, second)


def test_a_stronger_setting_changes_more_than_a_weaker_one() -> None:
    """The existing 0..1 UI semantics still mean what they meant."""
    source = textured(14.0)

    gentle = mean_change(source, unsharp_mask(source, 0.25, scale=4))
    firm = mean_change(source, unsharp_mask(source, 1.0, scale=4))

    assert firm > gentle > 0


# --------------------------------------------------------------------- misc


def test_release_frees_the_models(service: EnhancementService) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    service.release()

    assert manager.released == 1


def test_a_service_can_be_constructed_from_settings_alone() -> None:
    """The default wiring must work, or nothing outside the tests can use it."""
    service = EnhancementService(Settings(environment="test"))

    assert service.plan("RealESRGAN_x4plus", 4) == ["RealESRGAN_x4plus"]


# ------------------------------------------------------------- capabilities


def test_supported_scales_are_derived_from_the_pass_planner(
    service: EnhancementService,
) -> None:
    """Published capability and job validation must agree, so both come from
    `plan` rather than from two copies of the same rule."""
    assert service.supported_scales("RealESRGAN_x4plus") == [4, 8]
    assert service.supported_scales("RealESRGAN_x2plus") == [2]


def test_every_published_scale_actually_plans(service: EnhancementService) -> None:
    for model_id in ("RealESRGAN_x4plus", "RealESRGAN_x2plus", "realesr-general-x4v3"):
        for scale in service.supported_scales(model_id):
            assert service.plan(model_id, scale)


def test_a_scale_that_is_not_published_is_refused(service: EnhancementService) -> None:
    """The two answers are the same answer: anything absent from the list is
    rejected by the planner."""
    published = service.supported_scales("RealESRGAN_x4plus")

    for scale in SUPPORTED_SCALES:
        if scale in published:
            continue
        with pytest.raises(ValidationError):
            service.plan("RealESRGAN_x4plus", scale)
