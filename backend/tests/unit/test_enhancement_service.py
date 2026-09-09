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
from app.core.exceptions import InsufficientMemoryError, OutputTooLargeError, ValidationError
from app.inference.upscaler import JobCancelledError, UpscaleReport
from app.services import enhancement_service
from app.services.enhancement_service import (
    LUMA_WEIGHTS,
    SHARPEN_DETAIL_CEILING,
    SHARPEN_MAX_AMOUNT,
    SHARPEN_MIN_STRIP_ROWS,
    SHARPEN_NOISE_FLOOR,
    SHARPEN_SIGMA_RANGE,
    SHARPEN_STRIP_BYTES,
    SUPPORTED_SCALES,
    EnhancementRequest,
    EnhancementService,
    attach_alpha,
    blur_margin,
    sharpen_sigma,
    split_alpha,
    strip_rows,
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


@pytest.mark.parametrize("scale", [1, 3, 6, 12, 32])
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


@pytest.mark.parametrize("scale", [2, 4, 8, 16])
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
    # 16x wants 12.0 and is held at the ceiling: past this radius the filter
    # works on nothing the source resolved, and the halo is all that grows.
    assert sharpen_sigma(16) == 6.0


def test_the_radius_is_clamped_at_both_ends() -> None:
    low, high = SHARPEN_SIGMA_RANGE

    assert sharpen_sigma(1) == low
    assert sharpen_sigma(64) == high


@pytest.mark.parametrize("scale", [2, 4, 8, 16])
def test_a_flat_region_is_left_alone_at_every_scale(scale: int) -> None:
    """Nothing to sharpen means nothing done - no drift, no grain."""
    source = grey(128)

    assert np.array_equal(unsharp_mask(source, 1.0, scale=scale), source)


@pytest.mark.parametrize("scale", [2, 4, 8, 16])
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


# --------------------------------------------- sharpening in bounded strips


def whole_frame_reference(
    image: np.ndarray[Any, Any], strength: float, scale: int
) -> np.ndarray[Any, Any]:
    """The sharpener as it was before strips, kept as an oracle.

    An 8x result reaches 195 MP, where this version asks OpenCV for a single
    780 MB block for the blur alone and 2.2 GiB for the float conversion before
    it. The strip form must produce exactly the same pixels, so it is checked
    against this rather than against itself.
    """
    import cv2

    amount = strength * SHARPEN_MAX_AMOUNT
    sigma = sharpen_sigma(scale)
    source = image[:, :, :3].astype(np.float32)
    detail = source @ np.array(LUMA_WEIGHTS, dtype=np.float32)

    blurred = cv2.GaussianBlur(detail, (0, 0), sigma)
    detail -= blurred
    np.clip(detail, -SHARPEN_NOISE_FLOOR, SHARPEN_NOISE_FLOOR, out=blurred)
    detail -= blurred
    del blurred
    np.clip(detail, -SHARPEN_DETAIL_CEILING, SHARPEN_DETAIL_CEILING, out=detail)
    detail *= amount

    source += detail[:, :, None]
    np.clip(source, 0.0, 255.0, out=source)
    result = source.astype(np.uint8)

    if image.shape[2] == 3:
        return result
    return np.dstack([result, image[:, :, 3]])


@pytest.fixture
def tiny_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force many strips on a small image.

    Without this the default budget swallows any test-sized image in one
    strip and the boundary logic - the only part that can be wrong - is never
    executed.
    """
    monkeypatch.setattr(enhancement_service, "SHARPEN_STRIP_BYTES", 1)
    monkeypatch.setattr(enhancement_service, "SHARPEN_MIN_STRIP_ROWS", 8)


@pytest.mark.parametrize("scale", [2, 4, 8, 16])
@pytest.mark.parametrize("channels", [3, 4])
def test_strips_are_bit_identical_to_whole_frame(
    tiny_strips: None, scale: int, channels: int
) -> None:
    """Every strip boundary must be invisible, at every radius.

    The margin is what makes this true: a strip reads the rows the kernel
    reaches into and discards them afterwards, so the blur sees the same
    neighbours it would have in one pass. 8x is the important case, because
    its sigma of 6 has the widest support and therefore the most to get wrong.
    """
    generator = np.random.default_rng(4)
    source = generator.integers(0, 256, (200, 160, channels), dtype=np.uint8)

    assert np.array_equal(
        unsharp_mask(source, 0.6, scale=scale),
        whole_frame_reference(source, 0.6, scale),
    )


def test_strips_are_identical_even_one_row_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The most adversarial split there is: a boundary at every row."""
    monkeypatch.setattr(enhancement_service, "SHARPEN_STRIP_BYTES", 1)
    monkeypatch.setattr(enhancement_service, "SHARPEN_MIN_STRIP_ROWS", 1)
    source = np.repeat(
        np.where((np.arange(120)[None, :] // 11) % 2 == 0, 20, 230)
        .repeat(90, 0)
        .astype(np.uint8)[:, :, None],
        3,
        axis=2,
    )

    assert np.array_equal(
        unsharp_mask(source, 1.0, scale=8),
        whole_frame_reference(source, 1.0, scale=8),
    )


def test_the_blur_margin_covers_the_kernel_support() -> None:
    """Measured, not assumed: a float Gaussian reaches exactly 4 sigma."""
    for scale in (2, 4, 8, 16):
        sigma = sharpen_sigma(scale)
        assert blur_margin(sigma) >= 4 * sigma


def test_the_working_set_is_bounded_by_the_budget_not_the_image() -> None:
    """The whole point: an 8x result must not cost more than a small one.

    The reported failure was a single 780 MB allocation for the blur of a
    195 MP frame. A strip's widest intermediate is float32 RGB at 12 bytes a
    pixel, and it has to stay inside the budget however wide the result is.
    """
    for width in (1280, 5600, 16128, 32000):
        rows = strip_rows(width)
        working_set = rows * width * 12
        assert rows >= SHARPEN_MIN_STRIP_ROWS
        if rows > SHARPEN_MIN_STRIP_ROWS:
            assert working_set <= SHARPEN_STRIP_BYTES, width


def test_an_eight_x_sized_frame_is_processed_in_many_strips() -> None:
    """Guards the guard: if this ever became one strip, the fix would be gone."""
    # 16128 px wide is the failing 8x output from the report.
    rows = strip_rows(16128)

    assert rows < 12096
    assert 12096 // rows > 10


def test_a_degenerate_width_still_yields_a_usable_strip() -> None:
    assert strip_rows(0) == SHARPEN_MIN_STRIP_ROWS
    assert strip_rows(1) >= SHARPEN_MIN_STRIP_ROWS


def test_the_callers_array_is_never_mutated(tiny_strips: None) -> None:
    """Strips write into a copy; the input must come back untouched."""
    generator = np.random.default_rng(9)
    source = generator.integers(0, 256, (120, 100, 3), dtype=np.uint8)
    original = source.copy()

    unsharp_mask(source, 0.8, scale=8)

    assert np.array_equal(source, original)


def test_alpha_survives_strip_processing(tiny_strips: None) -> None:
    colour = np.random.default_rng(2).integers(0, 256, (150, 120, 3), dtype=np.uint8)
    alpha = np.linspace(0, 255, 150 * 120, dtype=np.uint8).reshape(150, 120)
    source = np.dstack([colour, alpha])

    sharpened = unsharp_mask(source, 0.8, scale=8)

    assert sharpened.shape == source.shape
    assert np.array_equal(sharpened[:, :, 3], alpha)


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
    assert service.supported_scales("RealESRGAN_x4plus") == [4, 8, 16]
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


# --------------------------------------------------------- the 16x cascade


def cascading_service(**overrides: Any) -> EnhancementService:
    """A service on the real manifest with stub weights and tuned settings."""
    settings = Settings(environment="test", **overrides)
    models = ModelService(settings)
    return EnhancementService(settings, StubManager(models), models)  # type: ignore[arg-type]


def test_sixteen_times_is_two_four_times_passes(service: EnhancementService) -> None:
    """There is no 16x weight, so it is composed - and composed of the largest
    passes available, rather than of three smaller ones."""
    assert service.plan("RealESRGAN_x4plus", 16) == ["RealESRGAN_x4plus", "RealESRGAN_x4plus"]


def test_the_sixteen_times_cascade_is_the_same_model_twice(
    service: EnhancementService,
) -> None:
    """Handing the second pass to a different network would change the look of
    the result halfway through it."""
    stages = service.plan_cascade("realesr-general-x4v3", 16, 100, 80)

    assert [stage.model_id for stage in stages] == ["realesr-general-x4v3"] * 2
    assert [stage.scale for stage in stages] == [4, 4]


def test_every_four_times_model_can_reach_sixteen(service: EnhancementService) -> None:
    for model_id in ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-general-x4v3"):
        assert service.plan(model_id, 16) == [model_id, model_id]


def test_a_two_times_model_cannot_reach_sixteen(service: EnhancementService) -> None:
    """Eight 2x passes is not a cascade, it is a different product."""
    with pytest.raises(ValidationError, match="cannot produce a 16x result"):
        service.plan("RealESRGAN_x2plus", 16)

    assert 16 not in service.supported_scales("RealESRGAN_x2plus")


def test_required_models_for_sixteen_are_the_one_model(service: EnhancementService) -> None:
    """Both passes share weights, so only one file has to be on disk."""
    required = [status.entry.id for status in service.required_models("RealESRGAN_x4plus", 16)]

    assert required == ["RealESRGAN_x4plus", "RealESRGAN_x4plus"]


# ------------------------------------------------------- cascade arithmetic


def test_a_cascade_carries_the_size_each_pass_will_see(
    service: EnhancementService,
) -> None:
    stages = service.plan_cascade("RealESRGAN_x4plus", 16, 1080, 720)

    assert [(s.input_width, s.input_height) for s in stages] == [(1080, 720), (4320, 2880)]
    assert [(s.output_width, s.output_height) for s in stages] == [(4320, 2880), (17280, 11520)]


def test_a_cascade_numbers_its_stages(service: EnhancementService) -> None:
    stages = service.plan_cascade("RealESRGAN_x4plus", 16, 100, 100)

    assert [(s.index, s.total) for s in stages] == [(0, 2), (1, 2)]
    assert [s.is_final for s in stages] == [False, True]


def test_a_single_pass_cascade_is_one_final_stage(service: EnhancementService) -> None:
    stages = service.plan_cascade("RealESRGAN_x4plus", 4, 100, 50)

    assert len(stages) == 1
    assert stages[0].is_final
    assert (stages[0].output_width, stages[0].output_height) == (400, 200)


def test_the_eight_times_cascade_is_unchanged(service: EnhancementService) -> None:
    """Backward compatibility: 8x is still 4x then 2x, on the same sizes."""
    stages = service.plan_cascade("RealESRGAN_x4plus", 8, 500, 400)

    assert [s.model_id for s in stages] == ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]
    assert [s.scale for s in stages] == [4, 2]
    assert (stages[-1].output_width, stages[-1].output_height) == (4000, 3200)


def test_a_cascade_never_disagrees_with_the_pass_plan(service: EnhancementService) -> None:
    """The two describe the same journey because one calls the other."""
    for model_id in ("RealESRGAN_x4plus", "RealESRGAN_x2plus", "realesr-general-x4v3"):
        for scale in service.supported_scales(model_id):
            stages = service.plan_cascade(model_id, scale, 64, 64)
            assert [s.model_id for s in stages] == service.plan(model_id, scale)
            # And the composition really does multiply out to the request.
            assert stages[-1].output_width == 64 * scale


def test_an_unreachable_scale_is_refused_before_any_arithmetic(
    service: EnhancementService,
) -> None:
    with pytest.raises(ValidationError):
        service.plan_cascade("RealESRGAN_x2plus", 16, 100, 100)


# -------------------------------------------------------------- running 16x


def test_sixteen_times_chains_the_first_pass_into_the_second(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    result = service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert result.image.shape == (192, 256, 3)
    calls = manager.issued["RealESRGAN_x4plus"].calls
    assert [call["shape"] for call in calls] == [(12, 16, 3), (48, 64, 3)]


def test_sixteen_times_is_two_recorded_passes(service: EnhancementService) -> None:
    result = service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert len(result.passes) == 2
    assert len(result.reports) == 2


def test_nothing_is_interpolated_to_reach_sixteen(service: EnhancementService) -> None:
    """The whole size comes from the two passes; no resample stands in for one."""
    result = service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert result.resized is False
    assert result.scale == 16
    assert result.image.shape[:2] == (12 * 16, 16 * 16)


def test_sixteen_times_keeps_alpha_out_of_the_network(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    result = service.enhance(image(8, 6, channels=4), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert result.image.shape == (96, 128, 4)
    # Three channels at every pass, never four.
    assert all(call["shape"][2] == 3 for call in manager.issued["RealESRGAN_x4plus"].calls)


def test_sixteen_times_sharpens_at_the_clamped_radius(
    service: EnhancementService,
) -> None:
    result = service.enhance(
        image(), EnhancementRequest("RealESRGAN_x4plus", 16, sharpen_strength=0.5)
    )

    assert result.sharpened


def test_progress_is_monotonic_and_complete_across_a_sixteen_times_cascade(
    service: EnhancementService,
) -> None:
    """A bar that restarts at the halfway point misreports the work left."""
    seen: list[float] = []

    service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 16), on_progress=seen.append)

    assert seen == sorted(seen)
    assert all(0.0 <= value <= 1.0 for value in seen)
    # Both passes are represented: one below the midpoint and one above it.
    assert any(value < 0.5 for value in seen)
    assert any(value > 0.5 for value in seen)
    assert seen[-1] == pytest.approx(1.0)


def test_the_second_pass_never_reports_less_than_the_first(
    service: EnhancementService,
) -> None:
    """Monotonic across the boundary, not merely within each pass."""
    seen: list[float] = []

    service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 16), on_progress=seen.append)

    first_pass = [value for value in seen if value <= 0.5]
    second_pass = [value for value in seen if value > 0.5]

    assert first_pass and second_pass
    assert max(first_pass) <= min(second_pass)


# ------------------------------------------- stopping between cascade passes


def test_a_cascade_stops_between_passes_when_cancelled(
    service: EnhancementService,
) -> None:
    """Cancellation lands at the pass boundary, before the second pass loads
    weights or allocates anything for a result nobody is waiting for."""
    manager: StubManager = service._manager  # type: ignore[assignment]
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        # False before the first pass, True before the second.
        return checks > 1

    with pytest.raises(JobCancelledError):
        service.enhance(
            image(),
            EnhancementRequest("RealESRGAN_x4plus", 16),
            should_cancel=should_cancel,
        )

    assert len(manager.issued["RealESRGAN_x4plus"].calls) == 1


def test_a_cascade_cancelled_before_it_starts_runs_nothing(
    service: EnhancementService,
) -> None:
    manager: StubManager = service._manager  # type: ignore[assignment]

    with pytest.raises(JobCancelledError):
        service.enhance(
            image(),
            EnhancementRequest("RealESRGAN_x4plus", 16),
            should_cancel=lambda: True,
        )

    assert manager.issued == {}


def test_a_failing_second_pass_fails_the_job_rather_than_returning_the_first(
    service: EnhancementService,
) -> None:
    """The worst possible outcome would be a 4x image presented as a 16x one."""
    manager: StubManager = service._manager  # type: ignore[assignment]
    upscaler = manager.get("RealESRGAN_x4plus")
    original = upscaler.upscale
    calls = 0

    def failing(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InsufficientMemoryError("there is not enough memory for the second pass")
        return original(*args, **kwargs)

    upscaler.upscale = failing  # type: ignore[method-assign]

    with pytest.raises(InsufficientMemoryError):
        service.enhance(image(), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert calls == 2


# ------------------------------------------------- the per-stage size limit


def test_a_stage_beyond_the_output_limit_is_refused() -> None:
    """A cascade that cannot finish safely says so instead of finishing badly."""
    service = cascading_service(max_output_pixels=10_000)

    with pytest.raises(OutputTooLargeError) as caught:
        service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert caught.value.context["stage"] == 2
    assert caught.value.context["stages"] == 2
    assert caught.value.context["projectedPixels"] == 256 * 192


def test_the_refusal_arrives_before_the_oversized_pass_runs() -> None:
    """16x12 -> 64x48 fits inside 10 000 px; 64x48 -> 256x192 does not."""
    service = cascading_service(max_output_pixels=10_000)
    manager: StubManager = service._manager  # type: ignore[assignment]

    with pytest.raises(OutputTooLargeError):
        service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert len(manager.issued["RealESRGAN_x4plus"].calls) == 1


def test_the_limit_refuses_the_very_first_pass_when_that_is_the_problem() -> None:
    service = cascading_service(max_output_pixels=1_000)
    manager: StubManager = service._manager  # type: ignore[assignment]

    with pytest.raises(OutputTooLargeError) as caught:
        service.enhance(image(16, 12), EnhancementRequest("RealESRGAN_x4plus", 16))

    assert caught.value.context["stage"] == 1
    assert manager.issued == {}


def test_the_stage_limit_never_fires_for_a_job_that_passed_submission(
    service: EnhancementService,
) -> None:
    """Every earlier stage is smaller than the last, and the last is what
    `assert_output_fits` already checked - so this guard is defence in depth
    and must never reject something the API accepted."""
    settings = get_settings()

    for scale in SUPPORTED_SCALES:
        for model_id in ("RealESRGAN_x4plus", "RealESRGAN_x2plus"):
            if scale not in service.supported_scales(model_id):
                continue
            # A source the submission check would accept at this factor.
            assert 1080 * 720 * scale * scale <= settings.max_output_pixels
            stages = service.plan_cascade(model_id, scale, 1080, 720)
            assert all(stage.output_pixels <= settings.max_output_pixels for stage in stages), scale


def test_an_intermediate_is_always_smaller_than_the_result(
    service: EnhancementService,
) -> None:
    """Which is why checking the final size at submission is sufficient."""
    for scale in (8, 16):
        stages = service.plan_cascade("RealESRGAN_x4plus", scale, 640, 480)
        pixels = [stage.output_pixels for stage in stages]

        assert pixels == sorted(pixels)
        assert pixels[-1] == 640 * 480 * scale * scale


def test_the_sixteen_times_intermediate_is_cheaper_than_the_eight_times_one(
    service: EnhancementService,
) -> None:
    """At the same output size a 16x cascade holds *less*, not more: its
    intermediate is a sixteenth of the result where 8x's is a quarter.

    This is the memory argument for composing 16x as 4x -> 4x, and it is why
    the existing limit needs no loosening to accommodate it.
    """
    # Two jobs that land on exactly the same 17280x11520 result, just under
    # the 200 MP ceiling, reached by different routes.
    eight = service.plan_cascade("RealESRGAN_x4plus", 8, 2160, 1440)
    sixteen = service.plan_cascade("RealESRGAN_x4plus", 16, 1080, 720)

    assert eight[-1].output_pixels == sixteen[-1].output_pixels == 17280 * 11520
    # 49.8 MP against 12.4 MP: a quarter of the intermediate, for the same result.
    assert sixteen[0].output_pixels * 4 == eight[0].output_pixels
