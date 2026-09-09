"""Planning a target resolution.

A target and a factor are different questions, and the whole point of this
planner is that it never confuses them. These check that the aspect ratio
survives, that the smallest sufficient factor is chosen, that nothing is ever
enlarged past what was asked for, and that an impossible request is refused
before any compute is spent on it.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import OutputTooLargeError, ValidationError
from app.models.enums import TargetResolution
from app.services.resolution_planner import (
    ResolutionPlan,
    plan_resolution,
    target_dimensions,
)

#: Every factor the registry offers, across all its models. No single model
#: offers all of them - a 4x model reaches [4, 8, 16] and a 2x model only
#: [2] - so the per-model cases below pass their own list.
SCALES = [2, 4, 8, 16]
#: The production limit.
LIMIT = 200_000_000


def plan(width: int, height: int, preset: TargetResolution, **overrides: object) -> ResolutionPlan:
    kwargs: dict[str, object] = {"supported_scales": SCALES, "max_output_pixels": LIMIT}
    kwargs.update(overrides)
    return plan_resolution(width, height, preset, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------ preset values


@pytest.mark.parametrize(
    ("preset", "long_edge"),
    [
        (TargetResolution.TWO_K, 1920),
        (TargetResolution.FOUR_K, 3840),
        (TargetResolution.SIX_K, 5760),
        (TargetResolution.EIGHT_K, 7680),
    ],
)
def test_each_preset_names_its_long_edge(preset: TargetResolution, long_edge: int) -> None:
    assert preset.long_edge == long_edge
    assert str(long_edge) in preset.label


# ------------------------------------------------------------- aspect ratio


def test_a_landscape_image_puts_the_target_on_its_width() -> None:
    assert target_dimensions(1600, 900, TargetResolution.FOUR_K) == (3840, 2160)


def test_a_portrait_image_puts_the_target_on_its_height() -> None:
    """The preset fixes the long edge, whichever edge that is."""
    assert target_dimensions(900, 1600, TargetResolution.FOUR_K) == (2160, 3840)


def test_a_square_image_stays_square() -> None:
    assert target_dimensions(1000, 1000, TargetResolution.TWO_K) == (1920, 1920)


def test_an_unusual_ratio_is_not_forced_into_a_standard_frame() -> None:
    """4K is 3840 on the long edge, not 3840x2160 for everything."""
    width, height = target_dimensions(3000, 1000, TargetResolution.FOUR_K)

    assert width == 3840
    assert height == 1280
    assert width / height == pytest.approx(3.0)


def test_the_aspect_ratio_survives_planning() -> None:
    result = plan(1000, 625, TargetResolution.FOUR_K)

    assert result.target_width / result.target_height == pytest.approx(1000 / 625, rel=1e-3)


def test_degenerate_dimensions_are_refused() -> None:
    with pytest.raises(ValidationError, match="not usable"):
        target_dimensions(0, 100, TargetResolution.TWO_K)


# ------------------------------------------------------------ scale choice


def test_the_smallest_sufficient_factor_is_chosen() -> None:
    """1000 px needs 1.92x for 2K, so 2x - not 4x, which would be waste."""
    result = plan(1000, 800, TargetResolution.TWO_K)

    assert result.neural_scale == 2
    assert (result.neural_width, result.neural_height) == (2000, 1600)


def test_a_non_integer_requirement_rounds_up_to_a_supported_factor() -> None:
    """1500 px needs 2.56x for 4K. There is no 3x, so 4x it is."""
    result = plan(1500, 1000, TargetResolution.FOUR_K)

    assert result.neural_scale == 4
    assert result.neural_width == 6000
    assert result.target_width == 3840


def test_the_neural_result_is_never_smaller_than_the_target() -> None:
    """Otherwise the final step would be an enlargement, which is forbidden."""
    for source in (400, 700, 1000, 1900, 2600, 3800):
        for preset in TargetResolution:
            if source >= preset.long_edge:
                continue
            try:
                result = plan(source, source, preset)
            except (ValidationError, OutputTooLargeError):
                continue
            assert result.neural_width >= result.target_width, (source, preset)
            assert result.neural_height >= result.target_height, (source, preset)


def test_an_exact_match_needs_no_resize() -> None:
    """960 x 2 is exactly 1920, so there is nothing left to resample."""
    result = plan(960, 540, TargetResolution.TWO_K)

    assert result.neural_scale == 2
    assert (result.neural_width, result.neural_height) == (1920, 1080)
    assert (result.target_width, result.target_height) == (1920, 1080)
    assert result.needs_resize is False


def test_an_inexact_match_reports_that_a_resize_is_needed() -> None:
    result = plan(1000, 800, TargetResolution.TWO_K)

    assert result.needs_resize is True
    assert result.target_width == 1920
    assert result.neural_width == 2000


def test_the_target_is_never_exceeded() -> None:
    """The output the user receives is the size they asked for, exactly."""
    for width, height in [(700, 500), (1000, 1000), (500, 1300), (1919, 1080)]:
        result = plan(width, height, TargetResolution.TWO_K)
        assert max(result.target_width, result.target_height) == 1920


# --------------------------------------------------------------- refusals


def test_an_image_already_at_the_target_is_refused() -> None:
    """Enhancement cannot make an image smaller, and a silent downscale would
    answer a question the user did not ask."""
    with pytest.raises(ValidationError, match="already"):
        plan(1920, 1080, TargetResolution.TWO_K)


def test_an_image_beyond_the_target_is_refused_with_a_usable_message() -> None:
    with pytest.raises(ValidationError) as caught:
        plan(4000, 3000, TargetResolution.TWO_K)

    assert "already" in str(caught.value)
    assert caught.value.context["targetLongEdge"] == 1920


def test_a_target_beyond_the_largest_factor_is_refused() -> None:
    """400 px would need 19.2x for 8K, and 16x is the ceiling."""
    with pytest.raises(ValidationError, match="largest factor available") as caught:
        plan(400, 300, TargetResolution.EIGHT_K)

    assert caught.value.context["maximumScale"] == 16
    # The message says what the image *can* reach, rather than only refusing.
    assert caught.value.context["reachableLongEdge"] == 6400


def test_a_plan_that_would_exceed_the_pixel_limit_is_refused() -> None:
    """The intermediate is what the limit applies to - it is the largest thing
    the job holds, and it exists before the resize shrinks it."""
    # 3800 px long edge needs 2.02x for 8K, so 4x - and 3800x3400 at 4x is
    # 206.7 MP, past the limit, even though the 8K target itself is only 62 MP.
    with pytest.raises(OutputTooLargeError) as caught:
        plan(3800, 3400, TargetResolution.EIGHT_K)

    assert caught.value.context["neuralScale"] == 4
    assert caught.value.context["projectedPixels"] == 206_720_000
    assert caught.value.context["limitPixels"] == LIMIT


def test_the_limit_is_checked_against_the_intermediate_not_the_target() -> None:
    """A target that fits can still need an intermediate that does not."""
    tight = 50_000_000
    # 1900 px needs 2.02x for 4K, so 4x: the intermediate is 51.7 MP and over
    # the limit, while the 4K result itself would be 13.2 MP and well under.
    with pytest.raises(OutputTooLargeError):
        plan(1900, 1700, TargetResolution.FOUR_K, max_output_pixels=tight)

    assert tight > 3840 * 3436


def test_a_model_with_no_factors_is_refused() -> None:
    with pytest.raises(ValidationError, match="cannot produce"):
        plan(1000, 1000, TargetResolution.TWO_K, supported_scales=[])


def test_a_model_limited_to_2x_can_still_reach_a_near_target() -> None:
    """supported_scales comes from the model, so the plan respects it."""
    result = plan(1200, 800, TargetResolution.TWO_K, supported_scales=[2])

    assert result.neural_scale == 2


def test_a_model_limited_to_2x_is_refused_a_target_it_cannot_reach() -> None:
    with pytest.raises(ValidationError, match="largest factor available"):
        plan(1200, 800, TargetResolution.FOUR_K, supported_scales=[2])


# ------------------------------------------------------- every preset works


@pytest.mark.parametrize("preset", list(TargetResolution))
def test_every_preset_can_be_planned_from_a_reasonable_source(
    preset: TargetResolution,
) -> None:
    """Every preset is reachable, and lands exactly on its long edge.

    The source is sized per preset rather than fixed, because the presets now
    span a factor of eight: one photograph that reaches 2K comfortably would
    need 16x for 16K and blow past the pixel limit doing it. A quarter of the
    target long edge is a source each preset can reach with a single 4x pass.
    """
    long_edge = preset.long_edge // 4
    result = plan(long_edge, long_edge * 5 // 8, preset)

    assert max(result.target_width, result.target_height) == preset.long_edge
    assert result.neural_scale in SCALES
    assert result.neural_pixels >= result.target_pixels


# ------------------------------------------------- the cross-language mirror


def test_the_frontend_preset_table_matches_this_one() -> None:
    """The browser greys out unreachable presets using its own copy of these
    long edges. A copy that drifted would offer a target the backend refuses,
    or hide one it would have accepted - so the two are checked against each
    other rather than trusted to stay in step.

    Read out of the TypeScript source deliberately: the alternative is a third
    place where the numbers are written down.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[3] / "frontend" / "src" / "types" / "job.ts"
    if not source.is_file():  # pragma: no cover - backend checked out alone
        pytest.skip("frontend sources are not present")

    text = source.read_text(encoding="utf-8")
    block = re.search(
        r"TARGET_LONG_EDGE:\s*Record<TargetResolution,\s*number>\s*=\s*\{(.*?)\}",
        text,
        re.DOTALL,
    )
    assert block is not None, "TARGET_LONG_EDGE is not where this test expects it"

    frontend = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"'(\w+)':\s*(\d+)", block.group(1))
    }

    assert frontend == {preset.value: preset.long_edge for preset in TargetResolution}


# ------------------------------------------------- 16K, and why it is not 16x

#: What a 4x model actually offers, now that 16x is composed.
FOUR_X_MODEL = [4, 8, 16]


def test_sixteen_k_is_eight_times_the_base_not_sixteen() -> None:
    """The name is the trap this whole module exists to avoid.

    16K is 15360 px - eight times the 1920 the family is built on - so its
    long edge is reached by an 8x pass from 1080p, not a 16x one.
    """
    assert TargetResolution.SIXTEEN_K.long_edge == 15360
    assert TargetResolution.SIXTEEN_K.long_edge == 8 * TargetResolution.TWO_K.long_edge
    assert "15360" in TargetResolution.SIXTEEN_K.label


def test_a_1080p_source_reaches_sixteen_k_with_an_eight_times_pass() -> None:
    """The example that makes "16K is not 16x" concrete."""
    result = plan(1920, 1080, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    assert result.neural_scale == 8
    assert (result.target_width, result.target_height) == (15360, 8640)
    # 1920 x 8 lands exactly on 15360, so nothing is resampled afterwards.
    assert result.needs_resize is False


def test_a_1080p_source_reaches_eight_k_with_a_four_times_pass() -> None:
    """The neighbouring case, unchanged, so the pair can be read together."""
    result = plan(1920, 1080, TargetResolution.EIGHT_K, supported_scales=FOUR_X_MODEL)

    assert result.neural_scale == 4
    assert (result.target_width, result.target_height) == (7680, 4320)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        # A quarter of 16K: one 4x pass, then a small resample.
        (4000, 2250, 4),
        # An eighth: 8x, landing exactly.
        (1920, 1080, 8),
        # A sixteenth: the only case where 16K really does mean 16x.
        (1000, 563, 16),
    ],
)
def test_the_factor_sixteen_k_needs_depends_entirely_on_the_source(
    width: int, height: int, expected: int
) -> None:
    result = plan(width, height, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    assert result.neural_scale == expected
    assert max(result.target_width, result.target_height) == 15360


def test_sixteen_k_keeps_the_aspect_ratio_of_a_portrait_source() -> None:
    result = plan(1080, 1920, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    assert (result.target_width, result.target_height) == (8640, 15360)
    assert result.neural_scale == 8


def test_sixteen_k_dimensions_follow_the_long_edge_whichever_it_is() -> None:
    assert target_dimensions(1920, 1080, TargetResolution.SIXTEEN_K) == (15360, 8640)
    assert target_dimensions(1080, 1920, TargetResolution.SIXTEEN_K) == (8640, 15360)
    assert target_dimensions(1000, 1000, TargetResolution.SIXTEEN_K) == (15360, 15360)


def test_a_square_source_can_be_refused_where_a_wide_one_is_not() -> None:
    """The long edge decides the factor; the *area* decides whether it fits.

    Two sources with the same long edge are not interchangeable at 16K: a
    square one carries nearly twice the pixels and can exceed the limit while
    the 16:9 one sails through.
    """
    plan(1920, 1080, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    with pytest.raises(OutputTooLargeError):
        plan(1920, 1920, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)


def test_a_3000x2000_source_cannot_reach_sixteen_k_within_the_limit() -> None:
    """It picks the smallest sufficient factor and *then* finds it too large.

    8x is correct - 5.12x is needed and 8 is the smallest offered above it -
    and 24000x16000 is 384 MP. The limit is not loosened to accommodate that;
    the request is refused with the number in the message.
    """
    with pytest.raises(OutputTooLargeError) as caught:
        plan(3000, 2000, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    assert caught.value.context["neuralScale"] == 8
    assert caught.value.context["projectedPixels"] == 24000 * 16000
    assert caught.value.context["limitPixels"] == LIMIT
    assert "384 MP" in caught.value.message


def test_a_source_already_past_sixteen_k_is_refused() -> None:
    with pytest.raises(ValidationError, match="already"):
        plan(16000, 9000, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)


def test_a_source_too_small_for_sixteen_k_is_told_what_it_can_reach() -> None:
    """400 px needs 38.4x, which nothing offers."""
    with pytest.raises(ValidationError, match="largest factor available") as caught:
        plan(400, 300, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL)

    assert caught.value.context["maximumScale"] == 16
    assert caught.value.context["reachableLongEdge"] == 6400


def test_a_two_times_model_cannot_reach_sixteen_k() -> None:
    with pytest.raises(ValidationError, match="largest factor available"):
        plan(4000, 2250, TargetResolution.SIXTEEN_K, supported_scales=[2])


def test_sixteen_k_never_enlarges_to_reach_the_target() -> None:
    """The neural result is at least the target at every reachable source."""
    for width in (1000, 1200, 1920, 2200, 3900, 4400):
        try:
            result = plan(
                width, width * 9 // 16, TargetResolution.SIXTEEN_K, supported_scales=FOUR_X_MODEL
            )
        except (ValidationError, OutputTooLargeError):
            continue
        assert result.neural_width >= result.target_width, width
        assert result.neural_height >= result.target_height, width


# ------------------------------------------ the older presets are untouched


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (TargetResolution.TWO_K, 4),
        (TargetResolution.FOUR_K, 8),
        (TargetResolution.SIX_K, 16),
        (TargetResolution.EIGHT_K, 16),
    ],
)
def test_adding_sixteen_x_changes_which_factor_a_small_source_uses(
    preset: TargetResolution, expected: int
) -> None:
    """Not a regression: 6K and 8K from a 600 px source were *unreachable*
    before 16x existed, and are now reachable. Nothing that worked changed.
    """
    result = plan(600, 400, preset, supported_scales=FOUR_X_MODEL)

    assert result.neural_scale == expected


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (TargetResolution.FOUR_K, 4),
        (TargetResolution.SIX_K, 8),
        (TargetResolution.EIGHT_K, 8),
    ],
)
def test_a_typical_source_plans_exactly_as_it_did_before(
    preset: TargetResolution, expected: int
) -> None:
    """1200 px reached 4K with 4x, and 6K and 8K with 8x, before this phase -
    and must still, because 16x is only ever chosen when nothing smaller does.
    """
    assert plan(1200, 800, preset, supported_scales=FOUR_X_MODEL).neural_scale == expected
