"""What a mode means, and what it must not quietly override.

The point of a central planner is that a mode is defaults, not a gate. These
check that an explicit request always wins - which is what keeps clients
written before modes existed working - and that Creative does not carry its
denoise setting to a model that has no use for it.
"""

from __future__ import annotations

import pytest

from app.models.enums import EnhancementMode
from app.services.mode_planner import (
    CREATIVE_DENOISE,
    CREATIVE_MODEL,
    PIPELINE_STAGES,
    STANDARD_MODEL,
    built_stages,
    plan_mode,
    resolve_denoise,
    resolve_model,
)

# ---------------------------------------------------------------- the modes


def test_standard_is_the_model_that_already_shipped() -> None:
    """Backward compatibility starts here: Standard must change nothing."""
    plan = plan_mode(EnhancementMode.STANDARD)

    assert plan.model_id == STANDARD_MODEL == "RealESRGAN_x4plus"
    assert plan.is_standard


def test_standard_sends_no_denoise() -> None:
    """Its model has no denoise pair, so a value would be refused, not ignored."""
    assert plan_mode(EnhancementMode.STANDARD).denoise_strength is None


def test_creative_uses_the_model_phase_3_recommended() -> None:
    plan = plan_mode(EnhancementMode.CREATIVE)

    assert plan.model_id == CREATIVE_MODEL == "realesr-general-x4v3"
    assert not plan.is_standard


def test_creative_denoises_conservatively_rather_than_not_at_all() -> None:
    """Phase 3 found this network amplifies flat-region noise by up to 182 %
    at denoise 0, so Creative cannot simply mean "less denoising"."""
    assert plan_mode(EnhancementMode.CREATIVE).denoise_strength == CREATIVE_DENOISE
    assert 0.0 < CREATIVE_DENOISE < 1.0


def test_creatives_denoise_is_its_own_constant_not_the_global_default() -> None:
    """DEFAULT_DENOISE stays at 1.0 until there is evidence to move it; this
    phase must not move it by the back door."""
    assert CREATIVE_DENOISE == 0.25


def test_each_mode_carries_the_wording_the_ui_shows() -> None:
    assert plan_mode(EnhancementMode.STANDARD).summary == "Natural enhancement with high fidelity"
    assert plan_mode(EnhancementMode.CREATIVE).summary == "Stronger detail and visual enhancement"


def test_the_two_modes_differ_in_model() -> None:
    assert (
        plan_mode(EnhancementMode.STANDARD).model_id != plan_mode(EnhancementMode.CREATIVE).model_id
    )


# ------------------------------------------------------------ model choice


def test_an_explicit_model_beats_the_mode() -> None:
    """Otherwise a mode would silently override the advanced controls."""
    assert resolve_model(EnhancementMode.CREATIVE, "RealESRGAN_x2plus") == "RealESRGAN_x2plus"


def test_no_mode_and_no_model_leaves_the_choice_to_the_caller() -> None:
    """A request from before modes existed must behave exactly as it did."""
    assert resolve_model(None, None) is None


def test_a_mode_alone_supplies_its_model() -> None:
    assert resolve_model(EnhancementMode.STANDARD, None) == STANDARD_MODEL
    assert resolve_model(EnhancementMode.CREATIVE, None) == CREATIVE_MODEL


def test_an_explicit_model_without_a_mode_is_untouched() -> None:
    assert resolve_model(None, "RealESRGAN_x4plus_anime_6B") == "RealESRGAN_x4plus_anime_6B"


# ---------------------------------------------------------- denoise choice


def test_an_explicit_denoise_beats_the_mode() -> None:
    assert resolve_denoise(EnhancementMode.CREATIVE, 0.9, model_supports_denoise=True) == 0.9


def test_an_explicit_zero_is_honoured_rather_than_treated_as_absent() -> None:
    """0.0 is a real setting - fully the wdn weights - not "unset"."""
    assert resolve_denoise(EnhancementMode.CREATIVE, 0.0, model_supports_denoise=True) == 0.0


def test_a_mode_supplies_denoise_only_to_a_model_that_can_use_it() -> None:
    """Offering Creative's value to a network with no pair would build a
    request that is refused downstream."""
    assert resolve_denoise(EnhancementMode.CREATIVE, None, model_supports_denoise=False) is None
    assert (
        resolve_denoise(EnhancementMode.CREATIVE, None, model_supports_denoise=True)
        == CREATIVE_DENOISE
    )


def test_no_mode_supplies_no_denoise() -> None:
    assert resolve_denoise(None, None, model_supports_denoise=True) is None


def test_standard_supplies_no_denoise_even_on_a_capable_model() -> None:
    """Standard means "as it shipped", and it shipped without one."""
    assert resolve_denoise(EnhancementMode.STANDARD, None, model_supports_denoise=True) is None


# ------------------------------------------------------------ the pipeline


def test_only_stages_that_exist_are_marked_as_built() -> None:
    """A stage that does nothing is worse than a stage that is absent."""
    assert built_stages() == ("denoise", "model", "sharpen")


def test_the_unbuilt_stages_are_named_so_their_place_is_agreed() -> None:
    unbuilt = [name for name, built in PIPELINE_STAGES if not built]

    assert unbuilt == ["analysis", "detail-recovery", "artifact-control"]


def test_the_stage_order_is_the_pipeline_order() -> None:
    """Analysis first, artifact control last - the shape future work slots into."""
    assert [name for name, _ in PIPELINE_STAGES] == [
        "analysis",
        "denoise",
        "model",
        "detail-recovery",
        "sharpen",
        "artifact-control",
    ]


@pytest.mark.parametrize("mode", list(EnhancementMode))
def test_every_mode_plans(mode: EnhancementMode) -> None:
    plan = plan_mode(mode)

    assert plan.model_id
    assert plan.summary
    assert plan.mode is mode
