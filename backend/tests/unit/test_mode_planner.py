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
    DEFAULT_DENOISE,
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
    """The two are separate decisions and must stay separately expressible.

    They currently agree at 0.25 because the same F2 evidence settled both, so
    equality alone cannot show they are distinct. What this asserts instead is
    that Creative reads its value from `CREATIVE_DENOISE` rather than from
    `DEFAULT_DENOISE`, which is what would break first if the two were ever
    collapsed into one.
    """
    assert CREATIVE_DENOISE == 0.25
    assert plan_mode(EnhancementMode.CREATIVE).denoise_strength == CREATIVE_DENOISE


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


def test_no_mode_on_a_capable_model_falls_back_to_the_default() -> None:
    """The case this behaviour was changed for.

    It used to return `None`, and `None` downstream loads the standard weights
    unblended - DNI 1.00 for the only model with a pair. So a request naming
    `realesr-general-x4v3` with no mode and no slider movement ran the one
    setting F2 and Phase 2.5 both ruled out, purely by omission.
    """
    assert resolve_denoise(None, None, model_supports_denoise=True) == DEFAULT_DENOISE


def test_standard_on_a_capable_model_falls_back_to_the_default() -> None:
    """Standard names no denoise, which is not the same as choosing none.

    `plan_mode(STANDARD).denoise_strength` is `None` because Standard's own
    model has no denoise pair. When the model is overridden to one that does,
    that `None` used to fall through as "unset" and land on DNI 1.00. Standard
    has no opinion here, so the default supplies one.
    """
    assert (
        resolve_denoise(EnhancementMode.STANDARD, None, model_supports_denoise=True)
        == DEFAULT_DENOISE
    )


def test_standard_on_its_own_model_is_unchanged() -> None:
    """The guard that keeps this fix from touching Standard at all.

    `RealESRGAN_x4plus` declares no denoise pair, so the capability tier
    returns `None` before any default is considered.
    """
    assert resolve_denoise(EnhancementMode.STANDARD, None, model_supports_denoise=False) is None
    assert resolve_denoise(None, None, model_supports_denoise=False) is None


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("mode", [None, EnhancementMode.STANDARD, EnhancementMode.CREATIVE])
def test_an_explicit_value_survives_every_mode(
    strength: float, mode: EnhancementMode | None
) -> None:
    """Explicit beats every tier below it, at every value including both ends.

    0.0 and 1.0 are the ones worth naming: 0.0 because a truth test would read
    it as absent, and 1.0 because it is the value the old default produced by
    accident and must still be reachable on purpose.
    """
    assert resolve_denoise(mode, strength, model_supports_denoise=True) == strength


def test_an_explicit_value_is_still_dropped_for_a_model_that_cannot_use_it() -> None:
    """Capability is checked after the explicit tier, not before it.

    An explicit value on a model with no pair is refused by `_validate_denoise`
    with a message naming the model, which is a better answer than silently
    dropping it here.
    """
    assert resolve_denoise(None, 0.5, model_supports_denoise=False) == 0.5


def test_the_default_is_the_value_the_research_settled_on() -> None:
    """0.25, and inside the range where it means something.

    F2: DNI 1.00 costs a median -49.9% of high-frequency energy with visibly
    plastic skin; 0.25 costs -14.1% for -12.9% noise. Phase 2.5 reached 0.25
    independently on a separate corpus.
    """
    assert DEFAULT_DENOISE == 0.25
    assert 0.0 < DEFAULT_DENOISE < 1.0


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


def test_the_frontend_mode_default_table_matches_this_one() -> None:
    """The panel shows a mode's model before the user picks one, using its own
    copy of these ids. A copy that drifted would name one model on screen while
    the planner ran another - which is exactly the bug this table was added to
    fix, reappearing from the other side.

    Read out of the TypeScript source deliberately, the same way the resolution
    planner checks `TARGET_LONG_EDGE`: the alternative is a third place where
    the ids are written down.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[3] / "frontend" / "src" / "types" / "job.ts"
    if not source.is_file():  # pragma: no cover - backend checked out alone
        pytest.skip("frontend sources are not present")

    text = source.read_text(encoding="utf-8")
    block = re.search(
        r"MODE_DEFAULT_MODEL:\s*Record<EnhancementMode,\s*string>\s*=\s*\{(.*?)\}",
        text,
        re.DOTALL,
    )
    assert block is not None, "MODE_DEFAULT_MODEL is not where this test expects it"

    frontend = dict(re.findall(r"(\w+):\s*'([^']+)'", block.group(1)))

    assert frontend == {mode.value: plan_mode(mode).model_id for mode in EnhancementMode}
