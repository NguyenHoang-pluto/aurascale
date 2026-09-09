"""What a mode means, in one place.

A mode is a named bundle of decisions the user should not have to make: which
network runs, and how much denoising it gets. Those decisions were previously
implicit in whatever the client happened to send. Centralising them means the
API, the worker and the UI cannot drift apart about what "Creative" is.

Two things this module deliberately does **not** do.

It does not gate anything. A mode selects defaults; every one of them remains
overridable through the existing explicit fields, and a request that names a
model and a denoise value still gets exactly those.

And it does not pretend. Creative today is a different network at a different
denoise setting - that is all it is, and the description says so. The stage
list below names where analysis, detail recovery and artifact control would
go, but nothing stands in for them, because a stage that does nothing is worse
than a stage that is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import ValidationError
from app.models.enums import EnhancementMode

#: Fidelity first. The model that shipped, and the one Phase 3 recommended
#: keeping: it never wins every metric but it never fails badly either.
STANDARD_MODEL = "RealESRGAN_x4plus"

#: Detail first. Phase 3 found this the only arm to beat the reference on
#: detail anywhere, at four times the speed and a fraction of the VRAM.
CREATIVE_MODEL = "realesr-general-x4v3"

#: Creative's denoise, and the reason this is a named constant rather than a
#: reuse of the global default.
#:
#: Phase 3 also found this network amplifies noise - up to +182 % in flat
#: regions at denoise 0 - so Creative cannot simply mean "less denoising".
#: Phase 2.5 measured the sweep on real photographs and recommended 0.25 as
#: the least destructive setting that still denoises. That is what this is.
#:
#: It is emphatically *not* `DEFAULT_DENOISE`, which stays at 1.0 for every
#: other path until there is evidence to move it.
CREATIVE_DENOISE = 0.25


@dataclass(frozen=True, slots=True)
class ModePlan:
    """The model and denoise a mode implies, before any user override."""

    mode: EnhancementMode
    model_id: str
    denoise_strength: float | None
    summary: str

    @property
    def is_standard(self) -> bool:
        return self.mode is EnhancementMode.STANDARD


#: The pipeline a plan describes, in order. Only the stages marked as built
#: exist; the rest are here so the shape is agreed before anything fills them,
#: and so nobody has to guess where a future stage belongs.
PIPELINE_STAGES: tuple[tuple[str, bool], ...] = (
    ("analysis", False),
    ("denoise", True),  # DNI weight interpolation, models with a pair
    ("model", True),  # the neural pass or passes
    ("detail-recovery", False),
    ("sharpen", True),  # unsharp mask, scale-aware, labelled a post-process
    ("artifact-control", False),
)


def built_stages() -> tuple[str, ...]:
    """The stages that actually run today."""
    return tuple(name for name, built in PIPELINE_STAGES if built)


def plan_mode(mode: EnhancementMode) -> ModePlan:
    """The defaults a mode implies."""
    if mode is EnhancementMode.STANDARD:
        return ModePlan(
            mode=mode,
            model_id=STANDARD_MODEL,
            # This model has no denoise pair, so the setting does not reach it
            # and sending one would be refused rather than ignored.
            denoise_strength=None,
            summary="Natural enhancement with high fidelity",
        )

    if mode is EnhancementMode.CREATIVE:
        return ModePlan(
            mode=mode,
            model_id=CREATIVE_MODEL,
            denoise_strength=CREATIVE_DENOISE,
            summary="Stronger detail and visual enhancement",
        )

    raise ValidationError(  # pragma: no cover - StrEnum admits nothing else
        f"{mode} is not a mode this build knows.",
        technical=f"mode={mode!r}",
    )


def resolve_model(mode: EnhancementMode | None, requested_model: str | None) -> str | None:
    """Which model a request should run, given a mode and an explicit choice.

    An explicit model always wins. That is what keeps every client written
    before modes existed working unchanged, and it is what lets the advanced
    controls stay meaningful now that they do.
    """
    if requested_model is not None:
        return requested_model
    if mode is None:
        return None
    return plan_mode(mode).model_id


def resolve_denoise(
    mode: EnhancementMode | None,
    requested_denoise: float | None,
    *,
    model_supports_denoise: bool,
) -> float | None:
    """The denoise a request should run at.

    An explicit value wins, as with the model. A mode's value is applied only
    when the chosen model can actually use it - offering Creative's 0.25 to a
    network with no denoise pair would be refused downstream, and the mode
    should not manufacture a request that cannot succeed.
    """
    if requested_denoise is not None:
        return requested_denoise
    if mode is None or not model_supports_denoise:
        return None
    return plan_mode(mode).denoise_strength
