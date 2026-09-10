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
CREATIVE_DENOISE = 0.25

#: What a denoise-capable model runs at when nobody said otherwise.
#:
#: This exists because "nobody said otherwise" used to mean something nobody
#: chose. A request that named `realesr-general-x4v3` without a denoise value
#: resolved to `None`, `None` reached `_resolve_blend`, and `None` there means
#: "load the standard weights unblended" - which for this model is DNI 1.00,
#: the strongest denoising it can do. Phase 3B confirmed the two are
#: byte-identical, so the shipped default was 1.00 by omission rather than by
#: decision.
#:
#: 1.00 is not defensible as a default and two phases said so independently.
#: F2 measured it at a median **-49.9 % of high-frequency energy** with skin
#: visibly plastic in a blind pass; Phase 2.5 reached the same conclusion on a
#: separate corpus. 0.25 is the least destructive setting that still denoises:
#: F2 puts it at -14.1 % hf for -12.9 % noise, equivalent to the baseline on
#: skin and foliage, and better on low light.
#:
#: 0.25 over 0.50 on the asymmetry Phase 2.5 argued and F2 did not overturn:
#: under-denoising is recoverable by the user, erased texture is not.
#:
#: Deliberately a separate constant from `CREATIVE_DENOISE` even though the two
#: currently agree. They answer different questions - "what does Creative mean"
#: and "what happens when nobody chose" - and collapsing them would make a
#: future change to one silently change the other.
DEFAULT_DENOISE = 0.25


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

    Four tiers, in order, and each one exists for a different reason:

      1. **An explicit value wins**, as with the model. That includes `0.0`,
         which is a real setting - fully the `wdn` weights - and not an absence.
         The check is `is not None` rather than a truth test precisely so zero
         survives it.
      2. **A model with no denoise pair gets `None`.** Offering Creative's 0.25
         to a network that cannot use it would be refused downstream, and this
         function should not manufacture a request that cannot succeed.
      3. **A mode's value**, when the mode names one.
      4. **`DEFAULT_DENOISE`** otherwise.

    Tier 4 is the one this function was missing, and its absence is the whole
    reason for it. Two paths fell through to `None`, and `None` downstream means
    the standard weights unblended - DNI 1.00 for the only model that has a
    pair:

      * a request with **no mode at all** that names `realesr-general-x4v3`
        directly, which is what the client sends when the denoise slider has
        not been touched;
      * a request in **Standard mode** that overrides the model to a
        denoise-capable one, because `plan_mode(STANDARD).denoise_strength` is
        `None` and that `None` was indistinguishable from "unset".

    Standard's own model has no denoise pair, so tier 2 still returns `None` for
    it and Standard is unchanged. What tier 4 fixes is the case where the user
    reached a denoise-capable network without ever choosing a strength.
    """
    if requested_denoise is not None:
        return requested_denoise

    if not model_supports_denoise:
        return None

    if mode is not None:
        planned = plan_mode(mode).denoise_strength
        if planned is not None:
            return planned

    return DEFAULT_DENOISE
