"""The Phase 3 model comparison: which super-resolution model to standardise on.

Definition only; nothing here runs inference. The matrix pins everything except
the model so the arms are comparable, and it is deliberately restricted to
weights already in the local registry - a candidate that has to be downloaded,
vendored or licence-checked first is an engineering decision, not a benchmark
arm, and is assessed in the report instead.

One fairness problem is worth stating, because it shapes the matrix. Only
`realesr-general-x4v3` has a denoise pair; the other two have no denoising
stage at all. Comparing it at the shipped default of 1.0 would measure the
denoiser rather than the network. It therefore appears twice: once at 0.0,
which is the like-for-like arm, and once at 1.0, which is what a user actually
gets today.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks.corpus import ManifestEntry

#: Every arm runs at this factor. 4x is the product default and the only scale
#: all three candidates reach natively, so no arm pays for a second pass.
COMPARISON_SCALE = 4

#: PNG, so the encoder's own quantisation is never mistaken for a model
#: difference. The whole point is to compare networks, not codecs.
COMPARISON_FORMAT = "png"


@dataclass(frozen=True, slots=True)
class ModelArm:
    """One model, at one setting, under one name."""

    name: str
    model: str
    #: None for a model with no denoise pair - the setting does not reach it,
    #: and sending one would be refused by `_denoise_for`.
    denoise: float | None = None
    notes: str = ""

    @property
    def is_reference(self) -> bool:
        """Whether this arm is what production ships today."""
        return self.name == REFERENCE_ARM


#: The arm every other is measured against: the current production default.
REFERENCE_ARM = "x4plus"

CANDIDATES: tuple[ModelArm, ...] = (
    ModelArm(
        name=REFERENCE_ARM,
        model="RealESRGAN_x4plus",
        notes=(
            "RRDBNet, 23 blocks, 63.9 MB. The production default and the baseline for every delta."
        ),
    ),
    ModelArm(
        name="anime6b",
        model="RealESRGAN_x4plus_anime_6B",
        notes=(
            "RRDBNet, 6 blocks, 17.1 MB. Trained for illustration and line "
            "art; included to see how a specialised network behaves on "
            "photographs, which is the failure mode a Creative mode would "
            "have to avoid."
        ),
    ),
    ModelArm(
        name="general-v3-dn0",
        model="realesr-general-x4v3",
        denoise=0.0,
        notes=(
            "SRVGGNetCompact, 4.7 MB. Denoise at 0.0 so the network is "
            "compared, not its denoiser - the like-for-like arm."
        ),
    ),
    ModelArm(
        name="general-v3-dn1",
        model="realesr-general-x4v3",
        denoise=1.0,
        notes=(
            "The same network at the shipped DEFAULT_DENOISE, which is what a "
            "user selecting this model gets today."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Everything held constant across the matrix."""

    scale: int = COMPARISON_SCALE
    output_format: str = COMPARISON_FORMAT
    quality: int | None = None
    sharpen_strength: float = 0.0
    tile_size: int | None = None
    tile_pad: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ComparisonConfig()


@dataclass(frozen=True, slots=True)
class PlannedComparison:
    """One cell of the matrix: one image through one arm."""

    arm: str
    model: str
    denoise: float | None
    image: str
    category: str
    config: ComparisonConfig = field(default=DEFAULT_CONFIG)

    @property
    def is_reference(self) -> bool:
        return self.arm == REFERENCE_ARM


def build_matrix(
    entries: list[ManifestEntry],
    *,
    arms: tuple[ModelArm, ...] = CANDIDATES,
    config: ComparisonConfig = DEFAULT_CONFIG,
) -> list[PlannedComparison]:
    """Every image crossed with every arm, image-major.

    Image-major so an interrupted run still leaves complete, comparable arms
    for the images it reached.
    """
    return [
        PlannedComparison(
            arm=arm.name,
            model=arm.model,
            denoise=arm.denoise,
            image=entry.filename,
            category=entry.category,
            config=config,
        )
        for entry in entries
        for arm in arms
    ]


def settings_payload(run: PlannedComparison) -> dict[str, Any] | None:
    """The `settings` JSON for one run, or None when there is nothing to send.

    A denoise value is sent only for the model that has a pair. The API refuses
    it for the others, and sending one anyway would fail the run rather than
    being ignored.
    """
    payload: dict[str, Any] = {}
    if run.denoise is not None:
        payload["denoiseStrength"] = run.denoise
    if run.config.sharpen_strength:
        payload["sharpenStrength"] = run.config.sharpen_strength

    return payload or None
