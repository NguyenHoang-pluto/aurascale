"""The Phase 2.5 denoise experiment: definition, not execution.

`DEFAULT_DENOISE` ships at 1.0. The synthetic corpus used in Phase 2 showed
that setting removing 98.4 % of high-frequency energy, but it disqualified
itself - the model correctly treated its gaussian "texture" as noise and erased
it - so it could not support choosing a replacement. This module defines the
matrix that answers the question on real photographs.

Everything except denoise is pinned. One variable, five values, one corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks.corpus import ManifestEntry

#: The model under test. The only one in the registry with a denoise pair, so
#: the only one the setting reaches at all - `_denoise_for` drops it for every
#: other model, which is why this experiment says nothing about x4plus.
MODEL_ID = "realesr-general-x4v3"

#: The five values, including both endpoints. 0.0 is fully the wdn weights and
#: keeps the most noise; 1.0 is the standard weights and denoises hardest,
#: matching upstream's --denoise_strength.
DENOISE_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: The arm every other arm is measured against. 0.0 is the natural baseline:
#: it is the least processed output the model can produce, so a delta from it
#: reads directly as "what denoising removed".
BASELINE_DENOISE = 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Everything held constant across the matrix.

    Pinned explicitly rather than left to defaults, so a later change to a
    default cannot silently alter what a rerun measures.
    """

    model: str = MODEL_ID
    scale: int = 4
    output_format: str = "png"
    quality: int | None = None
    sharpen_strength: float = 0.0
    tile_size: int | None = None
    tile_pad: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: PNG, deliberately. JPEG would put its own quantisation between the model and
#: the metrics, and at these sizes chroma subsampling and block artefacts would
#: be measured as if they were denoising.
DEFAULT_CONFIG = BenchmarkConfig()


def arm_name(denoise: float) -> str:
    """A stable, sortable identifier for one arm."""
    return f"denoise-{denoise:.2f}"


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """One cell of the matrix: one image at one denoise value."""

    arm: str
    denoise: float
    image: str
    category: str
    config: BenchmarkConfig = field(default=DEFAULT_CONFIG)

    @property
    def is_baseline(self) -> bool:
        return self.denoise == BASELINE_DENOISE


def build_matrix(
    entries: list[ManifestEntry],
    *,
    values: tuple[float, ...] = DENOISE_VALUES,
    config: BenchmarkConfig = DEFAULT_CONFIG,
) -> list[PlannedRun]:
    """Every image crossed with every denoise value.

    Image-major, so the runs for one photograph are adjacent: a partial run
    still yields complete, comparable arms for the images it reached rather
    than one value across everything.
    """
    return [
        PlannedRun(
            arm=arm_name(value),
            denoise=value,
            image=entry.filename,
            category=entry.category,
            config=config,
        )
        for entry in entries
        for value in values
    ]


def settings_payload(run: PlannedRun) -> dict[str, Any]:
    """The `settings` JSON the job API expects for one run."""
    payload: dict[str, Any] = {"denoiseStrength": run.denoise}
    if run.config.sharpen_strength:
        payload["sharpenStrength"] = run.config.sharpen_strength
    return payload
