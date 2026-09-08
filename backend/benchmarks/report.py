"""Benchmark records and the deltas between them.

A record is what a single arm produced for a single image: the settings, the
cost, and the metrics. A report is a baseline plus the arms measured against
it.

The one rule enforced here is that a delta is only reported between arms whose
outputs are the same size. Every metric in `metrics` is resolution-dependent -
the same subject at 4x and at 8x has different gradient statistics without
either being better - so a cross-resolution delta would be a number that looks
meaningful and is not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.metrics import ImageMetrics

# Metrics where a delta is reported. `tiles_sampled` is provenance, not a
# measurement, so it is carried but never differenced.
DELTA_FIELDS: tuple[str, ...] = (
    "sobel_mean",
    "sobel_p95",
    "local_contrast",
    "high_frequency_ratio",
    "flat_noise",
    "edge_overshoot",
)


@dataclass(frozen=True, slots=True)
class RunSettings:
    """Exactly what produced a result, so a row can be reproduced."""

    model: str
    scale: int
    denoise: float | None = None
    sharpen: float = 0.0
    output_format: str = "png"
    quality: int | None = None


@dataclass(frozen=True, slots=True)
class RunCost:
    """What it cost. Fields are None when the platform could not report them."""

    processing_ms: int | None = None
    peak_rss_mb: float | None = None
    peak_vram_mb: float | None = None
    output_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class Record:
    """One arm, one image."""

    arm: str
    category: str
    image: str
    settings: RunSettings
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    cost: RunCost
    metrics: ImageMetrics

    @property
    def output_size(self) -> tuple[int, int]:
        return (self.output_width, self.output_height)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "category": self.category,
            "image": self.image,
            "settings": asdict(self.settings),
            "input": {"width": self.input_width, "height": self.input_height},
            "output": {"width": self.output_width, "height": self.output_height},
            "cost": asdict(self.cost),
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Delta:
    """One arm measured against the baseline for the same image."""

    arm: str
    image: str
    comparable: bool
    #: Why not, when `comparable` is False. Never silently dropped.
    reason: str | None = None
    absolute: dict[str, float] = field(default_factory=dict)
    relative: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def delta_between(baseline: Record, candidate: Record) -> Delta:
    """Compare one arm to the baseline, or explain why it cannot be compared.

    Refused rather than approximated when the outputs differ in size. Comparing
    a 4x arm to an 8x arm on gradient statistics would produce a confident
    number describing nothing but the resolution change.
    """
    if baseline.image != candidate.image:
        return Delta(
            arm=candidate.arm,
            image=candidate.image,
            comparable=False,
            reason=f"different images: {baseline.image} vs {candidate.image}",
        )

    if baseline.output_size != candidate.output_size:
        return Delta(
            arm=candidate.arm,
            image=candidate.image,
            comparable=False,
            reason=(
                "different output sizes "
                f"({baseline.output_width}x{baseline.output_height} vs "
                f"{candidate.output_width}x{candidate.output_height}); "
                "these metrics are resolution-dependent"
            ),
        )

    absolute: dict[str, float] = {}
    relative: dict[str, float] = {}

    base_values = baseline.metrics.as_dict()
    candidate_values = candidate.metrics.as_dict()

    for name in DELTA_FIELDS:
        before = float(base_values[name])
        after = float(candidate_values[name])
        absolute[name] = after - before
        # A zero baseline has no percentage; reported as absolute only.
        relative[name] = ((after - before) / before * 100.0) if before != 0.0 else float("nan")

    return Delta(
        arm=candidate.arm,
        image=candidate.image,
        comparable=True,
        absolute=absolute,
        relative=relative,
    )


@dataclass
class Report:
    """A baseline arm plus everything measured against it."""

    baseline_arm: str
    records: list[Record] = field(default_factory=list)

    def add(self, record: Record) -> None:
        self.records.append(record)

    def baseline_for(self, image: str) -> Record | None:
        for record in self.records:
            if record.arm == self.baseline_arm and record.image == image:
                return record
        return None

    def deltas(self) -> list[Delta]:
        """Every non-baseline record, against its own image's baseline."""
        results: list[Delta] = []

        for record in self.records:
            if record.arm == self.baseline_arm:
                continue

            baseline = self.baseline_for(record.image)
            if baseline is None:
                results.append(
                    Delta(
                        arm=record.arm,
                        image=record.image,
                        comparable=False,
                        reason=f"no baseline run for {record.image}",
                    )
                )
                continue

            results.append(delta_between(baseline, record))

        return results

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_arm": self.baseline_arm,
            "records": [record.as_dict() for record in self.records],
            "deltas": [delta.as_dict() for delta in self.deltas()],
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    def to_text(self) -> str:
        """A readable summary. Deltas only, with no verdict attached to them."""
        lines = [f"baseline: {self.baseline_arm}", ""]

        for delta in self.deltas():
            lines.append(f"{delta.arm}  [{delta.image}]")
            if not delta.comparable:
                lines.append(f"  not comparable: {delta.reason}")
                lines.append("")
                continue

            for name in DELTA_FIELDS:
                absolute = delta.absolute[name]
                relative = delta.relative[name]
                percent = "     n/a" if relative != relative else f"{relative:+7.1f}%"
                lines.append(f"  {name:22} {absolute:+.6f}  {percent}")
            lines.append("")

        lines.append("Higher is not automatically better: read sobel_p95 and")
        lines.append("edge_overshoot together, and local_contrast against flat_noise.")
        return "\n".join(lines)
