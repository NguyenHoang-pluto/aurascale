"""Measure result files and report deltas against a baseline arm.

Deliberately does **not** run inference. Phase 0 changes nothing in the
production pipeline, so this reads results that already exist - produced by the
API, by a script, or by hand - and measures them. Keeping the two apart also
means a benchmark run cannot perturb the thing it is measuring.

Usage:

    python -m benchmarks.runner --corpus            # what is present
    python -m benchmarks.runner --spec runs.json    # measure and compare

The spec is a small JSON document:

    {
      "baseline": "x4plus-4x",
      "runs": [
        {
          "arm": "x4plus-4x",
          "category": "landscape-detail",
          "image": "lake.png",
          "result": "/path/to/result.png",
          "settings": {"model": "RealESRGAN_x4plus", "scale": 4},
          "input": {"width": 2000, "height": 1500},
          "cost": {"processing_ms": 32837}
        }
      ]
    }

`cost` is optional and every field inside it is optional: a run measured after
the fact has no timing, and inventing one would be worse than leaving it null.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.corpus import DEFAULT_CORPUS_DIR, describe_corpus
from benchmarks.metrics import measure_file
from benchmarks.report import Record, Report, RunCost, RunSettings


def peak_rss_mb() -> float | None:
    """Peak resident memory, when the platform can report it.

    `psutil` gives current RSS everywhere; a true peak is only available on
    Windows. Returns None rather than a current-usage figure dressed up as a
    peak.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return None

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return float(peak) / 1024 / 1024 if peak is not None else None


def peak_vram_mb() -> float | None:
    """Peak CUDA allocation this process has made, or None without a GPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return None

    if not torch.cuda.is_available():
        return None

    return float(torch.cuda.max_memory_allocated()) / 1024 / 1024


def _output_size(path: Path) -> tuple[int, int]:
    """The result's real dimensions, read through the trusted opener."""
    from app.services.image_service import open_trusted

    with open_trusted(path) as opened:
        return (int(opened.size[0]), int(opened.size[1]))


def record_from_spec(entry: dict[str, Any]) -> Record:
    """Build one record, measuring the result file it points at."""
    result = Path(entry["result"])
    if not result.is_file():
        raise FileNotFoundError(f"result not found: {result}")

    settings_raw = dict(entry.get("settings", {}))
    cost_raw = dict(entry.get("cost", {}))
    source = dict(entry.get("input", {}))

    width, height = _output_size(result)

    return Record(
        arm=str(entry["arm"]),
        category=str(entry.get("category", "uncategorised")),
        image=str(entry["image"]),
        settings=RunSettings(
            model=str(settings_raw.get("model", "unknown")),
            scale=int(settings_raw.get("scale", 0)),
            denoise=settings_raw.get("denoise"),
            sharpen=float(settings_raw.get("sharpen", 0.0)),
            output_format=str(settings_raw.get("output_format", result.suffix.lstrip("."))),
            quality=settings_raw.get("quality"),
        ),
        input_width=int(source.get("width", 0)),
        input_height=int(source.get("height", 0)),
        output_width=width,
        output_height=height,
        cost=RunCost(
            processing_ms=cost_raw.get("processing_ms"),
            peak_rss_mb=cost_raw.get("peak_rss_mb"),
            peak_vram_mb=cost_raw.get("peak_vram_mb"),
            output_bytes=result.stat().st_size,
        ),
        metrics=measure_file(result),
    )


def build_report(spec: dict[str, Any]) -> Report:
    """Measure every run in a spec and assemble the report."""
    report = Report(baseline_arm=str(spec["baseline"]))

    for entry in spec.get("runs", []):
        report.add(record_from_spec(entry))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.runner", description=__doc__)
    parser.add_argument("--corpus", action="store_true", help="report what the corpus holds")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--spec", type=Path, help="JSON spec of runs to measure")
    parser.add_argument("--json", type=Path, help="write the full report here")
    arguments = parser.parse_args(argv)

    if arguments.corpus:
        print(describe_corpus(arguments.corpus_dir))
        return 0

    if arguments.spec is None:
        parser.error("one of --corpus or --spec is required")

    spec = json.loads(arguments.spec.read_text(encoding="utf-8"))
    report = build_report(spec)

    print(report.to_text())
    if arguments.json is not None:
        written = report.write_json(arguments.json)
        print(f"\nwrote {written}")

    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
