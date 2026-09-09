"""Phase 4 F2: which denoise strength should `realesr-general-x4v3` default to?

    python -m benchmarks.phase4_f2 --sweep    # 5 photographs x 5 strengths
    python -m benchmarks.phase4_f2 --strips   # blind, shuffled comparison strips

Research only. Nothing here changes production behaviour; `DEFAULT_DENOISE` and
`CREATIVE_DENOISE` are read for the record and never written.

Why this exists
---------------

Phase 2.5 recommended 0.25 on metrics plus a visual pass, but its own report
flagged that the lettering was never actually inspected and that the corpus was
thin. Phase 4 F1 then found that the recommendation had never reached a user
anyway: the client always sent an explicit denoise, so `CREATIVE_DENOISE` was
unreachable. With F1 committed, Creative genuinely runs at 0.25 - which makes
"is 0.25 the right number" a question with consequences rather than a
hypothetical.

What is reused rather than rebuilt
----------------------------------

`corpus` for the manifest-validated five photographs, `metrics` for the luma
statistics, `phase3b` for the chroma statistics and the shared-flat-mask
correction, `phase3a._load` for reading a source. Arms run through the
production `RealEsrganUpscaler.upscale`, so tiling, the OOM ladder and the
precision policy are the product's and not this file's.

The x4plus reference is **not** re-run. Phase 3C already measured it on the same
five photographs at 4x, PNG, sharpening 0, and re-running it would spend four
minutes to reproduce numbers that are already recorded. It is loaded from
`phase3c_measurements.json` and labelled as such wherever it appears.

DNI semantics, verified rather than assumed
-------------------------------------------

`ModelManager._resolve_blend` was interrogated directly for every value this
sweep uses:

    passed=None  -> blend=None   no blend, the plain x4v3 weights
    passed=0.00  -> blend=0.00   blended fully to the wdn counterpart
    passed=0.25  -> blend=0.25
    passed=0.50  -> blend=0.50
    passed=0.75  -> blend=0.75
    passed=1.00  -> blend=None   no blend, the plain x4v3 weights

So **1.00 and None are the same code path**, which Phase 3B confirmed produces
byte-identical output. And 0.00 is *not* "denoise off" - it is the fully-wdn
end. Both are recorded per run so no reader has to take that on trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.corpus import DEFAULT_CORPUS_DIR, CorpusImage, load_corpus, load_manifest
from benchmarks.metrics import ImageMetrics, measure
from benchmarks.phase3a import _load
from benchmarks.phase3b import chroma_metrics, flat_mask, ycrcb_tiles
from benchmarks.phase3c import REGIONS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase4_f2_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase4-f2-crops"
PHASE3C = RESULTS_DIR / "phase3c_measurements.json"

MODEL = "realesr-general-x4v3"
SCALE = 4

#: The sweep. Both endpoints included: 0.00 is fully the wdn weights and keeps
#: the most noise, 1.00 is the plain x4v3 weights and denoises hardest. 1.00 is
#: also what `DEFAULT_DENOISE` ships, which is the value under question.
STRENGTHS: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)

#: Every delta is measured against this arm - the least-denoised output the
#: model can produce, so a delta reads directly as "what denoising removed".
BASELINE = 0.00


def arm(strength: float) -> str:
    return f"dn{strength:.2f}"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Exactly how one cell ran. Recorded, never assumed."""

    device: str
    fp16: bool
    tile_size: int
    tiles: int
    tile_reduced: bool
    peak_vram_mb: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_digests(models: Any) -> dict[str, str]:
    """SHA-256 of the weights actually on disk, so a rerun can prove identity."""
    digests: dict[str, str] = {}
    for name in (MODEL, "realesr-general-wdn-x4v3"):
        path = models.get(name).path
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def encoded_png(image: np.ndarray[Any, Any]) -> tuple[int, str]:
    """PNG size and hash. Lossless, so the metrics are unaffected by it."""
    import cv2

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return -1, ""
    raw = buffer.tobytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def x4plus_reference() -> dict[str, dict[str, Any]]:
    """Phase 3C's x4plus arm, by category. Reused rather than re-measured.

    Returns an empty mapping if the file is absent, which the report then says
    rather than quietly dropping the comparison.
    """
    if not PHASE3C.is_file():
        return {}

    payload = json.loads(PHASE3C.read_text(encoding="utf-8"))
    return {
        row["category"]: row
        for row in payload.get("rows", [])
        if row.get("experiment") == "pilot" and row.get("arm") == "x4plus" and "failed" not in row
    }


# ------------------------------------------------------------------ the sweep


def run_sweep(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Five strengths over five photographs, one variable and nothing else.

    Image-major so the flat-block mask can be taken once from that image's
    `dn0.00` output and reused for its other four arms - the Phase 3B
    correction. A mask recomputed per arm would compare different blocks,
    because denoising changes which blocks are flattest.
    """
    import torch

    from app.core.config import Settings
    from app.inference.model_manager import ModelManager
    from app.services.model_service import ModelService

    settings = Settings()
    models = ModelService(settings)
    manager = ModelManager(settings, models)
    target = manager.target

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image in images:
        source = _load(image)
        shared_mask: Any = None
        print(f"\n  --- {image.category}  {source.shape[1]}x{source.shape[0]} ---", flush=True)

        for strength in STRENGTHS:
            upscaler = manager.get(MODEL, denoise_strength=strength)
            # What the manager actually did with the value, not what was asked.
            blend = manager._resolve_blend(models.get(MODEL), strength)

            if target.is_cuda:
                torch.cuda.reset_peak_memory_stats()

            started = time.perf_counter()
            out = upscaler.upscale(source, tile=settings.tile_size, tile_pad=settings.tile_pad)
            elapsed = (time.perf_counter() - started) * 1000.0

            report = upscaler.last_report
            assert report is not None, "a completed upscale always records a report"
            config = RunConfig(
                device=report.device,
                fp16=report.fp16,
                tile_size=report.tile_size,
                tiles=report.tiles,
                tile_reduced=report.tile_size_reduced,
                peak_vram_mb=(
                    float(torch.cuda.max_memory_allocated()) / 1024 / 1024
                    if target.is_cuda
                    else None
                ),
            )

            tiles = ycrcb_tiles(out)
            if strength == BASELINE:
                shared_mask = flat_mask(tiles)
            stats = chroma_metrics(tiles, shared_mask)
            luma: ImageMetrics = measure(out)
            size, digest = encoded_png(out)

            _write_crops(image.category, arm(strength), out)

            rows.append(
                {
                    "experiment": "sweep",
                    "model": MODEL,
                    "arm": arm(strength),
                    "denoise_passed": strength,
                    "dni_blend_resolved": blend,
                    "dni_note": (
                        "no blend - plain x4v3 weights"
                        if blend is None
                        else ("fully the wdn weights" if blend == 0.0 else "interpolated")
                    ),
                    "category": image.category,
                    "image": image.name,
                    "input": f"{source.shape[1]}x{source.shape[0]}",
                    "output": f"{out.shape[1]}x{out.shape[0]}",
                    "luma": luma.as_dict(),
                    "chroma_metrics": stats.as_dict(),
                    "config": config.as_dict(),
                    "sr_ms": elapsed,
                    "png_bytes": size,
                    "output_sha256": digest,
                }
            )
            print(
                f"    dn{strength:.2f}  blend={blend!s:5}  {elapsed / 1000:5.1f}s  "
                f"tile={config.tile_size} reduced={config.tile_reduced} "
                f"peak={config.peak_vram_mb:.0f}MiB  "
                f"hf={luma.high_frequency_ratio:.5f} flat={luma.flat_noise:.5f}  "
                f"sha={digest[:12]}",
                flush=True,
            )
            del out, tiles

    freed = manager.release()
    print(f"\n  released {freed} model(s)", flush=True)
    return rows


def _write_crops(category: str, name: str, out: np.ndarray[Any, Any]) -> None:
    """Save each inspection region of one arm, at 1:1 on the 4x result."""
    from PIL import Image as PILImage

    for region, (cx, cy, size) in REGIONS.get(category, {}).items():
        span = size * SCALE
        left = min(max(0, cx * SCALE - span // 2), max(0, out.shape[1] - span))
        top = min(max(0, cy * SCALE - span // 2), max(0, out.shape[0] - span))
        crop = out[top : top + span, left : left + span]
        PILImage.fromarray(crop).save(CROPS_DIR / f"{category}__{region}__{name}.png")


# ------------------------------------------------------------- blind strips


def build_strips(seed: int = 20260909) -> list[dict[str, Any]]:
    """Comparison strips whose panel order is shuffled per region.

    Deliberately not 0.00 -> 1.00 left to right: a monotone ladder tells the eye
    what it is about to see, and Phase 3B showed how much expectation shapes a
    visual verdict. The mapping goes in the measurements file, not on the image.
    """
    from PIL import Image as PILImage

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for category, regions in REGIONS.items():
        for region in regions:
            available = [
                arm(s)
                for s in STRENGTHS
                if (CROPS_DIR / f"{category}__{region}__{arm(s)}.png").is_file()
            ]
            if len(available) < 2:
                continue

            order = list(available)
            generator.shuffle(order)

            panels = []
            for name in order:
                panel = np.asarray(
                    PILImage.open(CROPS_DIR / f"{category}__{region}__{name}.png").convert("RGB")
                ).copy()
                panel[:, -4:] = 255
                panels.append(panel)

            out_name = f"blind__{category}__{region}.png"
            PILImage.fromarray(np.concatenate(panels, axis=1)).save(CROPS_DIR / out_name)

            rows.append(
                {
                    "experiment": "strips",
                    "category": category,
                    "region": region,
                    "file": out_name,
                    "panel_order": order,
                }
            )
            print(f"  {category:18} {region:18} -> {out_name}", flush=True)

    return rows


# ------------------------------------------------------------------ metadata


def provenance(images: list[CorpusImage]) -> dict[str, Any]:
    import cv2
    import torch

    from app.core.config import Settings
    from app.services.model_service import ModelService

    settings = Settings()
    models = ModelService(settings)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "corpus_root": str(DEFAULT_CORPUS_DIR),
        "corpus_images": [f"{i.category}/{i.name}" for i in images],
        "model": MODEL,
        "checkpoint_sha256": checkpoint_digests(models),
        "scale": SCALE,
        "strengths": list(STRENGTHS),
        "baseline_arm": arm(BASELINE),
        "tile_size": settings.tile_size,
        "tile_pad": settings.tile_pad,
        "sharpen_strength": 0.0,
        "output_format": "png (lossless); metrics from the array, size and hash from the encode",
        "target_resolution": None,
        "chroma_prefilter": None,
        "x4plus_reference": (
            "reused from phase3c_measurements.json (same 5 photographs, 4x, PNG, sharpening 0)"
        ),
        "dni_semantics": {
            "None": "no blend - plain x4v3 weights",
            "0.00": "blend 0.00 - fully the wdn weights",
            "0.25": "blend 0.25",
            "0.50": "blend 0.50",
            "0.75": "blend 0.75",
            "1.00": "resolves to blend=None - plain x4v3 weights, identical to passing None",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--strips", action="store_true")
    args = parser.parse_args(argv)

    images = load_corpus()
    if not images:
        print("no corpus images", file=sys.stderr)
        return 1
    load_manifest()

    rows: list[dict[str, Any]] = []
    if MEASUREMENTS.is_file():
        rows = list(json.loads(MEASUREMENTS.read_text(encoding="utf-8")).get("rows", []))

    def replace(experiment: str, new: list[dict[str, Any]]) -> None:
        nonlocal rows
        rows = [r for r in rows if r.get("experiment") != experiment] + new

    if args.sweep:
        print("\n=== sweep: 5 photographs x 5 denoise strengths ===", flush=True)
        replace("sweep", run_sweep(images))
    if args.strips:
        print("\n=== blind comparison strips ===", flush=True)
        replace("strips", build_strips())

    if not rows:
        parser.print_help()
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(
        json.dumps({"provenance": provenance(images), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {MEASUREMENTS} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
