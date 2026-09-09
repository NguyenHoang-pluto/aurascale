"""Phase 4 F3: does AuraScale's sharpening improve real photographs?

    python -m benchmarks.phase4_f3 --sweep    # 5 photographs x 6 strengths
    python -m benchmarks.phase4_f3 --strips   # blind, shuffled comparison strips

Research only. Nothing here changes production behaviour. The production
sharpening default stays 0.0, `DEFAULT_DENOISE` stays 1.0 and `CREATIVE_DENOISE`
stays 0.25; this module reads the implementation and never writes to it.

Why this is worth measuring
---------------------------

`unsharp_mask` is the most carefully built post-process in the codebase - a
scale-aware radius, a luma-only correction so no colour fringe appears, a dead
zone so flat regions stay quiet, a ceiling so edges cannot grow a pronounced
rim, and strip processing so a 195 MP result costs the same as a 12 MP one.

And **not one benchmark has ever measured it.** Every phase from 2 onward pinned
`sharpen_strength = 0.0` - correctly, to isolate whatever else it was studying -
so the one detail-recovery lever the pipeline already owns is both disabled by
default and unevaluated. The Phase 4 consolidation identified that as the
largest untested component of the current pipeline.

The measurement it is subjected to
----------------------------------

Sharpening raises high-frequency energy by construction. That is what it does,
and it is therefore **not evidence that it helped**. An unsharp mask can raise
`hf_ratio` while adding a bright rim to every edge and turning film grain into
crunch. So the metrics here are read second, after a blind visual pass, and the
report is written to separate "looks sharper" from "contains more believable
detail". Phase 3B is the precedent: there `hf_ratio` rose while the picture
visibly got worse.

Isolating sharpening
--------------------

The neural pass runs **once per photograph** and its output is cached in memory;
all six strengths are applied to that identical array. So no arm differs by
model variance, tiling or precision - only by the strength. Five SR passes, not
thirty.

`RealESRGAN_x4plus` is the model, and that choice removes the other confound:
it has no `denoise_pair`, so `_denoise_for` drops any denoise value and there is
no denoising stage at all. Denoise is not "held constant" here, it is
structurally absent, which is stronger.
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
from benchmarks.research_utils import (
    CROP_REGIONS,
    chroma_metrics,
    flat_mask,
    load_rgb,
    ycrcb_tiles,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase4_f3_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase4-f3-crops"

MODEL = "RealESRGAN_x4plus"
SCALE = 4

#: The ladder. All inside the [0, 1] contract `unsharp_mask` validates; the
#: implementation multiplies by SHARPEN_MAX_AMOUNT = 1.5, so 0.75 drives the
#: correction at 1.125x and 1.00 would be the maximum the API allows.
#:
#: Weighted toward the low end on purpose. If sharpening has a defensible
#: default it will be a cautious one, and resolving 0.15 from 0.25 matters more
#: than resolving 0.75 from 1.00 - which the eye can already tell apart.
STRENGTHS: tuple[float, ...] = (0.00, 0.15, 0.25, 0.35, 0.50, 0.75)

#: The off arm. `unsharp_mask` returns the caller's array unchanged at 0, so
#: this is the cached neural output itself and not a re-encoded copy of it.
BASELINE = 0.00

#: Regions to inspect. The ten shared with Phase 3C, plus two chosen for this
#: phase because sharpening's characteristic failure is at strong edges and
#: neither existing region is a maximally hard boundary:
#:
#:   * `snow-rock-edge` - dark rock silhouetted against sunlit snow, the highest
#:     local contrast in the corpus, where a halo would be unmissable;
#:   * `lamp-edge` - the bridge's string of point lamps against a dark
#:     structure, which tests ringing and noise amplification at once.
#:
#: Both were checked against the sources before the sweep ran.
EXTRA_REGIONS: dict[str, dict[str, tuple[int, int, int]]] = {
    "landscape-detail": {"snow-rock-edge": (1365, 855, 180)},
    "low-light-noise": {"lamp-edge": (1200, 645, 180)},
}


def regions_for(category: str) -> dict[str, tuple[int, int, int]]:
    """The shared inspection regions for a category, plus this phase's own."""
    return {**CROP_REGIONS.get(category, {}), **EXTRA_REGIONS.get(category, {})}


def arm(strength: float) -> str:
    return f"sh{strength:.2f}"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """How the neural pass ran. One per photograph - all arms share it."""

    device: str
    fp16: bool
    tile_size: int
    tiles: int
    tile_reduced: bool
    peak_vram_mb: float | None
    sr_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def encoded_png(image: np.ndarray[Any, Any]) -> tuple[int, str]:
    """PNG size and hash. Lossless, so the metrics are unaffected by it."""
    import cv2

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return -1, ""
    raw = buffer.tobytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def rss_mb() -> float | None:
    """Current resident set size, when psutil can report it."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return None

    return float(psutil.Process().memory_info().rss) / 1024 / 1024


# ------------------------------------------------------------------ the sweep


def run_sweep(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Six strengths over five photographs, from one neural pass each.

    Image-major, and the flat-block mask is taken once from the unsharpened arm
    and reused for the rest - the Phase 3A correction, since sharpening changes
    which blocks are flattest and a per-arm mask would compare different
    regions.
    """
    import torch

    from app.core.config import Settings
    from app.inference.model_manager import ModelManager
    from app.services.enhancement_service import (
        SHARPEN_DETAIL_CEILING,
        SHARPEN_MAX_AMOUNT,
        SHARPEN_NOISE_FLOOR,
        sharpen_sigma,
        unsharp_mask,
    )
    from app.services.model_service import ModelService

    settings = Settings()
    models = ModelService(settings)
    manager = ModelManager(settings, models)
    target = manager.target

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    sigma = sharpen_sigma(SCALE)
    print(
        f"  sharpening: sigma={sigma} (0.75 x scale, clamped), "
        f"amount = strength x {SHARPEN_MAX_AMOUNT}, "
        f"dead zone +-{SHARPEN_NOISE_FLOOR}, ceiling +-{SHARPEN_DETAIL_CEILING}",
        flush=True,
    )

    for image in images:
        source = load_rgb(image)

        # One neural pass. Every arm below sharpens this identical array, so
        # nothing between arms can differ except the strength.
        if target.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        upscaler = manager.get(MODEL, denoise_strength=None)
        started = time.perf_counter()
        neural = upscaler.upscale(source, tile=settings.tile_size, tile_pad=settings.tile_pad)
        sr_ms = (time.perf_counter() - started) * 1000.0

        report = upscaler.last_report
        assert report is not None, "a completed upscale always records a report"
        config = RunConfig(
            device=report.device,
            fp16=report.fp16,
            tile_size=report.tile_size,
            tiles=report.tiles,
            tile_reduced=report.tile_size_reduced,
            peak_vram_mb=(
                float(torch.cuda.max_memory_allocated()) / 1024 / 1024 if target.is_cuda else None
            ),
            sr_ms=sr_ms,
        )
        print(
            f"\n  --- {image.category}  {source.shape[1]}x{source.shape[0]}"
            f" -> {neural.shape[1]}x{neural.shape[0]}  "
            f"SR {sr_ms / 1000:.1f}s tile={config.tile_size} reduced={config.tile_reduced} ---",
            flush=True,
        )

        shared_mask: Any = None

        for strength in STRENGTHS:
            before = rss_mb()
            started = time.perf_counter()
            out = unsharp_mask(neural, strength, scale=SCALE)
            sharpen_ms = (time.perf_counter() - started) * 1000.0
            after = rss_mb()

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
                    "sharpen_strength": strength,
                    "sharpen_amount": strength * SHARPEN_MAX_AMOUNT,
                    "sharpen_sigma": sigma,
                    "denoise": None,
                    "denoise_note": f"{MODEL} has no denoise_pair; the stage does not exist",
                    "category": image.category,
                    "image": image.name,
                    "input": f"{source.shape[1]}x{source.shape[0]}",
                    "output": f"{out.shape[1]}x{out.shape[0]}",
                    "luma": luma.as_dict(),
                    "chroma_metrics": stats.as_dict(),
                    "config": config.as_dict(),
                    "sharpen_ms": sharpen_ms,
                    "rss_delta_mb": None if before is None or after is None else after - before,
                    "png_bytes": size,
                    "output_sha256": digest,
                }
            )
            print(
                f"    sh{strength:.2f}  sharpen {sharpen_ms:7.1f} ms  "
                f"hf={luma.high_frequency_ratio:.5f} contrast={luma.local_contrast:.5f} "
                f"p95={luma.sobel_p95:.4f} flat={luma.flat_noise:.5f} "
                f"over={luma.edge_overshoot:.5f}  png={size / 1e6:.1f}MB",
                flush=True,
            )
            if strength != BASELINE:
                del out
            del tiles

        del neural

    freed = manager.release()
    print(f"\n  released {freed} model(s)", flush=True)
    return rows


def _write_crops(category: str, name: str, out: np.ndarray[Any, Any]) -> None:
    """Save each inspection region of one arm, at 1:1 on the 4x output."""
    from PIL import Image as PILImage

    for region, (cx, cy, size) in regions_for(category).items():
        span = size * SCALE
        left = min(max(0, cx * SCALE - span // 2), max(0, out.shape[1] - span))
        top = min(max(0, cy * SCALE - span // 2), max(0, out.shape[0] - span))
        crop = out[top : top + span, left : left + span]
        PILImage.fromarray(crop).save(CROPS_DIR / f"{category}__{region}__{name}.png")


# -------------------------------------------------------------- blind strips


def build_strips(seed: int = 20260910) -> list[dict[str, Any]]:
    """Comparison strips whose panel order is shuffled per region.

    Deliberately not 0.00 -> 0.75 left to right. A monotone ladder tells the eye
    which way "more" is, and with sharpening that is exactly the bias to avoid:
    the whole question is whether more is better, and a reader who knows the
    direction will find it.
    """
    from PIL import Image as PILImage

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for category in sorted({*CROP_REGIONS, *EXTRA_REGIONS}):
        for region in regions_for(category):
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
    from app.services.enhancement_service import (
        SHARPEN_DETAIL_CEILING,
        SHARPEN_MAX_AMOUNT,
        SHARPEN_NOISE_FLOOR,
        SHARPEN_SIGMA_PER_SCALE,
        SHARPEN_SIGMA_RANGE,
        sharpen_sigma,
    )

    settings = Settings()

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
        "scale": SCALE,
        "strengths": list(STRENGTHS),
        "baseline_arm": arm(BASELINE),
        "tile_size": settings.tile_size,
        "tile_pad": settings.tile_pad,
        "denoise": None,
        "denoise_note": (
            f"{MODEL} declares no denoise_pair, so _denoise_for drops any value and no "
            "denoising stage runs. Denoise is structurally absent rather than held constant."
        ),
        "output_format": "png (lossless); metrics from the array, size and hash from the encode",
        "neural_pass": "run once per photograph and cached; every arm sharpens that same array",
        "sharpening_semantics": {
            "range": "[0, 1], validated by unsharp_mask",
            "amount": f"strength x SHARPEN_MAX_AMOUNT ({SHARPEN_MAX_AMOUNT})",
            "sigma": f"clamp(SHARPEN_SIGMA_PER_SCALE x scale, {SHARPEN_SIGMA_RANGE}) "
            f"= {sharpen_sigma(SCALE)} at {SCALE}x",
            "sigma_per_scale": SHARPEN_SIGMA_PER_SCALE,
            "dead_zone": SHARPEN_NOISE_FLOOR,
            "ceiling": SHARPEN_DETAIL_CEILING,
            "channel": "correction computed on Rec.709 luma and added equally to R, G and B",
            "zero": "strength 0 returns the caller's array unchanged - a true no-op",
        },
        "regions": {
            category: {name: list(box) for name, box in regions_for(category).items()}
            for category in sorted({*CROP_REGIONS, *EXTRA_REGIONS})
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
        print("\n=== sweep: 5 photographs x 6 sharpening strengths ===", flush=True)
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
