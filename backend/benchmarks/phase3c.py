"""Phase 3C pilot: does a different restoration network give believable detail?

    python -m benchmarks.phase3c --probe   # HAT load / VRAM / tiling safety only
    python -m benchmarks.phase3c --pilot   # 5 photographs x 4 arms
    python -m benchmarks.phase3c --strips  # blind comparison strips from the crops

The product complaint is perceptual - eyes, skin, hair and fine texture do not
look real - so this is a perceptual benchmark. Every arm is run through the
**production** `RealEsrganUpscaler.upscale`, so tiling, the OOM ladder and the
progress/report path are identical across arms and nothing is measured under a
private configuration.

Crops are written per (image, region, arm) and named so that a comparison strip
can be assembled **without the arm names being visible in the image**. The
visual verdict is formed from those strips first; metrics are computed and read
afterwards. Phase 3B is the reason for that ordering: there, `hf_ratio` rose
while the picture visibly got grainier, so a metric-led verdict would have been
backwards.
"""

from __future__ import annotations

import argparse
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
from benchmarks.detail_models import CANDIDATES, candidate, load_hat
from benchmarks.metrics import ImageMetrics, measure
from benchmarks.phase3a import _load
from benchmarks.phase3b import chroma_metrics, flat_mask, png_bytes, ycrcb_tiles

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase3c_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase3c-crops"

SCALE = 4

#: Regions to inspect, in **source** pixel coordinates, as (centre_x, centre_y,
#: size). Verified by eye against the source before the pilot ran - the portrait
#: regions land on the iris, the sclera, the lip surface, individual hair
#: strands and open cheek skin respectively.
#:
#: The portrait carries five of the nine required regions because that is where
#: the product complaint lives. `saturated colour` is the foliage region (bright
#: sunlit green), and `fine texture` is the landscape rock.
REGIONS: dict[str, dict[str, tuple[int, int, int]]] = {
    "portrait-skin": {
        "eye-left": (315, 510, 140),
        "eye-right": (929, 510, 140),
        "hair": (180, 120, 160),
        "lips": (645, 1185, 160),
        "skin-cheek": (1019, 900, 160),
        "identity-face": (620, 760, 640),
    },
    "landscape-detail": {"fine-texture": (760, 420, 180)},
    "foliage-texture": {"saturated-foliage": (500, 760, 180)},
    "text-signage": {"text": (600, 300, 180)},
    "low-light-noise": {"dark-noisy": (900, 450, 180)},
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Exactly how one arm was executed. Recorded per run, never assumed."""

    device: str
    fp16: bool
    tile_size: int
    tiles: int
    tile_reduced: bool
    peak_vram_mb: float | None
    #: True when the arm could not run at the production precision.
    precision_forced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _upscaler_for(arm: str, manager: Any, target: Any) -> tuple[Any, dict[str, Any]]:
    """The upscaler for one arm, plus whatever provenance it carries."""
    entry = candidate(arm)

    if arm == "hat":
        loaded = load_hat(target)
        return loaded.upscaler, {
            "params_millions": loaded.params_millions,
            "vram_after_load_mb": loaded.vram_after_load_mb,
            **loaded.extras,
        }

    upscaler = manager.get(entry.model_id, denoise_strength=entry.denoise)
    return upscaler, {"params_millions": None, "precision_forced": False}


def run_pilot(images: list[CorpusImage], arms: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every arm over every photograph, through the production tiling path."""
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

    for arm in arms:
        entry = candidate(arm)
        print(f"\n  --- {arm}  ({entry.model_id}, denoise={entry.denoise}) ---", flush=True)

        try:
            upscaler, provenance_extra = _upscaler_for(arm, manager, target)
        except Exception as exc:  # a candidate that cannot load is reported, not fatal
            print(f"    LOAD FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            rows.append(
                {
                    "experiment": "pilot",
                    "arm": arm,
                    "failed": "load",
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                }
            )
            continue

        for image in images:
            source = _load(image)

            if target.is_cuda:
                torch.cuda.reset_peak_memory_stats()

            started = time.perf_counter()
            try:
                out = upscaler.upscale(source, tile=settings.tile_size, tile_pad=settings.tile_pad)
            except Exception as exc:
                print(
                    f"    {image.category:18} RUN FAILED: {type(exc).__name__}: "
                    f"{getattr(exc, 'technical', str(exc))[:160]}",
                    flush=True,
                )
                rows.append(
                    {
                        "experiment": "pilot",
                        "arm": arm,
                        "category": image.category,
                        "failed": "inference",
                        "error": f"{type(exc).__name__}: "
                        f"{getattr(exc, 'technical', str(exc))[:400]}",
                    }
                )
                continue
            elapsed = (time.perf_counter() - started) * 1000.0

            report = upscaler.last_report
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
                precision_forced=bool(provenance_extra.get("precision_forced", False)),
            )

            tiles = ycrcb_tiles(out)
            luma: ImageMetrics = measure(out)
            stats = chroma_metrics(tiles, flat_mask(tiles))
            size = png_bytes(out)

            _write_crops(image.category, arm, out)

            rows.append(
                {
                    "experiment": "pilot",
                    "arm": arm,
                    "model": entry.model_id,
                    "denoise": entry.denoise,
                    "category": image.category,
                    "image": image.name,
                    "input": f"{source.shape[1]}x{source.shape[0]}",
                    "output": f"{out.shape[1]}x{out.shape[0]}",
                    "luma": luma.as_dict(),
                    "chroma_metrics": stats.as_dict(),
                    "config": config.as_dict(),
                    "sr_ms": elapsed,
                    "png_bytes": size,
                    "provenance": provenance_extra,
                }
            )
            print(
                f"    {image.category:18} {elapsed / 1000:7.1f}s  "
                f"tile={config.tile_size} tiles={config.tiles} "
                f"reduced={config.tile_reduced} fp16={config.fp16} "
                f"peak={config.peak_vram_mb:.0f}MiB  out={out.shape[1]}x{out.shape[0]}",
                flush=True,
            )
            del out, tiles

        if arm == "hat":
            # Free the transformer before the next arm loads; 2.9 GiB peak on a
            # 4 GiB card leaves no room for two resident models.
            del upscaler
            if target.is_cuda:
                torch.cuda.empty_cache()

    manager.release()
    return rows


def _write_crops(category: str, arm: str, out: np.ndarray[Any, Any]) -> None:
    """Save each region of one arm's output, at 1:1 on the 4x result."""
    from PIL import Image as PILImage

    for region, (cx, cy, size) in REGIONS.get(category, {}).items():
        span = size * SCALE
        left = max(0, cx * SCALE - span // 2)
        top = max(0, cy * SCALE - span // 2)
        left = min(left, max(0, out.shape[1] - span))
        top = min(top, max(0, out.shape[0] - span))
        crop = out[top : top + span, left : left + span]
        PILImage.fromarray(crop).save(CROPS_DIR / f"{category}__{region}__{arm}.png")


# ------------------------------------------------------------- blind strips


def build_strips(arms: tuple[str, ...], seed: int = 20260909) -> list[dict[str, Any]]:
    """Assemble comparison strips whose panel order is shuffled per region.

    The mapping is written to the measurements file, not drawn on the image, so
    the strip can be inspected without knowing which arm is which. That is the
    whole point: Phase 3B showed that knowing what *should* win biases what you
    see.
    """
    from PIL import Image as PILImage

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for category, regions in REGIONS.items():
        for region in regions:
            available = [
                arm for arm in arms if (CROPS_DIR / f"{category}__{region}__{arm}.png").is_file()
            ]
            if len(available) < 2:
                continue

            order = list(available)
            generator.shuffle(order)

            panels = []
            for arm in order:
                panel = np.asarray(
                    PILImage.open(CROPS_DIR / f"{category}__{region}__{arm}.png").convert("RGB")
                ).copy()
                panel[:, -4:] = 255
                panels.append(panel)

            strip = np.concatenate(panels, axis=1)
            name = f"blind__{category}__{region}.png"
            PILImage.fromarray(strip).save(CROPS_DIR / name)

            rows.append(
                {
                    "experiment": "strips",
                    "category": category,
                    "region": region,
                    "file": name,
                    "panel_order": order,
                    "panel_count": len(order),
                }
            )
            print(f"  {category:18} {region:18} -> {name}  ({len(order)} panels)", flush=True)

    return rows


# ------------------------------------------------------------------ metadata


def provenance(images: list[CorpusImage]) -> dict[str, Any]:
    import cv2
    import torch

    from app.core.config import Settings

    settings = Settings()
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_mib": (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            if torch.cuda.is_available()
            else None
        ),
        "corpus_root": str(DEFAULT_CORPUS_DIR),
        "corpus_images": [f"{i.category}/{i.name}" for i in images],
        "scale": SCALE,
        "tile_size": settings.tile_size,
        "tile_pad": settings.tile_pad,
        "sharpen_strength": 0.0,
        "output_format": "png (lossless); metrics from the array, size from the encode",
        "candidates": [c.as_dict() for c in CANDIDATES],
        "regions": {k: {r: list(v) for r, v in rs.items()} for k, rs in REGIONS.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--strips", action="store_true")
    parser.add_argument(
        "--arms", default="x4plus,v3-dn0,v3-dn1,hat", help="comma-separated arm names"
    )
    args = parser.parse_args(argv)

    images = load_corpus()
    if not images:
        print("no corpus images", file=sys.stderr)
        return 1
    load_manifest()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    rows: list[dict[str, Any]] = []
    if MEASUREMENTS.is_file():
        rows = list(json.loads(MEASUREMENTS.read_text(encoding="utf-8")).get("rows", []))

    def replace(experiment: str, new: list[dict[str, Any]]) -> None:
        nonlocal rows
        rows = [r for r in rows if r.get("experiment") != experiment] + new

    if args.pilot:
        print("\n=== pilot: 5 photographs x arms ===", flush=True)
        replace("pilot", run_pilot(images, arms))
    if args.strips:
        print("\n=== blind comparison strips ===", flush=True)
        replace("strips", build_strips(arms))

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
