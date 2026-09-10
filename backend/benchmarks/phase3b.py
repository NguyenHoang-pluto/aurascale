"""Phase 3B: does a chroma-only prefilter add anything on top of Real-ESRGAN DNI?

    python -m benchmarks.phase3b --all
    python -m benchmarks.phase3b --matrix    # the factorial, through the model
    python -m benchmarks.phase3b --scaling   # uint8 / memory / runtime probe
    python -m benchmarks.phase3b --crops     # D / A / B / C strips per photograph

Phase 3A ended with one promising result and one explicit gap. The result:
chroma-only prefiltering removed 7.5-37.8 % of chroma noise at 0.0 % luma cost,
for about 187 ms at 16 MP. The gap: it was never run *together with* DNI, so
"promising" rested on an untested assumption that the two do not overlap.

Two structural facts shape the whole design
-------------------------------------------

**1. `RealESRGAN_x4plus` has no `denoise_pair`.** `_denoise_for` drops any
denoise value handed to it. So for Standard's model there is no DNI at all, and
arms A (DNI only) and C (chroma + DNI) *do not exist*. The four-arm D/B/A/C
comparison is only measurable on `realesr-general-x4v3`, which is Creative's
model. That is not a gap in the experiment; it is the shape of the product, and
it is why the model-interaction question has a different answer from the
interaction question.

**2. For `realesr-general-x4v3`, "no DNI" is not the un-denoised state.**
`_resolve_blend` returns `None` - meaning no blend, the plain x4v3 weights -
for `denoise_strength=None` *and* for `1.0`. The two are the same code path.
And Phase 2.5 established empirically that higher values denoise *harder*, so
the plain weights are the strongest-denoising state, not the weakest.

The least-denoised output the model can produce is therefore `dni=0.00`, which
blends fully to the `wdn` counterpart. That is what this phase uses as the D
reference, matching Phase 2.5. Calling `None` "no denoise" would have made the
baseline the most-denoised arm and inverted every conclusion.

Methodology fix carried over from Phase 3A
------------------------------------------

Phase 3A measured chroma sigma by recomputing the flat-block mask on each
filtered image. Filters that change luma change which blocks are flattest, so
before and after sampled *different blocks*, and the chroma figures for
`nlm-*` and `bilateral-*` had to be withheld as confounded.

Here the mask is computed **once, from the baseline arm's output for that
image**, and the identical block indices are reused for every arm of that
image. Every chroma comparison below is over the same regions.
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
from benchmarks.metrics import ImageMetrics, measure, tile_boxes
from benchmarks.phase3a import _load
from benchmarks.prefilter import (
    NOISE_FLAT_PERCENTILE,
    _blocks_of,
    _immerkaer_sigma,
    apply_filter,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase3b_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase3b-crops"

#: The chroma ladder. `off` is the absence of the filter rather than a fourth
#: setting of it, and is what every delta within a config is measured against.
#: The three strengths are Phase 3A's, unchanged, so the two phases' arms mean
#: the same thing.
CHROMA_ARMS: tuple[str, ...] = ("off", "chroma-weak", "chroma-medium", "chroma-strong")

SCALE = 4


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """One model at one DNI setting, and what it is in production terms."""

    model: str
    #: None where the model has no denoise pair. Distinct from 0.0: 0.0 is a
    #: blend the model supports, None is the absence of the control entirely.
    dni: float | None
    label: str
    note: str
    #: Which chroma rungs to run for this config. Restricted for x4plus - see
    #: CONFIGS - because its cells cost 15-25x a v3 cell on this hardware.
    chroma_arms: tuple[str, ...] = CHROMA_ARMS

    @property
    def key(self) -> str:
        return f"{self.label}|dni-{'na' if self.dni is None else f'{self.dni:.2f}'}"

    @property
    def is_reference(self) -> bool:
        """The least-denoised arm of this model - what D means here."""
        return self.dni in (None, 0.0)


#: Run order. Grouped by model and DNI so the manager loads five weight sets in
#: total rather than one per cell.
#:
#: Every v3 rung is production-reachable: 0.25 is `CREATIVE_DENOISE`, 1.00 is
#: `DEFAULT_DENOISE` (and the plain weights), 0.50 and 0.75 are settings a user
#: can select today, and 0.00 is the least-denoised end and Phase 2.5's
#: baseline.
CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        "realesr-general-x4v3",
        0.0,
        "v3",
        "D reference: fully the wdn weights, the least-denoised output v3 can make.",
    ),
    ModelConfig("realesr-general-x4v3", 0.25, "v3", "CREATIVE_DENOISE - what Creative ships."),
    ModelConfig("realesr-general-x4v3", 0.50, "v3", "Mid ladder rung."),
    ModelConfig("realesr-general-x4v3", 0.75, "v3", "Upper ladder rung."),
    ModelConfig(
        "realesr-general-x4v3",
        1.0,
        "v3",
        "DEFAULT_DENOISE. Identical code path to denoise=None - no blend, plain x4v3.",
    ),
    # Last, and with the ladder trimmed to the baseline and the candidate rung.
    #
    # Two measured reasons, both recorded rather than assumed. Its cells cost
    # 26-47 s against v3's 1.8 s; and it hits the OOM ladder on a 4 GB card,
    # falling back from tile 256 (35 tiles) to tile 128 (140 tiles) once VRAM
    # is occupied, which makes its *runtime* figures unusable for comparison
    # anyway. It contributes only to the model-interaction question - it has no
    # DNI, so it takes no part in the interaction proper - and `chroma-medium`
    # against `off` answers that. The full four-rung ladder is measured on v3,
    # where it is affordable.
    ModelConfig(
        "RealESRGAN_x4plus",
        None,
        "x4plus",
        "Standard's model. No denoise_pair, so DNI is unavailable by construction.",
        chroma_arms=("off", "chroma-medium"),
    ),
)


# ------------------------------------------------------------ chroma metrics
#
# `benchmarks/metrics.py` is entirely Rec.709 luma and is left untouched - it
# is shared with three earlier reports and changing it would silently move
# their numbers. What it cannot see is exactly what this phase is about, so the
# chroma statistics live here, built from `prefilter`'s primitives and sampled
# over `metrics.tile_boxes` so both metric families read the same regions.


def ycrcb_tiles(rgb: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    """The 16 deterministic tiles of an image, as float64 YCrCb.

    The same boxes `metrics.measure` uses, from the same seed, so the luma and
    chroma statistics describe identical regions. Sampling rather than whole-
    frame also keeps the cost flat: a 4x output is 32 MP and blocking three
    full channels of it costs more than the SR pass that produced it.
    """
    import cv2

    height, width = rgb.shape[:2]
    return [
        cv2.cvtColor(
            np.ascontiguousarray(rgb[top:bottom, left:right, :3]), cv2.COLOR_RGB2YCrCb
        ).astype(np.float64)
        for left, top, right, bottom in tile_boxes(width, height)
    ]


def flat_mask(tiles: list[np.ndarray[Any, Any]], percentile: float = NOISE_FLAT_PERCENTILE) -> Any:
    """Which 32 px blocks across the sampled tiles are flat, by luma variance.

    Computed once from the baseline arm and then reused for every other arm of
    that image. This is the Phase 3A correction: a filter that changes luma
    changes which blocks are flattest, so a mask recomputed per arm silently
    compares different regions.
    """
    variance = np.concatenate([_blocks_of(tile[:, :, 0]).var(axis=(1, 2)) for tile in tiles])
    return variance <= np.percentile(variance, percentile)


@dataclass(frozen=True, slots=True)
class ChromaMetrics:
    """Colour-channel statistics the luma metrics are structurally blind to."""

    #: Noise sigma in Y over the shared mask. A chroma-only filter must leave
    #: this untouched - it is the control on the entire premise.
    sigma_luma: float
    #: The same in Cr/Cb, averaged. What chroma filtering is meant to lower.
    sigma_chroma: float
    #: Mean Sobel magnitude on Cr/Cb. Colour-boundary preservation: chroma
    #: smearing and colour bleeding show up here as a fall, and it is what
    #: stops "less chroma noise" from being read as "better".
    chroma_gradient: float
    #: The strongest colour edges specifically.
    chroma_gradient_p95: float
    blocks_used: int
    blocks_total: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sobel_magnitude(channel: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    import cv2

    gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    magnitude: np.ndarray[Any, Any] = np.hypot(gx, gy)[1:-1, 1:-1]
    return magnitude


def chroma_metrics(tiles: list[np.ndarray[Any, Any]], mask: Any) -> ChromaMetrics:
    """Chroma statistics over a *given* mask, so arms are directly comparable."""
    luma_blocks = np.concatenate([_blocks_of(tile[:, :, 0]) for tile in tiles])
    cr_blocks = np.concatenate([_blocks_of(tile[:, :, 1]) for tile in tiles])
    cb_blocks = np.concatenate([_blocks_of(tile[:, :, 2]) for tile in tiles])

    gradient = np.concatenate(
        [_sobel_magnitude(tile[:, :, channel]).ravel() for tile in tiles for channel in (1, 2)]
    )

    return ChromaMetrics(
        sigma_luma=_immerkaer_sigma(luma_blocks[mask]),
        sigma_chroma=(_immerkaer_sigma(cr_blocks[mask]) + _immerkaer_sigma(cb_blocks[mask])) / 2.0,
        chroma_gradient=float(gradient.mean()),
        chroma_gradient_p95=float(np.percentile(gradient, 95)),
        blocks_used=int(np.count_nonzero(mask)),
        blocks_total=int(mask.size),
    )


def png_bytes(rgb: np.ndarray[Any, Any]) -> int:
    """Encoded PNG size. Lossless, so the metrics are unaffected by it.

    Recorded because Phase 2.5 found it a useful independent read on how much
    detail survives: an arm that removes structure compresses smaller, and that
    is one number no proxy metric can argue with.
    """
    import cv2

    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return int(buffer.nbytes) if ok else -1


# ------------------------------------------------------------- 1. the matrix


def run_matrix(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Every chroma rung crossed with every DNI setting, through the model.

    Config-major so the model manager loads each weight set once. Within a
    config the `off` arm runs first for each image, because its output is what
    the shared flat mask is derived from.
    """
    from app.core.config import Settings
    from app.services.enhancement_service import EnhancementRequest, EnhancementService

    service = EnhancementService(Settings())
    rows: list[dict[str, Any]] = []

    for config in CONFIGS:
        print(f"\n  --- {config.key}  ({config.note}) ---", flush=True)

        for image in images:
            source = _load(image)
            shared_mask: Any = None

            for chroma in config.chroma_arms:
                if chroma == "off":
                    filtered, filter_ms = source, 0.0
                else:
                    filtered, cost = apply_filter(source, chroma)
                    filter_ms = cost.elapsed_ms
                    # The whole large-image argument rests on this staying uint8.
                    assert filtered.dtype == np.uint8, filtered.dtype
                    assert filtered.shape == source.shape, filtered.shape

                started = time.perf_counter()
                result = service.enhance(
                    filtered,
                    EnhancementRequest(config.model, SCALE, denoise_strength=config.dni),
                )
                sr_ms = (time.perf_counter() - started) * 1000.0

                tiles = ycrcb_tiles(result.image)
                if chroma == "off":
                    shared_mask = flat_mask(tiles)
                stats = chroma_metrics(tiles, shared_mask)
                luma: ImageMetrics = measure(result.image)
                size = png_bytes(result.image)

                rows.append(
                    {
                        "experiment": "matrix",
                        "config": config.key,
                        "model": config.model,
                        "dni": config.dni,
                        "chroma": chroma,
                        "category": image.category,
                        "image": image.name,
                        "luma": luma.as_dict(),
                        "chroma_metrics": stats.as_dict(),
                        "filter_ms": filter_ms,
                        "sr_ms": sr_ms,
                        "png_bytes": size,
                        "output": f"{result.image.shape[1]}x{result.image.shape[0]}",
                        # Tiling is recorded because x4plus falls back to a
                        # smaller tile under VRAM pressure, which changes its
                        # runtime by ~2x and would otherwise look like an
                        # effect of the arm.
                        "tiling": (
                            {
                                "device": report.device,
                                "tile_size": report.tile_size,
                                "tiles": report.tiles,
                                "reduced": report.tile_size_reduced,
                                "fp16": report.fp16,
                            }
                            if (report := (result.reports[0] if result.reports else None))
                            else None
                        ),
                    }
                )
                del tiles, result

            recent = rows[-len(config.chroma_arms) :]
            print(
                f"    {image.category:18} chroma sigma "
                + " ".join(f"{row['chroma_metrics']['sigma_chroma']:7.4f}" for row in recent)
                + "   luma sigma "
                + " ".join(f"{row['chroma_metrics']['sigma_luma']:6.3f}" for row in recent),
                flush=True,
            )

    freed = service.release()
    print(f"\n  released {freed} model(s)", flush=True)
    return rows


# ------------------------------------------------------------ 2. the probe

SCALING_MEGAPIXELS: tuple[float, ...] = (2.0, 8.0, 16.0)


def run_scaling(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Task: uint8 in, uint8 out, bounded memory, linear time.

    Probes the *input* sizes a prefilter would see, up to `max_input_pixels`
    (16 MP). It deliberately does not probe output sizes: a prefilter never
    sees an output, which is the whole reason the previous full-resolution
    float32 `GaussianBlur` failure cannot recur in this position.
    """
    rows: list[dict[str, Any]] = []
    base = _load(images[0])

    for megapixels in SCALING_MEGAPIXELS:
        target = int(np.sqrt(megapixels * 1_000_000 * base.shape[1] / base.shape[0]))
        reps = int(np.ceil(target / base.shape[1]))
        tiled = np.tile(base, (reps, reps, 1))
        height = int(megapixels * 1_000_000 / target)
        probe = np.ascontiguousarray(tiled[:height, :target])
        actual = probe.shape[0] * probe.shape[1] / 1_000_000
        input_mb = probe.nbytes / 1024 / 1024
        print(
            f"  {actual:5.1f} MP  ({probe.shape[1]}x{probe.shape[0]}, {input_mb:.0f} MB uint8)",
            flush=True,
        )

        for chroma in CHROMA_ARMS:
            if chroma == "off":
                continue
            out, cost = apply_filter(probe, chroma)
            rows.append(
                {
                    "experiment": "scaling",
                    "megapixels": actual,
                    "arm": chroma,
                    "input_mb": input_mb,
                    "dtype_in": str(probe.dtype),
                    "dtype_out": str(out.dtype),
                    "shape_preserved": bool(out.shape == probe.shape),
                    "cost": cost.as_dict(),
                }
            )
            rss = cost.rss_delta_mb
            print(
                f"    {chroma:16} {cost.elapsed_ms:8.1f} ms "
                f"({cost.ms_per_megapixel:5.1f} ms/MP)  "
                f"dtype {probe.dtype}->{out.dtype}  "
                f"rss+{None if rss is None else round(rss, 1)} MB",
                flush=True,
            )
            del out

        del probe, tiled

    return rows


# ------------------------------------------------------------- 3. the crops

#: Regions, in source pixels, chosen by hand per photograph. Each names the
#: failure mode it exists to expose, so a strip that shows nothing is still
#: informative about that mode.
CROP_REGIONS: dict[str, tuple[int, int, str]] = {
    "portrait-skin": (500, 500, "skin gradient and hair - plastic-skin check"),
    "foliage-texture": (450, 700, "frond detail on saturated green - smearing and colour bleed"),
    "text-signage": (560, 250, "lettering against sky - text edges and colour boundary"),
    "low-light-noise": (800, 400, "dark water and sky - chroma mottle, the target case"),
    "landscape-detail": (700, 400, "rock and snow - fine luma texture retention"),
}

CROP_SIZE = 180
CROP_ZOOM = 2

#: The four arms, on Creative's model where all four exist. D is dni 0.00 (the
#: least-denoised state), *not* `None` - see the module docstring.
VISUAL_ARMS: tuple[tuple[str, float, str], ...] = (
    ("off", 0.0, "D baseline"),
    ("off", 0.25, "A DNI"),
    ("chroma-medium", 0.0, "B chroma"),
    ("chroma-medium", 0.25, "C both"),
)


def run_crops(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """D / A / B / C side by side, on SR output, at identical coordinates."""
    from PIL import Image as PILImage

    from app.core.config import Settings
    from app.services.enhancement_service import EnhancementRequest, EnhancementService

    service = EnhancementService(Settings())
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image in images:
        if image.category not in CROP_REGIONS:
            continue

        left, top, why = CROP_REGIONS[image.category]
        source = _load(image)
        panels: list[np.ndarray[Any, Any]] = []

        for chroma, dni, _label in VISUAL_ARMS:
            filtered = source if chroma == "off" else apply_filter(source, chroma)[0]
            result = service.enhance(
                filtered,
                EnhancementRequest("realesr-general-x4v3", SCALE, denoise_strength=dni),
            )
            height, width = result.image.shape[:2]
            span = CROP_SIZE * SCALE
            out_left = min(left * SCALE, max(0, width - span))
            out_top = min(top * SCALE, max(0, height - span))
            panel = result.image[out_top : out_top + span, out_left : out_left + span].copy()
            panel[:, -3:] = 255
            panels.append(panel)
            del result

        strip = np.concatenate(panels, axis=1)
        if CROP_ZOOM > 1:
            strip = np.repeat(np.repeat(strip, CROP_ZOOM, axis=0), CROP_ZOOM, axis=1)
        out_path = CROPS_DIR / f"p3b-{image.category}-DABC.png"
        PILImage.fromarray(strip).save(out_path)

        rows.append(
            {
                "experiment": "crops",
                "category": image.category,
                "file": out_path.name,
                "region_source_px": {"left": left, "top": top, "size": CROP_SIZE},
                "panels": [label for _, _, label in VISUAL_ARMS],
                "model": "realesr-general-x4v3",
                "why": why,
            }
        )
        print(f"  {image.category:18} -> {out_path.name}  ({why})", flush=True)

    service.release()
    return rows


# ------------------------------------------------------------------ metadata


@dataclass(frozen=True, slots=True)
class Provenance:
    generated_at: str
    python: str
    platform: str
    numpy: str
    opencv: str
    torch: str
    cuda_device: str | None
    corpus_root: str
    corpus_images: list[str]
    scale: int
    output_format: str
    configs: list[dict[str, Any]]
    chroma_arms: list[str]
    note_dni_none_equals_one: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def provenance(images: list[CorpusImage]) -> Provenance:
    import cv2
    import torch

    return Provenance(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        opencv=cv2.__version__,
        torch=torch.__version__,
        cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        corpus_root=str(DEFAULT_CORPUS_DIR),
        corpus_images=[f"{i.category}/{i.name}" for i in images],
        scale=SCALE,
        output_format="png (lossless; metrics taken from the array, size from the encode)",
        configs=[asdict(config) for config in CONFIGS],
        chroma_arms=list(CHROMA_ARMS),
        note_dni_none_equals_one=(
            "_resolve_blend returns None for denoise_strength=None and for 1.0 alike, so "
            "'no DNI' and dni=1.00 are the same code path (plain x4v3 weights). The "
            "least-denoised v3 arm is dni=0.00, which is what D means here."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--crops", action="store_true")
    args = parser.parse_args(argv)

    images = load_corpus()
    if not images:
        print("no corpus images; run benchmarks/fetch_corpus.py first", file=sys.stderr)
        return 1
    load_manifest()

    rows: list[dict[str, Any]] = []
    if MEASUREMENTS.is_file():
        rows = list(json.loads(MEASUREMENTS.read_text(encoding="utf-8")).get("rows", []))

    def replace(experiment: str, new: list[dict[str, Any]]) -> None:
        nonlocal rows
        rows = [row for row in rows if row.get("experiment") != experiment] + new

    if args.all or args.matrix:
        print("\n=== 1. interaction matrix ===", flush=True)
        replace("matrix", run_matrix(images))
    if args.all or args.scaling:
        print("\n=== 2. scaling / dtype / memory ===", flush=True)
        replace("scaling", run_scaling(images))
    if args.all or args.crops:
        print("\n=== 3. visual crops ===", flush=True)
        replace("crops", run_crops(images))

    if not rows:
        parser.print_help()
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(
        json.dumps({"provenance": provenance(images).as_dict(), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {MEASUREMENTS} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
