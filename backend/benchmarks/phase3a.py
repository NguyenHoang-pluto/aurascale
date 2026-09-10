"""Phase 3A runner: execute the pre-denoise experiments and record what they say.

    python -m benchmarks.phase3a --all       # everything below
    python -m benchmarks.phase3a --filters   # 1. candidates on the real corpus
    python -m benchmarks.phase3a --ladder    # 2. estimator against known noise
    python -m benchmarks.phase3a --sr        # 3. through the real SR pipeline
    python -m benchmarks.phase3a --scaling   # 4. runtime and memory versus size
    python -m benchmarks.phase3a --crops     # 5. visual crops for inspection

Writes `results/phase3a_measurements.json`. Nothing here changes production:
the SR experiment calls `EnhancementService` exactly as a job would, with
settings passed per run, and the prefilter is applied to the numpy array in
this process before handing it over.

Four experiments rather than one because they answer different questions and
have very different costs. The filter sweep is CPU-only and cheap enough to run
every arm; the SR sweep costs a GPU pass per cell and is restricted to the arms
the sweep left standing.
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
from benchmarks.metrics import ImageMetrics, measure
from benchmarks.prefilter import (
    CANDIDATES,
    SIGMA_CLEAN,
    SIGMA_NOISY,
    adaptive_strength,
    apply_filter,
    calibrate_anchors,
    estimate_noise,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase3a_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase3a-crops"

#: The injected per-channel sigmas for the controlled ladder. Chosen to bracket
#: what real photographs show: 2 is a clean modern sensor, 8 is a visibly noisy
#: high-ISO frame, 16 is worse than anything a user is likely to upload.
INJECTED_SIGMAS: tuple[int, ...] = (0, 2, 4, 8, 16)

#: Which arms go through the full SR pipeline. Restricted deliberately - each
#: cell is a GPU pass, and running all fifteen arms would spend an hour to
#: re-derive what the CPU sweep already showed about the losers.
SR_ARMS: tuple[str, ...] = (
    "none",
    "gaussian-medium",
    "bilateral-medium",
    "chroma-medium",
    "nlm-medium",
)

#: The seed for every noise injection, so the ladder is reproducible.
NOISE_SEED = 20260909


def _load(image: CorpusImage) -> np.ndarray[Any, Any]:
    from PIL import Image

    return np.asarray(Image.open(image.path).convert("RGB"), dtype=np.uint8)


def _inject(base: np.ndarray[Any, Any], sigma: int, seed: int = NOISE_SEED) -> np.ndarray[Any, Any]:
    """Add independent gaussian noise per channel at a known sigma.

    Synthetic, and used only where a *known* answer is the point: validating
    that the estimator tracks noise it can be checked against. No conclusion
    about production quality rests on these images - Task 4 is explicit that
    those come from the photographs, and they do.
    """
    if sigma == 0:
        return base

    generator = np.random.default_rng(seed)
    noisy = base.astype(np.float32) + generator.normal(0.0, sigma, base.shape)
    clipped: np.ndarray[Any, Any] = np.clip(noisy, 0, 255).astype(np.uint8)
    return clipped


def _metric_dict(metrics: ImageMetrics) -> dict[str, Any]:
    return metrics.as_dict()


# ------------------------------------------------------- 1. the filter sweep


def run_filters(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Every candidate over every real photograph, measured directly.

    Directly, meaning on the filtered input rather than on an SR result. This
    isolates what the filter itself does to noise, texture and edges, which is
    the question Tasks 3 and 5 ask. What the model then does with a filtered
    input is a different question and is experiment 3.
    """
    rows: list[dict[str, Any]] = []

    for image in images:
        source = _load(image)
        estimate = estimate_noise(source)
        print(
            f"  {image.category:18} {source.shape[1]}x{source.shape[0]}  "
            f"sigma_luma={estimate.sigma_luma:.3f} score={estimate.score:.3f}"
        )

        for spec, _ in CANDIDATES:
            filtered, cost = apply_filter(source, spec.arm)
            metrics = measure(filtered)
            after = estimate_noise(filtered)

            rows.append(
                {
                    "experiment": "filters",
                    "category": image.category,
                    "image": image.name,
                    "arm": spec.arm,
                    "family": spec.family,
                    "strength": spec.strength,
                    "params": spec.params,
                    "metrics": _metric_dict(metrics),
                    "cost": cost.as_dict(),
                    "noise_before": estimate.as_dict(),
                    "noise_after": after.as_dict(),
                }
            )
            print(
                f"    {spec.arm:20} {cost.elapsed_ms:8.1f} ms  "
                f"({cost.ms_per_megapixel:6.1f} ms/MP)  "
                f"flat_noise={metrics.flat_noise:.5f} hf={metrics.high_frequency_ratio:.5f}"
            )

    return rows


# --------------------------------------------------- 2. the controlled ladder


def run_ladder(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Does the estimator track noise it can be checked against?

    The real corpus cannot answer this: nothing in it has a known noise level,
    and its five images span only 2.4x. Injecting a known sigma into those same
    photographs keeps the *content* real while making the noise measurable, so
    a failure to track shows up as a failure rather than as a plausible number.
    """
    rows: list[dict[str, Any]] = []

    for image in images:
        base = _load(image)

        for sigma in INJECTED_SIGMAS:
            noisy = _inject(base, sigma)
            estimate = estimate_noise(noisy)
            metrics = measure(noisy)

            rows.append(
                {
                    "experiment": "ladder",
                    "category": image.category,
                    "image": image.name,
                    "injected_sigma": sigma,
                    "noise": estimate.as_dict(),
                    "metrics": _metric_dict(metrics),
                    "proposed_strength": adaptive_strength(estimate.score),
                }
            )

        recovered = [row["noise"]["sigma_luma"] for row in rows[-len(INJECTED_SIGMAS) :]]
        print(
            f"  {image.category:18} recovered sigma_luma: "
            + " ".join(f"{v:6.2f}" for v in recovered)
        )

    return rows


# ------------------------------------------------------ 3. through the model


def run_sr(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """The arms that survived, through the real 4x pipeline.

    This is the experiment that matters for a production decision. A prefilter
    is not judged by what it does to the input - the user never sees the input
    - but by what the model then produces from it, and Real-ESRGAN was trained
    on degraded inputs, so pre-smoothing is not guaranteed to help just because
    the intermediate looks cleaner.
    """
    from app.core.config import Settings
    from app.services.enhancement_service import EnhancementRequest, EnhancementService

    settings = Settings()
    service = EnhancementService(settings)
    rows: list[dict[str, Any]] = []

    for image in images:
        source = _load(image)
        print(f"  {image.category:18} {source.shape[1]}x{source.shape[0]}")

        for arm in SR_ARMS:
            filtered, filter_cost = apply_filter(source, arm)

            started = time.perf_counter()
            result = service.enhance(
                filtered,
                # Denoise left off so the prefilter is the only variable. The
                # DNI arm is measured separately below, against the same
                # baseline, so the two denoising strategies are comparable.
                EnhancementRequest("realesr-general-x4v3", 4, denoise_strength=0.0),
            )
            sr_ms = (time.perf_counter() - started) * 1000.0
            metrics = measure(result.image)

            rows.append(
                {
                    "experiment": "sr",
                    "category": image.category,
                    "image": image.name,
                    "arm": arm,
                    "denoise_dni": 0.0,
                    "metrics": _metric_dict(metrics),
                    "filter_ms": filter_cost.elapsed_ms,
                    "sr_ms": sr_ms,
                    "output": f"{result.image.shape[1]}x{result.image.shape[0]}",
                }
            )
            print(
                f"    {arm:20} filter {filter_cost.elapsed_ms:7.1f} ms  "
                f"sr {sr_ms:8.1f} ms  flat_noise={metrics.flat_noise:.5f} "
                f"hf={metrics.high_frequency_ratio:.5f}"
            )

        # The incumbent, for a like-for-like comparison: no prefilter, denoise
        # done by the model's own DNI blend at the value Creative ships.
        for dni in (0.25, 1.0):
            started = time.perf_counter()
            result = service.enhance(
                source, EnhancementRequest("realesr-general-x4v3", 4, denoise_strength=dni)
            )
            sr_ms = (time.perf_counter() - started) * 1000.0
            metrics = measure(result.image)

            rows.append(
                {
                    "experiment": "sr",
                    "category": image.category,
                    "image": image.name,
                    "arm": f"dni-{dni:.2f}",
                    "denoise_dni": dni,
                    "metrics": _metric_dict(metrics),
                    "filter_ms": 0.0,
                    "sr_ms": sr_ms,
                    "output": f"{result.image.shape[1]}x{result.image.shape[0]}",
                }
            )
            print(
                f"    {f'dni-{dni:.2f}':20} filter     0.0 ms  sr {sr_ms:8.1f} ms  "
                f"flat_noise={metrics.flat_noise:.5f} hf={metrics.high_frequency_ratio:.5f}"
            )

    service.release()
    return rows


# ------------------------------------------------------- 4. scaling and memory


#: Sizes the scaling probe runs at. The last is `max_input_pixels`, which is
#: the worst case a prefilter would ever face - the point of the probe is to
#: know what happens there before anything is built, not after.
SCALING_MEGAPIXELS: tuple[float, ...] = (2.0, 8.0, 16.0)


def run_scaling(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Runtime and memory against input size, up to the 16 MP input ceiling.

    Task 9. The previous out-of-memory failure in this project came from a
    full-resolution float32 buffer on the *output* side, so what matters here
    is whether any candidate allocates on that scale on the input side, and
    whether the worst case is affordable at all.
    """
    from PIL import Image as PILImage

    rows: list[dict[str, Any]] = []
    # One image, tiled up to size rather than upscaled: resampling would smooth
    # the noise away and make the filters look cheaper than they are.
    base = _load(images[0])

    for megapixels in SCALING_MEGAPIXELS:
        target = int(np.sqrt(megapixels * 1_000_000 * base.shape[1] / base.shape[0]))
        reps = int(np.ceil(target / base.shape[1]))
        tiled = np.tile(base, (reps, reps, 1))
        height = int(megapixels * 1_000_000 / target)
        probe = np.ascontiguousarray(tiled[:height, :target])
        actual = probe.shape[0] * probe.shape[1] / 1_000_000
        print(f"  {actual:5.1f} MP  ({probe.shape[1]}x{probe.shape[0]})")

        for spec, _ in CANDIDATES:
            if spec.arm == "none":
                continue
            try:
                _, cost = apply_filter(probe, spec.arm)
            except Exception as exc:  # a probe reports what failed; it never fails itself
                rows.append(
                    {
                        "experiment": "scaling",
                        "megapixels": actual,
                        "arm": spec.arm,
                        "failed": type(exc).__name__,
                        "message": str(exc)[:200],
                    }
                )
                print(f"    {spec.arm:20} FAILED {type(exc).__name__}: {str(exc)[:60]}")
                continue

            rows.append(
                {
                    "experiment": "scaling",
                    "megapixels": actual,
                    "arm": spec.arm,
                    "family": spec.family,
                    "cost": cost.as_dict(),
                }
            )
            rss = cost.rss_delta_mb
            print(
                f"    {spec.arm:20} {cost.elapsed_ms:9.1f} ms  "
                f"({cost.ms_per_megapixel:6.1f} ms/MP)  "
                f"rss+{rss if rss is None else round(rss, 1)} MB"
            )

        del probe, tiled
        _ = PILImage  # keep the import meaningful for the type checker

    return rows


# ------------------------------------------------------------- 5. visual crops


#: Where to crop from, per category, in source pixel coordinates. Chosen by
#: hand after looking at each photograph, because Phase 2.5's automatic
#: highest-variance selection landed on dark foliage instead of the lettering
#: it was meant to inspect - the one visual gap that phase reported.
CROP_REGIONS: dict[str, tuple[int, int, str]] = {
    "landscape-detail": (700, 400, "rock and snow texture"),
    "portrait-skin": (500, 500, "skin gradient and hair"),
    "text-signage": (560, 250, "lettering - the region Phase 2.5 missed"),
    "foliage-texture": (450, 700, "frond detail"),
    "low-light-noise": (800, 400, "dark water and sky, where chroma mottle lives"),
}

CROP_SIZE = 256
CROP_ZOOM = 3


def run_crops(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """Side-by-side crops of every arm, for the eye rather than the metrics.

    Task 6 is explicit that metrics alone are not sufficient, and the failure
    modes that matter most here - plastic skin, erased micro-texture, haloing -
    are exactly the ones a flat-region statistic can miss.
    """
    from PIL import Image as PILImage

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image in images:
        if image.category not in CROP_REGIONS:
            continue

        left, top, why = CROP_REGIONS[image.category]
        source = _load(image)
        height, width = source.shape[:2]
        left = min(left, max(0, width - CROP_SIZE))
        top = min(top, max(0, height - CROP_SIZE))

        panels: list[np.ndarray[Any, Any]] = []
        labels: list[str] = []
        for spec, _ in CANDIDATES:
            filtered, _cost = apply_filter(source, spec.arm)
            panels.append(filtered[top : top + CROP_SIZE, left : left + CROP_SIZE])
            labels.append(spec.arm)

        strip = np.concatenate(panels, axis=1)
        zoomed = np.repeat(np.repeat(strip, CROP_ZOOM, axis=0), CROP_ZOOM, axis=1)
        out = CROPS_DIR / f"{image.category}-strip.png"
        PILImage.fromarray(zoomed).save(out)

        rows.append(
            {
                "experiment": "crops",
                "category": image.category,
                "file": out.name,
                "region": {"left": left, "top": top, "size": CROP_SIZE, "why": why},
                "panels": labels,
                "zoom": CROP_ZOOM,
            }
        )
        print(f"  {image.category:18} -> {out.name}  ({len(labels)} panels, {why})")

    return rows


# ------------------------------------------------------------------ metadata


@dataclass(frozen=True, slots=True)
class Provenance:
    """Exactly what produced these numbers. Task 12's reproducibility block."""

    generated_at: str
    python: str
    platform: str
    numpy: str
    opencv: str
    torch: str
    cuda_device: str | None
    corpus_root: str
    corpus_images: list[str]
    sigma_clean: float
    sigma_noisy: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def provenance(images: list[CorpusImage]) -> Provenance:
    import cv2
    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    return Provenance(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        opencv=cv2.__version__,
        torch=torch.__version__,
        cuda_device=device,
        corpus_root=str(DEFAULT_CORPUS_DIR),
        corpus_images=[f"{i.category}/{i.name}" for i in images],
        sigma_clean=SIGMA_CLEAN,
        sigma_noisy=SIGMA_NOISY,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--filters", action="store_true")
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--sr", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--crops", action="store_true")
    args = parser.parse_args(argv)

    images = load_corpus()
    if not images:
        print("no corpus images; run benchmarks/fetch_corpus.py first", file=sys.stderr)
        return 1

    # Validated, so a run cannot silently measure a corpus other than the one
    # it reports on.
    load_manifest()

    rows: list[dict[str, Any]] = []
    existing: dict[str, Any] = {}
    if MEASUREMENTS.is_file():
        existing = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
        rows = list(existing.get("rows", []))

    def replace(experiment: str, new: list[dict[str, Any]]) -> None:
        nonlocal rows
        rows = [row for row in rows if row.get("experiment") != experiment] + new

    if args.all or args.filters:
        print("\n=== 1. filter sweep (real corpus) ===")
        replace("filters", run_filters(images))
    if args.all or args.ladder:
        print("\n=== 2. controlled noise ladder ===")
        replace("ladder", run_ladder(images))
    if args.all or args.sr:
        print("\n=== 3. through the SR pipeline ===")
        replace("sr", run_sr(images))
    if args.all or args.scaling:
        print("\n=== 4. scaling and memory ===")
        replace("scaling", run_scaling(images))
    if args.all or args.crops:
        print("\n=== 5. visual crops ===")
        replace("crops", run_crops(images))

    if not rows:
        parser.print_help()
        return 1

    # Anchors, recomputed from whatever the corpus actually measured, so the
    # report can state what the corpus supports rather than what was assumed.
    sweep = [row for row in rows if row.get("experiment") == "filters" and row["arm"] == "none"]
    anchors: dict[str, Any] = {}
    if sweep:
        combined = [
            row["noise_before"]["sigma_luma"] + 0.35 * row["noise_before"]["sigma_chroma"]
            for row in sweep
        ]
        low, high = calibrate_anchors(combined)
        anchors = {
            "corpus_combined_sigmas": [round(value, 4) for value in combined],
            "percentile_20": round(low, 4),
            "percentile_80": round(high, 4),
            "in_use_clean": SIGMA_CLEAN,
            "in_use_noisy": SIGMA_NOISY,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(
        json.dumps(
            {
                "provenance": provenance(images).as_dict(),
                "anchors": anchors,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {MEASUREMENTS} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
