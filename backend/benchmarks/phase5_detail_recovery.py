"""Phase 5 - can detail be *recovered* after Real-ESRGAN, or only amplified?

Research only. Nothing here changes production. Every candidate is a
post-process applied to the output of the production upscaler; the upscaler,
its defaults, the model routing and the sharpening default are untouched.

The question
------------
F3 measured the sharpening the codebase already owns and established the lesson
this phase is built on: **high-frequency energy can rise while the picture gets
worse.** Sobel, HF ratio and local contrast all increase under an unsharp mask
whether the thing being amplified is texture, sensor noise or a JPEG block edge.
So "the metric went up" is not evidence, and this phase refuses to treat it as
evidence.

What would count as evidence is a candidate that raises detail *where detail
exists* and leaves noise alone. That is a claim about selectivity, not about
gain, and it needs two things F3 did not have:

  1. **A reference.** §`degradation` builds one. A corpus photograph is degraded
     by a fixed, pre-committed pipeline, upscaled back, and compared against the
     original it came from. With ground truth in hand, "recovered real detail"
     and "invented plausible detail" stop being the same measurement.

  2. **Proof the operators differ.** §`operators` applies every candidate to
     synthetic signals whose answer is known - a step edge, pure noise, a
     frequency sweep - and reports overshoot, noise gain and band response. An
     operator that matches the unsharp mask on all three is a duplicate
     experiment and is reported as one rather than dressed up as a new idea.

The candidates
--------------
`A` baseline, `B` the production unsharp mask at 0.25 - F3's recommended value,
included as the incumbent to beat rather than as a new proposal - and three
post-processes that are *not* unsharp masks:

  `C` guided-filter detail boost. The base layer comes from an edge-preserving
      filter rather than a Gaussian, so a strong edge stays in the base and the
      detail layer is near zero there. Halo and ringing are suppressed by
      construction, not by a dead zone.

  `D` structure-gated detail boost. The same detail layer as `C`, but the gain
      is multiplied by structure-tensor coherence. Oriented structure - hair,
      veins in a leaf, a rock seam - is boosted; isotropic signal, which is what
      sensor noise looks like, is not. This is the arm that tests selectivity
      directly.

  `E` band-limited detail boost. A difference-of-Gaussians band, so the very
      highest frequencies - where noise and JPEG ringing live - are left alone
      while mid-scale texture is raised.

`B` is linear-plus-pointwise with a spatially constant gain. `C` is signal
dependent through the guided filter's variance term, `D` is spatially varying
through the coherence gate, and `E` occupies a different band. None of the three
reduces to `B`, and §`operators` is where that is demonstrated rather than
asserted.

Reuses `corpus`, `metrics`, `research_utils` and the SSIM already written for
F4. No new dependency: the guided filter is four box filters, written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.corpus import DEFAULT_CORPUS_DIR, CorpusImage, load_corpus
from benchmarks.metrics import ImageMetrics, measure, to_luma
from benchmarks.phase4_f4 import compare, digest, ssim_sampled
from benchmarks.research_utils import CROP_REGIONS, blocks_of, immerkaer_sigma, load_rgb

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEASUREMENTS = RESULTS_DIR / "phase5_detail_recovery_measurements.json"
CROPS_DIR = RESULTS_DIR / "phase5-detail-crops"

MODEL = "RealESRGAN_x4plus"
SCALE = 4

#: Luma is on the 8-bit scale throughout this module. `metrics.to_luma` returns
#: [0, 1]; every threshold below (dead zones, eps, gate floors) is quoted in
#: levels out of 255, so the conversion happens once, here.
LUMA_FULL_SCALE = 255.0

# --------------------------------------------------------------- the arms

#: F3's recommended value. Not a new proposal - the incumbent, carried in so the
#: new candidates are measured against the best thing the codebase already has
#: rather than against nothing.
SHARPEN_ARM_STRENGTH = 0.25

#: Every candidate gain below was calibrated so that all four arms deliver the
#: **same mid-band gain, 1.191** - the gain the production unsharp mask produces
#: at 0.25. Equalising the benefit is what makes the comparison mean anything:
#: an arm that boosts less will always look safer, and "safer because it does
#: less" is not a finding. With benefit held equal, the only thing left to
#: differ is cost, which is what the report compares.
#:
#: The calibration was solved on the synthetic signals in `characterise` alone,
#: by bisection, **before any photograph was processed**. No parameter here was
#: chosen or revised after seeing a picture result.
CALIBRATION_TARGET_MID_GAIN = 1.191

#: Guided filter. `radius` is in output pixels at 4x; `eps` is a variance
#: threshold in squared luma levels: content whose local variance is far above
#: it stays in the base layer, content far below it lands in the detail layer.
#: 200 is a ~14-level standard deviation, which sits above textural variation
#: and below a hard edge - so edges are preserved and texture is available to
#: boost. A much smaller eps was tried first and is the reason this constant is
#: documented: at eps = 4 the filter treats ordinary texture as an edge, the
#: detail layer holds almost nothing but noise, and the arm is a near no-op.
GUIDED_RADIUS = 4
GUIDED_EPS = 200.0
GUIDED_GAIN = 0.396

#: Structure tensor. The gradient is smoothed over `TENSOR_SIGMA` before the
#: tensor is formed, which is what makes coherence a neighbourhood property
#: rather than a per-pixel one.
TENSOR_SIGMA = 2.0
#: Coherence below this is treated as unoriented - noise-like - and gets no
#: boost at all. Above `GATE_HIGH` the gate is fully open. Between them it ramps
#: linearly. Values are on the coherence scale, which is [0, 1] by construction.
GATE_LOW = 0.15
GATE_HIGH = 0.55
GATED_GAIN = 0.396

#: Difference-of-Gaussians band. `DOG_INNER` is the short end - frequencies
#: finer than this are *excluded* from the boost, which is the whole point of
#: the arm - and `DOG_OUTER` the long end.
DOG_INNER = 1.2
DOG_OUTER = 3.2
DOG_GAIN = 0.342

#: Ceiling applied to every candidate's correction, in luma levels, so no arm
#: can produce an unbounded excursion at a pathological pixel. Matches the
#: production sharpener's ceiling so the arms are limited alike.
CORRECTION_CEILING = 10.0

# ------------------------------------------------------- the degradation

#: Fixed before any candidate was run and not touched since. Tuning these after
#: seeing results would be choosing the answer, so they are constants and the
#: report says so.
DEGRADE_BLUR_SIGMA = 0.8
DEGRADE_NOISE_SIGMA = 2.0
DEGRADE_JPEG_QUALITY = 85
DEGRADE_SEED = 20260909

#: The reference track needs the *original* photograph as ground truth, so the
#: degraded input is built at 1/4 size and the 4x pass lands back on the
#: original grid. Any residual off-by-one from odd dimensions is cropped, not
#: resized, so no arm is compared through an extra resampling step.

# ------------------------------------------------------------------ crops

DETAIL_QUALITY = 95
CROP_SPAN = 384

#: Regions inspected per photograph. Chosen from the tracked crop table to cover
#: what the phase is required to judge - skin, hair, eye, text, foliage, dark
#: noise, fine texture - without carpeting the frame.
INSPECT: dict[str, tuple[str, ...]] = {
    "portrait-skin": ("skin-cheek", "hair", "eye-left"),
    "landscape-detail": ("fine-texture",),
    "foliage-texture": ("saturated-foliage",),
    "text-signage": ("text",),
    "low-light-noise": ("dark-noisy",),
}


def provenance() -> dict[str, Any]:
    import torch

    from app.core.config import Settings
    from benchmarks.corpus import load_manifest

    settings = Settings()
    entries = load_manifest()
    return {
        "phase": "5",
        "generated": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": MODEL,
        "scale": SCALE,
        "production_tile_size": settings.tile_size,
        "production_tile_pad": settings.tile_pad,
        "production_sharpen_default": 0.0,
        "arms": [name for name, _ in ARMS],
        "sharpen_arm_strength": SHARPEN_ARM_STRENGTH,
        "guided": {"radius": GUIDED_RADIUS, "eps": GUIDED_EPS, "gain": GUIDED_GAIN},
        "gated": {
            "tensor_sigma": TENSOR_SIGMA,
            "gate_low": GATE_LOW,
            "gate_high": GATE_HIGH,
            "gain": GATED_GAIN,
        },
        "dog": {"inner": DOG_INNER, "outer": DOG_OUTER, "gain": DOG_GAIN},
        "calibration_target_mid_gain": CALIBRATION_TARGET_MID_GAIN,
        "calibrated_on": "synthetic signals only, before any photograph",
        "correction_ceiling": CORRECTION_CEILING,
        "degradation": {
            "blur_sigma": DEGRADE_BLUR_SIGMA,
            "downsample": SCALE,
            "noise_sigma": DEGRADE_NOISE_SIGMA,
            "jpeg_quality": DEGRADE_JPEG_QUALITY,
            "seed": DEGRADE_SEED,
            "order": "blur -> downsample -> noise -> jpeg",
            "fixed_before_candidates": True,
        },
        "corpus": [
            {
                "category": entry.category,
                "filename": entry.filename,
                "sha256": hashlib.sha256(
                    (DEFAULT_CORPUS_DIR / entry.filename).read_bytes()
                ).hexdigest(),
            }
            for entry in entries
        ],
    }


# ------------------------------------------------------------- primitives


def _box(channel: np.ndarray[Any, Any], radius: int) -> np.ndarray[Any, Any]:
    import cv2

    size = 2 * radius + 1
    return cv2.boxFilter(channel, -1, (size, size), borderType=cv2.BORDER_REFLECT)


def guided_self(channel: np.ndarray[Any, Any], radius: int, eps: float) -> np.ndarray[Any, Any]:
    """He et al.'s guided filter, self-guided, as the edge-preserving base.

    Written here rather than imported: `cv2.ximgproc` is not installed and
    adding `opencv-contrib` for four box filters would be a dependency the
    research does not need.

    The `eps` term is what makes this different in kind from a Gaussian. Where
    local variance is large - an edge - `a` approaches 1 and the filter returns
    the input, so the edge stays in the base and the detail layer is near zero
    there. Where variance is small - flat or lightly textured - `a` approaches 0
    and the filter returns the local mean, so the texture lands in the detail
    layer and can be boosted.
    """
    mean_i = _box(channel, radius)
    corr_i = _box(channel * channel, radius)
    var_i = corr_i - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    filtered: np.ndarray[Any, Any] = _box(a, radius) * channel + _box(b, radius)
    return filtered


def coherence(channel: np.ndarray[Any, Any], sigma: float = TENSOR_SIGMA) -> np.ndarray[Any, Any]:
    """Structure-tensor coherence in [0, 1]: 1 is oriented, 0 is isotropic.

    This is the arm-D gate and the one measurement in this phase that separates
    "there is signal here" from "there is *structure* here". Sensor noise has
    energy at every orientation, so its tensor is close to isotropic and its
    coherence is near zero; an edge, a hair or a leaf vein has energy along one
    axis and scores high. Gradient products are smoothed before the ratio is
    taken, which is what makes this a neighbourhood property - a per-pixel
    tensor is rank 1 everywhere and its coherence is identically 1, which would
    gate nothing.
    """
    import cv2

    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    kernel = (0, 0)
    jxx = cv2.GaussianBlur(gx * gx, kernel, sigma)
    jyy = cv2.GaussianBlur(gy * gy, kernel, sigma)
    jxy = cv2.GaussianBlur(gx * gy, kernel, sigma)

    trace = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    return np.asarray(disc / np.maximum(trace, 1e-6), dtype=np.float32)


def _apply_luma_correction(
    rgb: np.ndarray[Any, Any], correction: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Add a luma-domain correction equally to R, G and B.

    The same rule the production sharpener uses, and for the same reason: moving
    a pixel along the neutral axis preserves hue and saturation, so no candidate
    can win by introducing a coloured fringe that a luma metric would not see.
    """
    np.clip(correction, -CORRECTION_CEILING, CORRECTION_CEILING, out=correction)
    out = rgb.astype(np.float32)
    out += correction[:, :, None]
    clipped: np.ndarray[Any, Any] = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return clipped


# ------------------------------------------------------------------ arms


def arm_baseline(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A - the neural output, untouched. Returned as-is, not copied."""
    return rgb


def arm_sharpen(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """B - the production unsharp mask at F3's recommended 0.25.

    Calls production's own `unsharp_mask`. Nothing is reimplemented, so this arm
    is exactly what the product would do if sharpening were switched on, and any
    difference between it and C/D/E is a difference between operators rather
    than between two spellings of one.
    """
    from app.services.enhancement_service import unsharp_mask

    return unsharp_mask(rgb, SHARPEN_ARM_STRENGTH, scale=SCALE)


def arm_guided(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """C - boost the detail layer of an edge-preserving decomposition."""
    luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
    base = guided_self(luma, GUIDED_RADIUS, GUIDED_EPS)
    return _apply_luma_correction(rgb, GUIDED_GAIN * (luma - base))


def arm_gated(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """D - the same detail layer, gated by structure-tensor coherence."""
    luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
    base = guided_self(luma, GUIDED_RADIUS, GUIDED_EPS)
    gate = np.clip((coherence(luma) - GATE_LOW) / (GATE_HIGH - GATE_LOW), 0.0, 1.0)
    return _apply_luma_correction(rgb, GATED_GAIN * gate * (luma - base))


def arm_dog(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """E - boost a mid-frequency band, leaving the finest scale alone."""
    import cv2

    luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
    inner = cv2.GaussianBlur(luma, (0, 0), DOG_INNER)
    outer = cv2.GaussianBlur(luma, (0, 0), DOG_OUTER)
    return _apply_luma_correction(rgb, DOG_GAIN * (inner - outer))


ARMS: tuple[tuple[str, Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]], ...] = (
    ("A-baseline", arm_baseline),
    ("B-sharpen025", arm_sharpen),
    ("C-guided", arm_guided),
    ("D-gated", arm_gated),
    ("E-dog", arm_dog),
)
REFERENCE_ARM = ARMS[0][0]


# ------------------------------------------------- operator characterisation


@dataclass(frozen=True, slots=True)
class OperatorSignature:
    """What an operator does to signals whose correct answer is known."""

    #: Peak overshoot beyond the step height on an ideal edge, in luma levels.
    #: This is ringing, measured directly rather than inferred.
    step_overshoot: float
    #: Output noise sigma divided by input noise sigma on pure Gaussian noise.
    #: 1.0 is transparent; above 1.0 the operator amplifies noise.
    noise_gain: float
    #: Gain on a mid-scale sinusoid and on a fine-scale one. An operator that
    #: raises mid and leaves fine alone is doing something an unsharp mask does
    #: not.
    gain_mid: float
    gain_fine: float
    #: Gain on a flat field. Must be 1.0 for every arm; a value away from it
    #: would mean the operator shifts overall level, which none should.
    gain_flat: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_rgb(luma: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.repeat(np.clip(luma, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _luma_of(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return to_luma(rgb).astype(np.float64) * LUMA_FULL_SCALE


def characterise(operator: Callable[..., Any]) -> OperatorSignature:
    """Apply one arm to synthetic signals and read off what it did.

    A grey background of 128 is used everywhere so no test clips, and every
    signal is large enough that the operators' kernels are never reaching past
    an edge into border handling.
    """
    size = 256

    # --- step edge: overshoot beyond the step is ringing
    step = np.full((size, size), 96.0)
    step[:, size // 2 :] = 160.0
    out = _luma_of(operator(_as_rgb(step)))
    band = out[:, size // 2 - 24 : size // 2 + 24]
    overshoot = float(max(band.max() - 160.0, 96.0 - band.min(), 0.0))

    # --- pure noise: how much of it survives, relative to input
    generator = np.random.default_rng(DEGRADE_SEED)
    noise = 128.0 + generator.normal(0.0, 6.0, (size, size))
    noise_in = _luma_of(_as_rgb(noise))
    noise_out = _luma_of(operator(_as_rgb(noise)))
    gain_noise = float(np.std(noise_out) / max(np.std(noise_in), 1e-9))

    # --- sinusoids: gain per band, measured as amplitude ratio
    def sine_gain(period: float) -> float:
        x = np.arange(size, dtype=np.float64)
        wave = 128.0 + 20.0 * np.sin(2.0 * np.pi * x / period)
        field = np.tile(wave, (size, 1))
        before = _luma_of(_as_rgb(field))
        after = _luma_of(operator(_as_rgb(field)))
        centre = slice(32, size - 32)
        amp_in = float(before[centre, centre].std())
        amp_out = float(after[centre, centre].std())
        return amp_out / max(amp_in, 1e-9)

    flat = np.full((size, size), 128.0)
    flat_gain = float(_luma_of(operator(_as_rgb(flat))).mean() / 128.0)

    return OperatorSignature(
        step_overshoot=overshoot,
        noise_gain=gain_noise,
        gain_mid=sine_gain(12.0),
        gain_fine=sine_gain(4.0),
        gain_flat=flat_gain,
    )


def run_operators() -> list[dict[str, Any]]:
    """Are C, D and E actually different operators from B?

    Runs first and is the gate on the rest: an arm whose signature matches the
    unsharp mask is a duplicate experiment, and the report has to say so rather
    than present it as a new idea.
    """
    print("\n  operator characterisation on synthetic signals", flush=True)
    rows: list[dict[str, Any]] = []
    for name, operator in ARMS:
        signature = characterise(operator)
        rows.append({"experiment": "operators", "arm": name, "signature": signature.as_dict()})
        print(
            f"    {name:14s} overshoot={signature.step_overshoot:6.2f}  "
            f"noise_gain={signature.noise_gain:5.3f}  "
            f"mid={signature.gain_mid:5.3f}  fine={signature.gain_fine:5.3f}  "
            f"flat={signature.gain_flat:6.4f}",
            flush=True,
        )
    return rows


# ------------------------------------------------------------ the SR pass


@dataclass(frozen=True, slots=True)
class PassRecord:
    requested_tile: int
    effective_tile: int
    tiles: int
    tile_reduced: bool
    fell_back_to_cpu: bool
    device: str
    seconds: float
    peak_vram_mb: float
    free_vram_before_mb: int | None
    honoured: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_manager() -> Any:
    from app.core.config import Settings
    from app.inference.model_manager import ModelManager
    from app.services.model_service import ModelService

    settings = Settings()
    return ModelManager(settings, ModelService(settings))


def run_pass(manager: Any, source: np.ndarray[Any, Any]) -> tuple[Any, PassRecord]:
    """One production upscale at the production tile, recorded as F4 records it.

    The neural pass is the expensive part and it is identical for every arm -
    the candidates are post-processes - so it runs once per photograph and its
    output is reused. That is not an optimisation detail: it also means no arm
    can differ from another by model variance, tiling or precision.
    """
    import torch

    from app.core.config import Settings
    from app.inference.device import free_vram_mb, release_cuda_memory

    settings = Settings()
    upscaler = manager.get(MODEL, denoise_strength=None)

    release_cuda_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    free_before = free_vram_mb(upscaler.target.index) if torch.cuda.is_available() else None

    started = time.perf_counter()
    output = upscaler.upscale(source, tile=settings.tile_size, tile_pad=settings.tile_pad)
    seconds = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    report = upscaler.last_report
    assert report is not None
    record = PassRecord(
        requested_tile=settings.tile_size,
        effective_tile=report.tile_size,
        tiles=report.tiles,
        tile_reduced=report.tile_size_reduced,
        fell_back_to_cpu=report.fell_back_to_cpu,
        device=report.device,
        seconds=seconds,
        peak_vram_mb=peak,
        free_vram_before_mb=free_before,
        honoured=report.tile_size == settings.tile_size and not report.fell_back_to_cpu,
    )
    return output, record


def _metrics_dict(metrics: ImageMetrics) -> dict[str, Any]:
    return asdict(metrics)


def dark_region_hf(rgb: np.ndarray[Any, Any]) -> float:
    """High-frequency energy in the darkest fifth of the frame.

    F3 introduced this because grain amplification shows up in shadow long
    before it shows anywhere else, and a whole-frame HF number averages it away.
    Reported as a raw quantity; it is only meaningful as a ratio between arms.
    """
    import cv2

    luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
    threshold = float(np.percentile(luma, 20.0))
    mask = luma <= threshold
    if not mask.any():
        return 0.0
    detail = luma - cv2.GaussianBlur(luma, (0, 0), 1.0)
    return float(np.abs(detail[mask]).mean())


def noise_sigma(rgb: np.ndarray[Any, Any]) -> float:
    """Immerkaer sigma on luma, via the tracked estimator."""
    luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
    return immerkaer_sigma(blocks_of(luma))


# --------------------------------------------------------- the real track


def run_real(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """5 photographs x 5 arms on real 4x output. No reference exists here.

    This track cannot say whether detail is real - nothing to compare against -
    so it reports no-reference metrics and writes the crops the eye will judge.
    The reference question is `--degradation`'s job.
    """
    manager = _make_manager()
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    try:
        for image in images:
            source = load_rgb(image)
            print(
                f"\n  {image.category}  {source.shape[1]}x{source.shape[0]}"
                f" -> {source.shape[1] * SCALE}x{source.shape[0] * SCALE}",
                flush=True,
            )
            neural, record = run_pass(manager, source)
            print(
                f"    neural pass {record.seconds:6.1f}s  tiles={record.tiles}  "
                f"effective={record.effective_tile}  peak={record.peak_vram_mb:.0f}MB  "
                f"honoured={record.honoured}",
                flush=True,
            )

            outputs: dict[str, np.ndarray[Any, Any]] = {}
            for name, operator in ARMS:
                started = time.perf_counter()
                out = operator(neural)
                elapsed = time.perf_counter() - started
                outputs[name] = out

                luma_metrics = measure(out)
                difference = compare(outputs[REFERENCE_ARM], out) if name != REFERENCE_ARM else None
                rows.append(
                    {
                        "experiment": "real",
                        "category": image.category,
                        "arm": name,
                        "dimensions": [int(out.shape[1]), int(out.shape[0])],
                        "sha256": digest(out),
                        "post_seconds": elapsed,
                        "neural_pass": record.as_dict(),
                        "luma": _metrics_dict(luma_metrics),
                        "dark_hf": dark_region_hf(out),
                        "noise_sigma": noise_sigma(out),
                        "vs_baseline": difference.as_dict() if difference else None,
                    }
                )
                print(
                    f"    {name:14s} {elapsed:5.2f}s  sobel={luma_metrics.sobel_mean:.4f} "
                    f"hf={luma_metrics.high_frequency_ratio:.4f} "
                    f"flat={luma_metrics.flat_noise:.4f} "
                    f"over={luma_metrics.edge_overshoot:.4f} "
                    f"darkHF={rows[-1]['dark_hf']:.3f}",
                    flush=True,
                )

            rows.extend(_write_crops(image.category, outputs))
            outputs.clear()
            del neural, source

    finally:
        manager.release()

    return rows


# -------------------------------------------------------- the reference track


def degrade(hr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Fixed degradation: blur, downsample, noise, JPEG. In that order.

    The order is the physical one - optical blur happens at the lens, sampling
    at the sensor, noise at readout, compression at storage - and every
    parameter was fixed before any candidate was compared. Nothing here was
    tuned to make an arm win, and none of it is adaptive to the image.
    """
    import cv2

    blurred = cv2.GaussianBlur(hr, (0, 0), DEGRADE_BLUR_SIGMA)
    height, width = hr.shape[:2]
    small: np.ndarray[Any, Any] = cv2.resize(
        blurred, (width // SCALE, height // SCALE), interpolation=cv2.INTER_AREA
    )

    generator = np.random.default_rng(DEGRADE_SEED)
    noisy = small.astype(np.float32) + generator.normal(
        0.0, DEGRADE_NOISE_SIGMA, small.shape
    ).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 255.0).astype(np.uint8)

    ok, encoded = cv2.imencode(
        ".jpg", noisy[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), DEGRADE_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return np.asarray(decoded[:, :, ::-1])


def edge_preservation(reference: np.ndarray[Any, Any], test: np.ndarray[Any, Any]) -> float:
    """Correlation between the two frames' gradient magnitudes.

    Answers a question PSNR does not: are the edges in the same places with the
    same strengths. A candidate that invents structure lowers this even when it
    raises Sobel, because the invented gradient does not correlate with the
    reference's.
    """
    import cv2

    a = to_luma(reference).astype(np.float32) * LUMA_FULL_SCALE
    b = to_luma(test).astype(np.float32) * LUMA_FULL_SCALE
    ga = cv2.magnitude(
        cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    )
    gb = cv2.magnitude(
        cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    )
    ga = ga.ravel() - ga.mean()
    gb = gb.ravel() - gb.mean()
    denominator = float(np.sqrt((ga * ga).sum()) * np.sqrt((gb * gb).sum()))
    return float((ga * gb).sum() / denominator) if denominator > 0 else 0.0


def texture_error(reference: np.ndarray[Any, Any], test: np.ndarray[Any, Any]) -> float:
    """RMS difference of local texture energy, in luma levels.

    Local standard deviation is computed on both frames and compared. A frame
    that is too smooth and a frame that is too busy both score badly, which is
    what separates this from any "more is better" metric - it is an error, and
    zero is the target.
    """
    import cv2

    def energy(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        luma = to_luma(rgb).astype(np.float32) * LUMA_FULL_SCALE
        mean = cv2.boxFilter(luma, -1, (7, 7))
        sq = cv2.boxFilter(luma * luma, -1, (7, 7))
        return np.sqrt(np.maximum(sq - mean * mean, 0.0))

    delta = energy(reference) - energy(test)
    return float(np.sqrt(float(np.mean(delta * delta))))


def run_degradation(images: list[CorpusImage]) -> list[dict[str, Any]]:
    """The one track with ground truth, and therefore the one that can decide.

    A corpus photograph is the reference. It is degraded by the fixed pipeline,
    upscaled 4x by the production path, and each candidate is applied to that
    output. Every arm is then scored against the original.

    The comparison is cropped to a common grid rather than resized: `1732 // 4 *
    4` is 1732 but `1154 // 4 * 4` is 1152, so the reconstruction can be two
    rows short of the source. Cropping both to the overlap keeps every arm on
    exactly the same pixels and avoids putting a resample between a candidate
    and its score.
    """
    manager = _make_manager()
    rows: list[dict[str, Any]] = []

    try:
        for image in images:
            reference = load_rgb(image)
            low = degrade(reference)
            print(
                f"\n  {image.category}  reference {reference.shape[1]}x{reference.shape[0]}"
                f"  degraded {low.shape[1]}x{low.shape[0]}",
                flush=True,
            )
            neural, record = run_pass(manager, low)

            height = min(reference.shape[0], neural.shape[0])
            width = min(reference.shape[1], neural.shape[1])
            truth = np.ascontiguousarray(reference[:height, :width])
            print(
                f"    neural pass {record.seconds:5.1f}s  reconstruction "
                f"{neural.shape[1]}x{neural.shape[0]} -> compared on {width}x{height}",
                flush=True,
            )

            for name, operator in ARMS:
                out = np.ascontiguousarray(operator(neural)[:height, :width])
                difference = compare(truth, out)
                row = {
                    "experiment": "degradation",
                    "category": image.category,
                    "arm": name,
                    "dimensions": [int(width), int(height)],
                    "sha256": digest(out),
                    "psnr_db": difference.psnr_db,
                    "ssim": ssim_sampled(truth, out),
                    "rmse": difference.rmse,
                    "edge_preservation": edge_preservation(truth, out),
                    "texture_error": texture_error(truth, out),
                    "luma": _metrics_dict(measure(out)),
                    "reference_luma": _metrics_dict(measure(truth)),
                }
                rows.append(row)
                print(
                    f"    {name:14s} psnr={row['psnr_db']:6.2f}dB  ssim={row['ssim']:.5f}  "
                    f"edge_corr={row['edge_preservation']:.5f}  "
                    f"tex_err={row['texture_error']:6.3f}",
                    flush=True,
                )
                del out

            del neural, low, reference, truth

    finally:
        manager.release()

    return rows


# ------------------------------------------------------------------ crops


def _write_crops(category: str, outputs: dict[str, np.ndarray[Any, Any]]) -> list[dict[str, Any]]:
    """One crop per inspected region per arm, at 1:1 on the 4x output.

    Deliberately small: the regions come from the tracked crop table, not from
    taste, and only the regions this phase has to judge are written.
    """
    from PIL import Image as PILImage

    rows: list[dict[str, Any]] = []
    for region in INSPECT.get(category, ()):
        if region not in CROP_REGIONS.get(category, {}):
            continue
        cx, cy, size = CROP_REGIONS[category][region]
        span = min(CROP_SPAN, size * SCALE)
        for name, output in outputs.items():
            left = min(max(0, cx * SCALE - span // 2), max(0, output.shape[1] - span))
            top = min(max(0, cy * SCALE - span // 2), max(0, output.shape[0] - span))
            PILImage.fromarray(output[top : top + span, left : left + span]).save(
                CROPS_DIR / f"{category}__{region}__{name}.png"
            )
        rows.append(
            {
                "experiment": "crop",
                "category": category,
                "region": region,
                "span": span,
                "arms": list(outputs),
            }
        )
    return rows


def build_strips(seed: int = 20260909) -> list[dict[str, Any]]:
    """Blind comparison strips, panel order shuffled per region.

    F3 established at its own expense that knowing the expected direction is
    enough to make a reader describe a progression the pixels do not contain.
    The order is recorded here and never drawn on the image.
    """
    from PIL import Image as PILImage

    generator = np.random.default_rng(seed)
    names = [name for name, _ in ARMS]
    rows: list[dict[str, Any]] = []

    for category, regions in sorted(INSPECT.items()):
        for region in regions:
            available = [
                name for name in names if (CROPS_DIR / f"{category}__{region}__{name}.png").exists()
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
            print(f"    {out_name}  order recorded", flush=True)
    return rows


# ------------------------------------------------------------------- io


def _load_existing() -> dict[str, Any]:
    if MEASUREMENTS.exists():
        loaded: dict[str, Any] = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
        return loaded
    return {"provenance": provenance(), "rows": []}


def _save(document: dict[str, Any], rows: list[dict[str, Any]], experiments: set[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kept = [row for row in document.get("rows", []) if row.get("experiment") not in experiments]
    document["rows"] = kept + rows
    document["provenance"] = provenance()
    MEASUREMENTS.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\n  wrote {MEASUREMENTS.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", action="store_true", help="synthetic characterisation")
    parser.add_argument("--real", action="store_true", help="5 photographs x 5 arms, no reference")
    parser.add_argument("--degradation", action="store_true", help="reference-based track")
    parser.add_argument("--strips", action="store_true", help="blind comparison strips")
    arguments = parser.parse_args()

    if not any((arguments.operators, arguments.real, arguments.degradation, arguments.strips)):
        parser.print_help()
        return 2

    document = _load_existing()
    images = load_corpus()
    if (arguments.real or arguments.degradation) and not images:
        raise SystemExit("corpus is empty - run benchmarks/fetch_corpus.py first")

    if arguments.operators:
        _save(document, run_operators(), {"operators"})

    # Refused rather than warned. If a candidate turns out to be the unsharp mask
    # in different clothing, its picture results are not a new finding and should
    # never have been gathered as though they were.
    if (arguments.real or arguments.degradation) and not any(
        row.get("experiment") == "operators" for row in document.get("rows", [])
    ):
        raise SystemExit("run --operators first: an arm that duplicates B is not a new experiment")

    if arguments.real:
        _save(document, run_real(images), {"real", "crop"})

    if arguments.degradation:
        _save(document, run_degradation(images), {"degradation"})

    if arguments.strips:
        _save(document, build_strips(), {"strips"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
