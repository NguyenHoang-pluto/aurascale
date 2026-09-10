"""Phase 3A: candidate pre-super-resolution denoisers, and a noise estimator.

Definition and execution of a *research* experiment. Nothing in `app/` imports
this module, and nothing here is wired into the pipeline: Phase 3A answers
whether adaptive pre-denoising is worth building, and deliberately stops before
building it.

Why pre-SR at all
-----------------

Real-ESRGAN's denoising is DNI weight interpolation - it blends the `x4v3`
weights toward their `wdn` counterpart at load time. That has two properties
this experiment exists to question:

  * it is **global**. One alpha for the whole image, chosen before anything has
    looked at the image;
  * it reaches **one model**. `realesr-general-x4v3` is the only registry entry
    with a denoise pair, so Standard (`RealESRGAN_x4plus`) has no denoising
    stage at all.

Phase 2.5 measured the cost of that global choice on real photographs and found
it varies by more than an order of magnitude: on the low-light image, denoising
cost 0.22-0.45 units of high-frequency detail per unit of noise removed; on the
portrait, where there was little noise to begin with, 0.25 cost **6.7x** more
detail than the noise it removed. A single default cannot be right for both.
That is the case for adaptivity, and it is the hypothesis under test here.

Where a prefilter would sit
---------------------------

Before the first neural pass, on the **input**. That placement is not
incidental - it is the memory argument. `max_input_pixels` is 16 MP;
`max_output_pixels` is 200 MP. A filter on the input is bounded 12.5x lower
than the same filter on the output, and the full-resolution `GaussianBlur` that
caused a real out-of-memory failure in this project was on the *output* side.
Task 9 asks that the class of bug not be reintroduced; running before the
upscale rather than after it is most of that answer, and this module measures
the rest.

What is deliberately not here
-----------------------------

No production integration, no defaults changed, no new dependency. Every
candidate is OpenCV or NumPy, both already required. A candidate that needed
`scikit-image` or `scipy` would be an engineering decision rather than a
benchmark arm, and is assessed in the report instead of being installed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------- noise levels

#: The four rungs Task 4 asks for. Named rather than numbered because the
#: parameter that expresses "medium" differs per algorithm - a bilateral sigma
#: and an NLM `h` are not on the same scale and pretending otherwise would make
#: the comparison meaningless.
STRENGTHS: tuple[str, ...] = ("none", "weak", "medium", "strong")


@dataclass(frozen=True, slots=True)
class FilterSpec:
    """One candidate, at one strength.

    `family` groups the rungs of a single algorithm so the report can show a
    ladder; `arm` is the unique identifier used in the measurements file.
    """

    arm: str
    family: str
    strength: str
    #: Human-readable parameters, recorded verbatim for reproducibility.
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.family == "none"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------- the candidates
#
# Parameters are chosen so the three rungs of each family span a comparable
# *perceptual* range rather than a comparable numeric one, calibrated against
# the corpus in a pilot pass: "weak" should be barely visible at 100 %,
# "strong" should be obviously over-filtered. Getting that span right matters
# more than the exact numbers, because the finding is where each family sits on
# the noise-versus-texture trade, not which sigma is optimal.


def _gaussian(sigma: float) -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Isotropic Gaussian blur. The control, not a serious candidate.

    Included because it is the obvious thing to reach for and because it is the
    reference failure: it has no notion of an edge, so whatever it does to
    noise it does equally to texture. Every edge-aware candidate has to beat
    it, and a candidate that does not is not worth its runtime.
    """

    def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import cv2

        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return run


def _bilateral(diameter: int, sigma_colour: float, sigma_space: float) -> Callable[..., Any]:
    """Edge-preserving: average only over neighbours of similar intensity.

    The classic answer to "blur the noise, keep the edges". Its known failure
    is exactly the one Task 2 warns about - at high `sigma_colour` it flattens
    low-contrast texture into plastic-looking patches, because fine texture and
    noise are both small intensity excursions and the filter cannot tell them
    apart. The strong rung is included to make that failure visible rather than
    to recommend it.
    """

    def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import cv2

        return cv2.bilateralFilter(image, diameter, sigma_colour, sigma_space)

    return run


def _nlm(h_luma: float, h_colour: float) -> Callable[..., Any]:
    """Non-local means: average over *similar patches*, not nearby pixels.

    The quality reference for photographic denoising. It exploits the
    self-similarity real photographs have and noise does not, which is why it
    preserves texture better than any local filter. It is also by far the most
    expensive candidate, and whether that cost is affordable at 16 MP is a
    question this benchmark answers with a measurement rather than an opinion.
    """

    def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import cv2

        return cv2.fastNlMeansDenoisingColored(
            image, None, h_luma, h_colour, templateWindowSize=7, searchWindowSize=21
        )

    return run


def _chroma_only(diameter: int, sigma_colour: float, sigma_space: float) -> Callable[..., Any]:
    """Denoise the colour channels and leave luminance untouched.

    The one candidate designed around Task 2's distinction rather than around
    an algorithm. Chroma noise is the blotchy colour mottle that looks worst in
    dark areas; luminance noise reads as film grain and is far less
    objectionable. Crucially, almost all the *detail* the eye resolves lives in
    luminance - which is why 4:2:0 subsampling has been acceptable for decades.

    So this filters Cr and Cb hard and Y not at all. If the trade is as
    asymmetric as that reasoning suggests, this should remove a large share of
    the visible noise at nearly zero cost in measured detail - and `metrics`,
    which works on Rec.709 luma, should be *almost blind to it*. That blindness
    is itself the finding, and is why this arm is also inspected visually.
    """

    def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import cv2

        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        cr = cv2.bilateralFilter(cr, diameter, sigma_colour, sigma_space)
        cb = cv2.bilateralFilter(cb, diameter, sigma_colour, sigma_space)
        return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2RGB)

    return run


def _median(kernel: int) -> Callable[..., Any]:
    """A median filter. Cheap, and the right tool for impulse noise only.

    Benchmarked to be ruled in or out rather than assumed: it is O(1) per pixel
    at small kernels and costs almost nothing, so if it were competitive on
    photographic noise it would be an attractive default. It is expected not to
    be - median filtering erases fine texture at any kernel large enough to
    matter - and the point is to have the number rather than the expectation.
    """

    def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import cv2

        result: np.ndarray[Any, Any] = cv2.medianBlur(image, kernel)
        return result

    return run


def _identity(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Arm A. The baseline every delta is measured against."""
    return image


#: Every arm, in report order. The parameter ladders were set in a pilot pass
#: over the corpus; the exact values are recorded here because they are part of
#: the experiment's reproducibility, not an implementation detail.
CANDIDATES: tuple[tuple[FilterSpec, Callable[..., Any]], ...] = (
    (FilterSpec("none", "none", "none", {}, "Arm A - no pre-denoise"), _identity),
    # B - Gaussian, the edge-blind control.
    (FilterSpec("gaussian-weak", "gaussian", "weak", {"sigma": 0.5}), _gaussian(0.5)),
    (FilterSpec("gaussian-medium", "gaussian", "medium", {"sigma": 1.0}), _gaussian(1.0)),
    (FilterSpec("gaussian-strong", "gaussian", "strong", {"sigma": 1.8}), _gaussian(1.8)),
    # C - bilateral, edge-preserving.
    (
        FilterSpec(
            "bilateral-weak",
            "bilateral",
            "weak",
            {"d": 5, "sigma_colour": 15, "sigma_space": 5},
        ),
        _bilateral(5, 15, 5),
    ),
    (
        FilterSpec(
            "bilateral-medium",
            "bilateral",
            "medium",
            {"d": 7, "sigma_colour": 35, "sigma_space": 7},
        ),
        _bilateral(7, 35, 7),
    ),
    (
        FilterSpec(
            "bilateral-strong",
            "bilateral",
            "strong",
            {"d": 9, "sigma_colour": 75, "sigma_space": 9},
        ),
        _bilateral(9, 75, 9),
    ),
    # C2 - chroma-only bilateral, the Task 2 candidate.
    (
        FilterSpec(
            "chroma-weak",
            "chroma",
            "weak",
            {"d": 5, "sigma_colour": 20, "sigma_space": 5},
        ),
        _chroma_only(5, 20, 5),
    ),
    (
        FilterSpec(
            "chroma-medium",
            "chroma",
            "medium",
            {"d": 7, "sigma_colour": 45, "sigma_space": 7},
        ),
        _chroma_only(7, 45, 7),
    ),
    (
        FilterSpec(
            "chroma-strong",
            "chroma",
            "strong",
            {"d": 9, "sigma_colour": 90, "sigma_space": 9},
        ),
        _chroma_only(9, 90, 9),
    ),
    # D - non-local means.
    (FilterSpec("nlm-weak", "nlm", "weak", {"h": 3.0, "h_colour": 3.0}), _nlm(3.0, 3.0)),
    (FilterSpec("nlm-medium", "nlm", "medium", {"h": 6.0, "h_colour": 6.0}), _nlm(6.0, 6.0)),
    (FilterSpec("nlm-strong", "nlm", "strong", {"h": 10.0, "h_colour": 10.0}), _nlm(10.0, 10.0)),
    # Median, benchmarked to be ruled out with evidence.
    (FilterSpec("median-3", "median", "weak", {"kernel": 3}), _median(3)),
    (FilterSpec("median-5", "median", "medium", {"kernel": 5}), _median(5)),
)


def candidate_by_arm(arm: str) -> tuple[FilterSpec, Callable[..., Any]]:
    """Look one arm up by name, for the runner and for the tests."""
    for spec, run in CANDIDATES:
        if spec.arm == arm:
            return spec, run
    raise KeyError(f"no such arm: {arm}")


# ----------------------------------------------------------- noise estimation
#
# Task 7. The design constraint is that it be *explainable*: a score whose
# value can be traced to a measurable property of the image, not a fitted
# black box. Two signals, both computed on the flattest blocks only.

#: Blocks this size are scored for flatness. Large enough to hold noise
#: statistics, small enough that a 2 MP image yields thousands of them.
NOISE_BLOCK = 32

#: Only the flattest share of blocks is measured. This is the whole trick: a
#: photograph's texture lives in high-variance blocks, so restricting to the
#: low-variance tail is what keeps foliage from reading as noise. 10 % is
#: deliberately tighter than `metrics.FLAT_PERCENTILE` (20 %), because an
#: estimator that decides how hard to filter should err toward *under*
#: estimating.
NOISE_FLAT_PERCENTILE = 10.0

#: Immerkaer's Laplacian mask. Convolving with it annihilates locally linear
#: signal - which is what a smooth patch is - and leaves noise, so the mean
#: absolute response over flat blocks estimates the noise sigma. The 6 and the
#: sqrt(pi/2) are the normalisation from the original derivation.
_IMMERKAER = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


@dataclass(frozen=True, slots=True)
class NoiseEstimate:
    """What the estimator saw, not only what it concluded.

    Every intermediate is kept because a score alone cannot be argued with. If
    the mapping later behaves oddly on an image, these are what say whether the
    estimator misread it or the mapping mishandled a correct reading.
    """

    #: Noise sigma in luma, 0-255 scale, from the flattest blocks.
    sigma_luma: float
    #: The same in chroma, averaged over Cr and Cb.
    sigma_chroma: float
    #: Share of blocks that qualified as flat. Low means a densely textured
    #: frame where the estimate rests on little evidence.
    flat_fraction: float
    #: Blocks measured. Provenance for the two figures above.
    blocks: int
    #: The combined score in [0, 1].
    score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _blocks_of(channel: np.ndarray[Any, Any], size: int = NOISE_BLOCK) -> np.ndarray[Any, Any]:
    """Reshape a channel into non-overlapping size x size blocks."""
    height, width = channel.shape[:2]
    rows, columns = height // size, width // size
    if rows == 0 or columns == 0:
        return channel.reshape(1, *channel.shape)

    trimmed = channel[: rows * size, : columns * size]
    blocks = trimmed.reshape(rows, size, columns, size).swapaxes(1, 2)
    return blocks.reshape(rows * columns, size, size)


def _immerkaer_sigma(blocks: np.ndarray[Any, Any]) -> float:
    """Noise sigma over a stack of blocks, by the Laplacian-response method.

    Computed per block and then taken as a **median** across blocks rather than
    a mean. A median is what makes the estimator robust: a handful of blocks
    that are flat by variance but contain a hard edge would drag a mean upward,
    and there is always a handful.
    """
    if blocks.size == 0:
        return 0.0

    # Valid-region 3x3 convolution across the whole stack at once.
    response = np.zeros((blocks.shape[0], blocks.shape[1] - 2, blocks.shape[2] - 2))
    for dy in range(3):
        for dx in range(3):
            weight = _IMMERKAER[dy, dx]
            if weight != 0.0:
                response += (
                    weight * blocks[:, dy : dy + response.shape[1], dx : dx + response.shape[2]]
                )

    if response.shape[1] == 0 or response.shape[2] == 0:
        return 0.0

    # sqrt(pi/2) / 6 converts the mean absolute Laplacian response to sigma.
    per_block = np.sqrt(np.pi / 2.0) / 6.0 * np.abs(response).mean(axis=(1, 2))
    return float(np.median(per_block))


#: The two anchors the score is stretched between. Both are **measured**, and
#: they come from different evidence, which is the single most important
#: caveat in this phase.
#:
#: `SIGMA_CLEAN` is constrained by the real corpus. Its five images measure
#: 0.30-0.73 combined sigma, and Phase 2.5 independently established which of
#: them denoising *harms*: text-signage and portrait-skin paid 1.34x and 6.71x
#: more detail than the noise they gained back. The floor is placed above
#: those and at the level of landscape-detail, so the images Phase 2.5 showed
#: denoising hurting are the images this scores at zero.
#:
#: `SIGMA_NOISY` is **not** constrained by the real corpus, because the corpus
#: does not contain a genuinely noisy photograph - its noisiest image sits at
#: 0.73, only 2.4x the cleanest. The ceiling instead comes from the injection
#: ladder in `results/phase3a_measurements.json`, where the estimator was shown
#: linear against known noise: 3.0 is roughly what this estimator reads on a
#: photograph with per-channel sigma 5-6, which is visibly noisy without being
#: extreme. This is the weakest number in the phase and the report says so.
SIGMA_CLEAN = 0.45
SIGMA_NOISY = 3.0

#: How much the chroma reading contributes. Chroma noise is more objectionable
#: per unit than luma noise, but it is also the cheaper thing to remove, so it
#: raises the score less than it raises the case for chroma-only filtering.
CHROMA_WEIGHT = 0.35


def estimate_noise(rgb: np.ndarray[Any, Any]) -> NoiseEstimate:
    """A small, explainable noise score for an RGB uint8 image.

    Three steps, each of which can be inspected in the returned estimate:

      1. split into 32 px blocks and keep only the flattest 10 % - texture
         lives in the rest, and counting it would be the exact failure Task 2
         names;
      2. estimate sigma on those blocks by Immerkaer's Laplacian response,
         separately for luma and chroma, taking the median across blocks;
      3. stretch the combined sigma between two anchors calibrated on the real
         corpus, and clamp to [0, 1].

    The score is not a physical quantity and is not claimed to be one. It is a
    monotone, bounded, reproducible summary of "how much of what this image
    contains is noise rather than signal", which is what a mapping needs.
    """
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3, got {rgb.shape}")

    ycrcb = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2YCrCb).astype(np.float64)
    luma, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    luma_blocks = _blocks_of(luma)
    variance = luma_blocks.var(axis=(1, 2))

    if variance.size == 0:
        return NoiseEstimate(0.0, 0.0, 0.0, 0, 0.0)

    threshold = float(np.percentile(variance, NOISE_FLAT_PERCENTILE))
    flat = variance <= threshold
    flat_fraction = float(flat.mean())

    sigma_luma = _immerkaer_sigma(luma_blocks[flat])
    sigma_cr = _immerkaer_sigma(_blocks_of(cr)[flat])
    sigma_cb = _immerkaer_sigma(_blocks_of(cb)[flat])
    sigma_chroma = (sigma_cr + sigma_cb) / 2.0

    combined = sigma_luma + CHROMA_WEIGHT * sigma_chroma
    score = (combined - SIGMA_CLEAN) / (SIGMA_NOISY - SIGMA_CLEAN)

    return NoiseEstimate(
        sigma_luma=sigma_luma,
        sigma_chroma=sigma_chroma,
        flat_fraction=flat_fraction,
        blocks=int(variance.size),
        score=float(min(1.0, max(0.0, score))),
    )


def calibrate_anchors(sigmas: list[float]) -> tuple[float, float]:
    """Where the clean and noisy anchors sit, given measured sigmas.

    The 20th and 80th percentiles of the corpus, rather than its extremes: an
    anchor placed on the single cleanest image would make every other image
    read as noisy, and one placed on the noisiest would flatten the useful
    range. Reported alongside the corpus it was derived from, because a
    five-image corpus cannot support more precision than that.
    """
    if not sigmas:
        raise ValueError("no sigmas to calibrate from")

    low = float(np.percentile(sigmas, 20))
    high = float(np.percentile(sigmas, 80))
    return low, high


# -------------------------------------------------------------- the mapping
#
# Task 8. Score in, denoise strength out.

#: The ceiling. Nothing the mapping proposes exceeds this, whatever the score,
#: because Phase 2.5 measured what stronger settings cost and the answer was
#: "more detail than they are worth". A cap is the mechanism that makes
#: "avoid aggressive denoise" a property of the design rather than a hope.
#:
#: The cap is not what failed - see `adaptive_strength`. Even capped, a mapping
#: driven by noise level alone applies 0.6 to a noisy portrait.
MAX_ADAPTIVE_STRENGTH = 0.6

#: Below this score nothing is applied at all. A clean image should be left
#: alone, and "almost nothing" is not the same as nothing: Phase 2.5 found the
#: portrait paying 6.7x more detail than it gained even at 0.25.
DEADBAND = 0.15


def adaptive_strength(score: float) -> float:
    """Denoise strength for a noise score. **Refuted - do not build on this.**

    The shape is a deadband then a straight line to the cap:

        score <= 0.15          ->  0.0     leave clean images alone
        0.15 < score < 1.0     ->  linear ramp
        score >= 1.0           ->  0.6     the cap

    It is kept because the experiment that refuted it needs something to refute,
    and because the refutation is the most useful thing Phase 3A found.

    What was tested
    ---------------

    The mapping rests on one premise: that a high noise score means denoising
    is cheap, and a low one means it is expensive. That premise was tested by
    injecting known noise into the five real photographs, scoring each, and
    measuring what `nlm-medium` then cost in high-frequency energy per unit of
    noise removed.

    The premise failed. Across 25 image-noise pairs the correlation between
    score and cost was **r = +0.077** - no relationship, and the sign is the
    opposite of the one predicted. The cost is instead a property of the
    *content*: landscape sat at 0.23-0.31 and portrait at 0.86-0.91, each
    essentially constant across every noise level from clean to sigma 16.

    The portrait is the case that matters. Injected at sigma 16 it scores a
    saturated 1.000, so this function proposes 0.6 - and the visual crop at
    that setting shows skin rendered plastic, every pore gone. A mapping from
    noise level alone will over-filter a noisy portrait, because how much noise
    an image carries says nothing about how much fine texture is at risk.

    A second, independent signal - a measure of the fine texture that would be
    destroyed - is necessary and is not designed here. `estimate_noise` remains
    validated and useful; this function is the part that does not work.
    """
    if score <= DEADBAND:
        return 0.0

    span = 1.0 - DEADBAND
    return float(min(MAX_ADAPTIVE_STRENGTH, MAX_ADAPTIVE_STRENGTH * (score - DEADBAND) / span))


# ----------------------------------------------------------------- execution


@dataclass(frozen=True, slots=True)
class FilterCost:
    """What one filter application cost. Measured, never estimated."""

    elapsed_ms: float
    #: Increase in resident memory across the call, in MiB. A crude figure -
    #: the allocator may not return memory to the OS - so it is reported as a
    #: floor on what was needed, not a peak.
    rss_delta_mb: float | None
    megapixels: float

    @property
    def ms_per_megapixel(self) -> float:
        return self.elapsed_ms / self.megapixels if self.megapixels else 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ms_per_megapixel"] = self.ms_per_megapixel
        return payload


def apply_filter(image: np.ndarray[Any, Any], arm: str) -> tuple[np.ndarray[Any, Any], FilterCost]:
    """Run one arm over an image and measure what it cost."""
    _, run = candidate_by_arm(arm)

    before = _rss_mb()
    started = time.perf_counter()
    result = run(image)
    elapsed = (time.perf_counter() - started) * 1000.0
    after = _rss_mb()

    return result, FilterCost(
        elapsed_ms=elapsed,
        rss_delta_mb=None if before is None or after is None else after - before,
        megapixels=image.shape[0] * image.shape[1] / 1_000_000,
    )


def _rss_mb() -> float | None:
    """Current resident set size, when psutil can report it."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return None

    return float(psutil.Process().memory_info().rss) / 1024 / 1024
