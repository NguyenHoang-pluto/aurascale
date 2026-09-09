"""The Phase 5 detail-recovery primitives, checked against their own claims.

The phase rests on three claims, and each one is a place where a plausible bug
would produce a plausible-looking report rather than an error:

  * the guided filter is **edge preserving**, which is the whole reason arm `C`
    is not an unsharp mask. If it blurred like a Gaussian the detail layer would
    contain the edge and the arm would ring exactly like `B`;
  * **coherence separates oriented structure from isotropic noise**, which is
    the whole reason arm `D` can be selective. A coherence that returned 1
    everywhere - which is what a per-pixel structure tensor does - would gate
    nothing while appearing to work;
  * the arms **only move pixels along the neutral axis**, so no candidate can
    win by introducing a colour shift that the luma metrics cannot see.

The characterisation harness is tested too. It is the gate that decides whether
a candidate is a duplicate experiment, so a harness that reported plausible
numbers for the identity operator would be worse than no harness at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.phase5_detail_recovery import (
    ARMS,
    CORRECTION_CEILING,
    DEGRADE_JPEG_QUALITY,
    GUIDED_RADIUS,
    LUMA_FULL_SCALE,
    SCALE,
    _apply_luma_correction,
    arm_baseline,
    characterise,
    coherence,
    degrade,
    edge_preservation,
    guided_self,
    texture_error,
)


def grey(height: int = 96, width: int = 96, level: int = 128) -> np.ndarray:
    return np.full((height, width, 3), level, dtype=np.uint8)


def step_luma(size: int = 96, low: float = 80.0, high: float = 176.0) -> np.ndarray:
    field = np.full((size, size), low, dtype=np.float32)
    field[:, size // 2 :] = high
    return field


# --------------------------------------------------------- guided filter


def test_guided_filter_leaves_a_flat_field_alone() -> None:
    flat = np.full((64, 64), 100.0, dtype=np.float32)
    assert np.allclose(guided_self(flat, GUIDED_RADIUS, 200.0), 100.0, atol=1e-3)


def test_guided_filter_preserves_a_step_better_than_a_gaussian() -> None:
    """The property arm `C` depends on, stated as a comparison.

    A Gaussian of comparable support smears the step across its kernel; the
    guided filter keeps it. Measured as the width of the transition, so the test
    is about edge preservation rather than about any particular pixel.
    """
    import cv2

    field = step_luma()
    guided = guided_self(field, GUIDED_RADIUS, 200.0)
    gaussian = cv2.GaussianBlur(field, (0, 0), GUIDED_RADIUS / 2.0)

    row = field.shape[0] // 2
    span = slice(field.shape[1] // 2 - 8, field.shape[1] // 2 + 8)
    # Residual against the ideal step: how much of the edge each filter destroyed.
    guided_error = float(np.abs(guided[row, span] - field[row, span]).mean())
    gaussian_error = float(np.abs(gaussian[row, span] - field[row, span]).mean())
    assert guided_error < gaussian_error


def test_guided_detail_layer_is_small_at_a_hard_edge() -> None:
    field = step_luma()
    detail = field - guided_self(field, GUIDED_RADIUS, 200.0)
    centre = field.shape[1] // 2
    at_edge = float(np.abs(detail[:, centre - 1 : centre + 1]).mean())
    # The step is 96 levels; anything the filter left in the detail layer there
    # is what would ring. A few levels is preservation, half the step is not.
    assert at_edge < 96.0 * 0.25


@pytest.mark.parametrize("eps", [4.0, 200.0, 800.0])
def test_guided_filter_output_stays_in_range(eps: float) -> None:
    generator = np.random.default_rng(7)
    field = generator.uniform(0.0, 255.0, (64, 64)).astype(np.float32)
    out = guided_self(field, GUIDED_RADIUS, eps)
    assert np.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 256.0


# -------------------------------------------------------------- coherence


def test_coherence_is_bounded() -> None:
    generator = np.random.default_rng(11)
    field = generator.uniform(0.0, 255.0, (96, 96)).astype(np.float32)
    values = coherence(field)
    assert values.min() >= 0.0
    assert values.max() <= 1.0 + 1e-5


def test_coherence_is_higher_on_an_oriented_edge_than_on_noise() -> None:
    """The claim arm `D`'s selectivity rests on.

    If this ever fails, the gate is not separating structure from noise and `D`
    is only a weaker `C`.

    Compared on the edge band rather than on the whole frame. A step image is
    mostly two flat fields, where coherence is correctly zero because there is
    no structure to orient, so a whole-frame median would compare the noise
    against the flat regions and say nothing about the edge.
    """
    generator = np.random.default_rng(13)
    noise = 128.0 + generator.normal(0.0, 6.0, (96, 96)).astype(np.float32)
    edge = coherence(step_luma())
    centre = edge.shape[1] // 2
    at_edge = float(np.median(edge[:, centre - 3 : centre + 3]))
    assert at_edge > float(np.median(coherence(noise)))


def test_coherence_of_pure_noise_is_low() -> None:
    generator = np.random.default_rng(17)
    noise = 128.0 + generator.normal(0.0, 6.0, (128, 128)).astype(np.float32)
    assert float(np.median(coherence(noise))) < 0.5


def test_coherence_is_not_identically_one() -> None:
    """Guards the specific bug the smoothing exists to prevent.

    A structure tensor formed per pixel is rank 1 everywhere, so its coherence
    is 1 everywhere and the gate would be open on noise as wide as on hair.
    """
    generator = np.random.default_rng(19)
    noise = 128.0 + generator.normal(0.0, 8.0, (128, 128)).astype(np.float32)
    assert float(coherence(noise).mean()) < 0.95


# ------------------------------------------------------- the correction


def test_correction_is_clipped_to_the_ceiling() -> None:
    base = grey(level=128)
    correction = np.full(base.shape[:2], 500.0, dtype=np.float32)
    out = _apply_luma_correction(base, correction)
    assert int(out.max()) == int(128 + CORRECTION_CEILING)


def test_correction_moves_all_three_channels_equally() -> None:
    """No arm may introduce a colour fringe, so the correction is achromatic."""
    coloured = np.zeros((32, 32, 3), dtype=np.uint8)
    coloured[..., 0], coloured[..., 1], coloured[..., 2] = 120, 90, 60
    correction = np.full((32, 32), 5.0, dtype=np.float32)
    out = _apply_luma_correction(coloured, correction)
    deltas = out.astype(np.int16) - coloured.astype(np.int16)
    assert deltas[..., 0].max() == deltas[..., 1].max() == deltas[..., 2].max()


def test_correction_does_not_wrap_around_at_white() -> None:
    bright = grey(level=252)
    out = _apply_luma_correction(bright, np.full((96, 96), 40.0, dtype=np.float32))
    assert int(out.max()) == 255


# -------------------------------------------------------------- the arms


def test_baseline_returns_the_same_array() -> None:
    """A must be a true no-op, or every `vs_baseline` difference is a lie."""
    image = grey()
    assert arm_baseline(image) is image


@pytest.mark.parametrize("name", [name for name, _ in ARMS])
def test_every_arm_preserves_shape_and_dtype(name: str) -> None:
    operator = dict(ARMS)[name]
    generator = np.random.default_rng(23)
    image = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    out = operator(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", [name for name, _ in ARMS])
def test_every_arm_is_deterministic(name: str) -> None:
    operator = dict(ARMS)[name]
    generator = np.random.default_rng(29)
    image = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(operator(image.copy()), operator(image.copy()))


@pytest.mark.parametrize("name", [name for name, _ in ARMS])
def test_no_arm_shifts_the_level_of_a_flat_field(name: str) -> None:
    """A detail operator has no business changing exposure."""
    operator = dict(ARMS)[name]
    out = operator(grey(level=128))
    assert abs(float(out.mean()) - 128.0) < 0.5


# ------------------------------------------------- the characterisation


def test_characterisation_reports_the_identity_as_the_identity() -> None:
    """The gate's own control. If this drifts, no signature can be trusted."""
    signature = characterise(arm_baseline)
    assert signature.step_overshoot == 0.0
    assert signature.noise_gain == pytest.approx(1.0, abs=1e-6)
    assert signature.gain_mid == pytest.approx(1.0, abs=1e-6)
    assert signature.gain_fine == pytest.approx(1.0, abs=1e-6)
    assert signature.gain_flat == pytest.approx(1.0, abs=1e-6)


def test_characterisation_detects_amplification() -> None:
    def double_detail(rgb: np.ndarray) -> np.ndarray:
        import cv2

        luma = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(
            np.float32
        )
        return _apply_luma_correction(rgb, luma - cv2.GaussianBlur(luma, (0, 0), 2.0))

    signature = characterise(double_detail)
    assert signature.noise_gain > 1.0
    assert signature.gain_mid > 1.0


# --------------------------------------------------------- the degradation


def test_degradation_reduces_size_by_the_scale_factor() -> None:
    generator = np.random.default_rng(31)
    source = generator.integers(0, 256, (256, 320, 3), dtype=np.uint8)
    low = degrade(source)
    assert low.shape[0] == 256 // SCALE
    assert low.shape[1] == 320 // SCALE
    assert low.dtype == np.uint8


def test_degradation_is_reproducible() -> None:
    """A seeded noise term that moved between runs would make the whole
    reference track unrepeatable, and nothing downstream would notice."""
    generator = np.random.default_rng(37)
    source = generator.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    assert np.array_equal(degrade(source), degrade(source))


def test_degradation_actually_degrades() -> None:
    """Guards a pipeline that silently became a no-op."""
    import cv2

    generator = np.random.default_rng(41)
    source = generator.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    naive = cv2.resize(source, (32, 32), interpolation=cv2.INTER_AREA)
    assert not np.array_equal(degrade(source), naive)


def test_jpeg_quality_is_the_documented_one() -> None:
    assert DEGRADE_JPEG_QUALITY == 85


# ---------------------------------------------------- reference metrics


def test_edge_preservation_is_one_for_identical_images() -> None:
    generator = np.random.default_rng(43)
    image = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert edge_preservation(image, image) == pytest.approx(1.0, abs=1e-6)


def test_edge_preservation_falls_when_structure_moves() -> None:
    generator = np.random.default_rng(47)
    a = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert edge_preservation(a, b) < edge_preservation(a, a)


def test_texture_error_is_zero_for_identical_images() -> None:
    generator = np.random.default_rng(53)
    image = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert texture_error(image, image) == pytest.approx(0.0, abs=1e-6)


def test_texture_error_penalises_both_directions() -> None:
    """The property that separates this from a "more is better" metric.

    Smoothing a frame and roughening it must both score badly, or the metric
    would reward any candidate that adds energy.
    """
    import cv2

    generator = np.random.default_rng(59)
    reference = generator.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    smoother = cv2.GaussianBlur(reference, (0, 0), 2.0)
    rougher = np.clip(
        reference.astype(np.float32)
        + generator.normal(0.0, 12.0, reference.shape).astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    assert texture_error(reference, smoother) > 0.0
    assert texture_error(reference, rougher) > 0.0


def test_luma_scale_constant_is_the_eight_bit_one() -> None:
    assert LUMA_FULL_SCALE == 255.0
