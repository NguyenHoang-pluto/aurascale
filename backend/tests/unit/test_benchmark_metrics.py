"""The benchmark harness.

These check that each metric responds to the thing it claims to measure, and -
just as importantly - that the harness refuses to compare things that are not
comparable. A metric that silently returns a plausible number for a nonsense
comparison is worse than no metric.

Images are synthesised here rather than supplied: the corpus is deliberately
not committed, and these assertions are about the arithmetic, not about
photographs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from benchmarks.corpus import CATEGORIES, describe_corpus, load_corpus
from benchmarks.metrics import (
    ImageMetrics,
    edge_overshoot,
    flat_region_noise,
    high_frequency_ratio,
    local_contrast,
    measure,
    measure_file,
    sobel_statistics,
    tile_boxes,
    to_luma,
)
from benchmarks.report import Record, Report, RunCost, RunSettings, delta_between
from PIL import Image, ImageFilter


def rgb(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Widen a single-channel float image in [0, 1] to uint8 RGB."""
    scaled = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def flat(value: float = 0.5, size: int = 600) -> np.ndarray[Any, Any]:
    return rgb(np.full((size, size), value))


def stripes(period: int = 8, size: int = 600) -> np.ndarray[Any, Any]:
    """A square wave - strong edges at a known spatial frequency."""
    columns = np.arange(size)
    return rgb(np.tile(((columns // period) % 2).astype(np.float64), (size, 1)))


def noisy(sigma: float = 0.05, size: int = 600, seed: int = 3) -> np.ndarray[Any, Any]:
    generator = np.random.default_rng(seed)
    return rgb(np.clip(0.5 + generator.normal(0, sigma, (size, size)), 0, 1))


# ----------------------------------------------------------------- primitives


def test_luma_uses_rec709_weights() -> None:
    pure = np.zeros((4, 4, 3), dtype=np.uint8)
    pure[:, :, 1] = 255  # green carries most of the luma

    assert to_luma(pure) == pytest.approx(0.7152, abs=1e-4)


def test_luma_rejects_an_array_that_is_not_rgb() -> None:
    with pytest.raises(ValueError, match="expected HxWx3"):
        to_luma(np.zeros((4, 4), dtype=np.uint8))


def test_a_flat_image_measures_as_featureless() -> None:
    """Every metric must bottom out on an image with nothing in it."""
    result = measure(flat())

    assert result.sobel_mean == 0.0
    assert result.sobel_p95 == 0.0
    assert result.local_contrast == 0.0
    assert result.flat_noise == 0.0
    assert result.edge_overshoot == pytest.approx(0.0, abs=1e-12)


def test_edges_raise_the_gradient_statistics() -> None:
    edged, _ = sobel_statistics(to_luma(stripes()))
    smooth, _ = sobel_statistics(to_luma(flat()))

    assert edged > smooth


def test_blurring_lowers_the_peak_gradient() -> None:
    """The core claim the whole workstream rests on: softer measures softer.

    On the 95th percentile, not the mean. The integral of |grad| across a
    monotone edge is the height it climbs, so blurring spreads a step over more
    pixels without changing the mean - the two come out identical here. The
    peak is what collapses. This is exactly why both statistics are collected,
    and why reading either one alone would mislead.
    """
    source = stripes(period=16)
    sharp = to_luma(source)
    blurred = to_luma(
        np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(2)), dtype=np.uint8)
    )

    sharp_mean, sharp_p95 = sobel_statistics(sharp)
    blurred_mean, blurred_p95 = sobel_statistics(blurred)

    assert blurred_p95 < sharp_p95
    assert blurred_mean == pytest.approx(sharp_mean, rel=0.05)


def test_texture_raises_local_contrast() -> None:
    assert local_contrast(to_luma(noisy(0.1))) > local_contrast(to_luma(flat()))


def test_noise_is_measured_only_where_there_is_no_structure() -> None:
    quiet = flat_region_noise(to_luma(flat()))
    loud = flat_region_noise(to_luma(noisy(0.08)))

    assert quiet == 0.0
    assert loud > quiet


def test_fine_detail_raises_the_high_frequency_ratio() -> None:
    fine = high_frequency_ratio(to_luma(stripes(period=4)))
    coarse = high_frequency_ratio(to_luma(stripes(period=64)))

    assert fine > coarse


def test_overshoot_rises_when_an_edge_is_given_a_rim() -> None:
    """A halo is exactly what this proxy is for."""
    # Mid-tones, not a 0/1 square wave: a saturated edge has no headroom for
    # a rim and the clip puts it straight back where it started.
    base = to_luma(stripes(period=32)) * 0.4 + 0.3

    # A crude unsharp mask: the rim it leaves is what should be detected.
    blurred = base.copy()
    blurred[1:-1, 1:-1] = (
        base[:-2, 1:-1] + base[2:, 1:-1] + base[1:-1, :-2] + base[1:-1, 2:]
    ) / 4.0
    haloed = np.clip(base + 0.8 * (base - blurred), 0, 1)

    assert edge_overshoot(haloed) > edge_overshoot(base)


# -------------------------------------------------------------------- sampling


def test_tiles_are_deterministic_for_a_seed() -> None:
    """Two runs must measure the same regions or the deltas mean nothing."""
    assert tile_boxes(4000, 3000) == tile_boxes(4000, 3000)
    assert tile_boxes(4000, 3000, seed=1) != tile_boxes(4000, 3000, seed=2)


def test_a_small_image_is_measured_whole_rather_than_padded() -> None:
    assert tile_boxes(100, 80) == [(0, 0, 100, 80)]


def test_sampling_keeps_the_cost_flat_as_the_image_grows() -> None:
    """16 tiles whether the result is 12 MP or 192 MP."""
    assert len(tile_boxes(4000, 3000)) == 16
    assert len(tile_boxes(16000, 12000)) == 16


def test_every_box_lies_inside_the_image() -> None:
    for left, top, right, bottom in tile_boxes(16000, 12000):
        assert 0 <= left < right <= 16000
        assert 0 <= top < bottom <= 12000


def test_a_file_is_measured_at_its_own_resolution(tmp_path: Path) -> None:
    path = tmp_path / "result.png"
    Image.fromarray(noisy(0.06)).save(path)

    from_file = measure_file(path)
    from_array = measure(noisy(0.06))

    assert from_file.tiles_sampled == from_array.tiles_sampled
    assert from_file.sobel_mean == pytest.approx(from_array.sobel_mean, rel=1e-9)


# ---------------------------------------------------------------------- report


def record(arm: str, *, width: int = 4000, height: int = 3000, sobel: float = 0.1) -> Record:
    return Record(
        arm=arm,
        category="landscape-detail",
        image="lake.png",
        settings=RunSettings(model="RealESRGAN_x4plus", scale=4),
        input_width=1000,
        input_height=750,
        output_width=width,
        output_height=height,
        cost=RunCost(processing_ms=1000),
        metrics=ImageMetrics(
            sobel_mean=sobel,
            sobel_p95=sobel * 2,
            local_contrast=0.05,
            high_frequency_ratio=0.2,
            flat_noise=0.01,
            edge_overshoot=0.02,
            tiles_sampled=16,
        ),
    )


def test_a_delta_reports_both_absolute_and_relative_change() -> None:
    delta = delta_between(record("base", sobel=0.10), record("candidate", sobel=0.15))

    assert delta.comparable
    assert delta.absolute["sobel_mean"] == pytest.approx(0.05)
    assert delta.relative["sobel_mean"] == pytest.approx(50.0)


def test_a_cross_resolution_delta_is_refused_rather_than_approximated() -> None:
    """The metrics are resolution-dependent, so 4x vs 8x is not a comparison."""
    delta = delta_between(record("base"), record("eight", width=8000, height=6000))

    assert not delta.comparable
    assert delta.reason is not None
    assert "resolution-dependent" in delta.reason
    assert delta.absolute == {}


def test_a_missing_baseline_is_reported_not_silently_skipped() -> None:
    report = Report(baseline_arm="base")
    report.add(record("candidate"))

    deltas = report.deltas()

    assert len(deltas) == 1
    assert not deltas[0].comparable
    assert deltas[0].reason is not None
    assert "no baseline" in deltas[0].reason


def test_a_zero_baseline_yields_no_percentage_rather_than_infinity() -> None:
    delta = delta_between(record("base", sobel=0.0), record("candidate", sobel=0.1))

    assert delta.absolute["sobel_mean"] == pytest.approx(0.1)
    # NaN, not inf and not a fabricated 100%.
    assert delta.relative["sobel_mean"] != delta.relative["sobel_mean"]


def test_the_report_writes_json_and_text(tmp_path: Path) -> None:
    report = Report(baseline_arm="base")
    report.add(record("base", sobel=0.10))
    report.add(record("candidate", sobel=0.12))

    written = report.write_json(tmp_path / "out" / "report.json")

    assert written.is_file()
    assert "baseline: base" in report.to_text()
    # The reminder is part of the output, not a docstring nobody reads.
    assert "Higher is not automatically better" in report.to_text()


# ---------------------------------------------------------------------- corpus


def test_an_absent_corpus_is_described_rather_than_failing(tmp_path: Path) -> None:
    summary = describe_corpus(tmp_path)

    assert load_corpus(tmp_path) == []
    for category in CATEGORIES:
        assert category in summary
    assert "MISSING" in summary


def test_supplied_images_are_found_in_a_stable_order(tmp_path: Path) -> None:
    directory = tmp_path / "landscape-detail"
    directory.mkdir()
    for name in ("b.png", "a.png"):
        Image.fromarray(flat(size=32)).save(directory / name)
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")

    found = load_corpus(tmp_path)

    assert [image.name for image in found] == ["a.png", "b.png"]
    assert {image.category for image in found} == {"landscape-detail"}
