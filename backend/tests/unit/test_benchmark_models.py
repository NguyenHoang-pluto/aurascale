"""The Phase 3 model comparison matrix.

Deterministic and GPU-independent: nothing here loads weights, runs inference
or needs the corpus present. What is checked is that the matrix compares
models fairly - one variable, everything else pinned - and that the denoise
setting only reaches the one model that has a pair.
"""

from __future__ import annotations

from typing import Any

import pytest
from benchmarks.corpus import ManifestEntry
from benchmarks.models import (
    CANDIDATES,
    COMPARISON_FORMAT,
    COMPARISON_SCALE,
    REFERENCE_ARM,
    ComparisonConfig,
    ModelArm,
    PlannedComparison,
    build_matrix,
    settings_payload,
)

#: The model that carries a denoise pair. Every other candidate must be sent
#: no denoise at all - the API refuses it rather than ignoring it.
DENOISE_CAPABLE = "realesr-general-x4v3"


def entry(category: str = "landscape-detail") -> ManifestEntry:
    return ManifestEntry(
        filename=f"{category}/{category}.jpg",
        category=category,
        source="https://commons.wikimedia.org/wiki/File:Example.jpg",
        license="CC BY 4.0",
        width=1732,
        height=1154,
    )


def plan(**overrides: Any) -> PlannedComparison:
    base: dict[str, Any] = {
        "arm": "x4plus",
        "model": "RealESRGAN_x4plus",
        "denoise": None,
        "image": "landscape-detail/landscape-detail.jpg",
        "category": "landscape-detail",
    }
    base.update(overrides)
    return PlannedComparison(**base)


# ------------------------------------------------------------------ candidates


def test_the_reference_arm_is_the_production_default() -> None:
    """Deltas are meaningless unless the baseline is what ships today."""
    reference = [arm for arm in CANDIDATES if arm.is_reference]

    assert len(reference) == 1
    assert reference[0].model == "RealESRGAN_x4plus"
    assert reference[0].name == REFERENCE_ARM


def test_arm_names_are_unique() -> None:
    names = [arm.name for arm in CANDIDATES]

    assert len(names) == len(set(names))


def test_every_candidate_explains_why_it_is_present() -> None:
    """A model in the matrix without a stated reason is one nobody can review."""
    for arm in CANDIDATES:
        assert arm.notes.strip(), arm.name


def test_denoise_is_only_set_for_the_model_that_has_a_pair() -> None:
    """The setting does not reach the others, and sending it would fail a run."""
    for arm in CANDIDATES:
        if arm.denoise is not None:
            assert arm.model == DENOISE_CAPABLE, arm.name


def test_the_denoise_capable_model_is_measured_at_both_ends() -> None:
    """Once like-for-like, once as production ships it.

    Comparing it only at the shipped default would measure its denoiser rather
    than its network, against two models that have no denoiser at all.
    """
    values = {arm.denoise for arm in CANDIDATES if arm.model == DENOISE_CAPABLE}

    assert values == {0.0, 1.0}


def test_every_candidate_names_a_distinct_configuration() -> None:
    configurations = {(arm.model, arm.denoise) for arm in CANDIDATES}

    assert len(configurations) == len(CANDIDATES)


# ---------------------------------------------------------------------- matrix


def test_the_matrix_covers_every_image_through_every_arm() -> None:
    entries = [entry("landscape-detail"), entry("portrait-skin")]

    matrix = build_matrix(entries)

    assert len(matrix) == len(entries) * len(CANDIDATES)
    assert {run.arm for run in matrix} == {arm.name for arm in CANDIDATES}


def test_the_matrix_is_image_major() -> None:
    """An interrupted run still leaves complete arms for the images it reached."""
    entries = [entry("landscape-detail"), entry("portrait-skin")]

    matrix = build_matrix(entries)

    assert {run.image for run in matrix[: len(CANDIDATES)]} == {entries[0].filename}


def test_only_the_model_varies_across_the_matrix() -> None:
    """Everything else is pinned, or the arms are not comparable."""
    matrix = build_matrix([entry()])

    configs = {run.config for run in matrix}
    assert len(configs) == 1
    only = configs.pop()
    assert only.scale == COMPARISON_SCALE == 4
    assert only.output_format == COMPARISON_FORMAT == "png"
    assert only.sharpen_strength == 0.0
    assert only.tile_size is None
    assert only.tile_pad is None


def test_the_comparison_scale_is_native_to_every_candidate() -> None:
    """4x, so no arm pays for a cascade the others do not."""
    assert COMPARISON_SCALE == 4


def test_the_matrix_carries_exactly_one_reference_run_per_image() -> None:
    matrix = build_matrix([entry("landscape-detail"), entry("portrait-skin")])

    references = [run for run in matrix if run.is_reference]

    assert len(references) == 2
    assert {run.image for run in references} == {
        "landscape-detail/landscape-detail.jpg",
        "portrait-skin/portrait-skin.jpg",
    }


def test_an_empty_corpus_yields_an_empty_matrix() -> None:
    assert build_matrix([]) == []


def test_a_custom_arm_set_is_honoured() -> None:
    arms = (ModelArm(name="only", model="RealESRGAN_x2plus", notes="x"),)

    matrix = build_matrix([entry()], arms=arms)

    assert [run.arm for run in matrix] == ["only"]


# ------------------------------------------------------------------- payload


def test_no_settings_are_sent_for_a_model_without_a_denoise_pair() -> None:
    """None, not an empty object: there is genuinely nothing to configure."""
    assert settings_payload(plan(denoise=None)) is None


def test_denoise_reaches_the_payload_when_the_model_supports_it() -> None:
    assert settings_payload(plan(denoise=0.0)) == {"denoiseStrength": 0.0}
    assert settings_payload(plan(denoise=1.0)) == {"denoiseStrength": 1.0}


def test_zero_denoise_is_sent_rather_than_treated_as_absent() -> None:
    """0.0 is a real setting - fully the wdn weights - not "unset"."""
    payload = settings_payload(plan(denoise=0.0))

    assert payload is not None
    assert payload["denoiseStrength"] == 0.0


def test_sharpening_reaches_the_payload_only_when_configured() -> None:
    payload = settings_payload(plan(config=ComparisonConfig(sharpen_strength=0.4)))

    assert payload is not None
    assert payload["sharpenStrength"] == 0.4


def test_the_default_configuration_sends_no_sharpening() -> None:
    assert "sharpenStrength" not in (settings_payload(plan(denoise=0.5)) or {})


@pytest.mark.parametrize("arm", CANDIDATES, ids=lambda a: a.name)
def test_every_candidate_produces_a_usable_payload(arm: ModelArm) -> None:
    run = plan(arm=arm.name, model=arm.model, denoise=arm.denoise)

    payload = settings_payload(run)

    if arm.denoise is None:
        assert payload is None
    else:
        assert payload == {"denoiseStrength": arm.denoise}
