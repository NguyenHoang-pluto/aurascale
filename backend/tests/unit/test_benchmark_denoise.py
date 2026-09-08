"""The Phase 2.5 corpus manifest and denoise matrix.

Deterministic and GPU-independent: nothing here runs inference, reaches the
network, or needs the corpus images to be present. What is checked is that a
malformed or incomplete manifest is refused rather than silently reducing the
corpus, and that the matrix holds every variable but one fixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks.corpus import (
    CATEGORIES,
    REQUIRED_FIELDS,
    ManifestEntry,
    ManifestError,
    load_manifest,
    manifest_path,
    parse_manifest,
)
from benchmarks.denoise import (
    BASELINE_DENOISE,
    DENOISE_VALUES,
    MODEL_ID,
    BenchmarkConfig,
    PlannedRun,
    arm_name,
    build_matrix,
    settings_payload,
)


def entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "filename": "landscape-detail/landscape-detail.jpg",
        "category": "landscape-detail",
        "source": "https://commons.wikimedia.org/wiki/File:Example.jpg",
        "license": "CC BY 4.0",
        "dimensions": {"width": 1732, "height": 1154},
        "notes": "a note",
    }
    base.update(overrides)
    return base


def manifest(*images: dict[str, Any]) -> dict[str, Any]:
    return {"description": "test", "images": list(images)}


# ------------------------------------------------------------------ manifest


def test_a_well_formed_manifest_parses() -> None:
    [parsed] = parse_manifest(manifest(entry()))

    assert isinstance(parsed, ManifestEntry)
    assert parsed.category == "landscape-detail"
    assert parsed.license == "CC BY 4.0"
    assert parsed.width == 1732
    assert parsed.megapixels == pytest.approx(2.0, abs=0.01)


def test_optional_provenance_fields_are_carried_when_present() -> None:
    [parsed] = parse_manifest(
        manifest(entry(attribution="A Photographer", license_url="https://example/cc-by"))
    )

    assert parsed.attribution == "A Photographer"
    assert parsed.license_url == "https://example/cc-by"


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_missing_required_field_is_refused(field: str) -> None:
    """Provenance is not optional.

    An image whose licence or source is unrecorded cannot be used, and a
    benchmark that skipped it silently would report on a corpus different from
    the one it names.
    """
    broken = entry()
    del broken[field]

    with pytest.raises(ManifestError, match=field):
        parse_manifest(manifest(broken))


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_an_empty_required_field_is_refused(field: str) -> None:
    with pytest.raises(ManifestError, match=field):
        parse_manifest(manifest(entry(**{field: ""})))


def test_an_unknown_category_is_refused() -> None:
    with pytest.raises(ManifestError, match="unknown category"):
        parse_manifest(manifest(entry(category="not-a-category")))


def test_every_declared_category_is_accepted() -> None:
    images = [entry(category=category, filename=f"{category}/x.jpg") for category in CATEGORIES]

    assert len(parse_manifest(manifest(*images))) == len(CATEGORIES)


def test_malformed_dimensions_are_refused() -> None:
    with pytest.raises(ManifestError, match="dimensions"):
        parse_manifest(manifest(entry(dimensions={"height": 100})))


def test_a_manifest_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ManifestError, match="must be an object"):
        parse_manifest([entry()])


def test_a_manifest_without_an_images_list_is_refused() -> None:
    with pytest.raises(ManifestError, match="no 'images' list"):
        parse_manifest({"description": "test"})


def test_an_image_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ManifestError, match="not an object"):
        parse_manifest({"images": ["landscape.jpg"]})


# -------------------------------------------------------- corpus discovery


def test_an_absent_manifest_says_how_to_create_one(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="fetch_corpus"):
        load_manifest(tmp_path)


def test_invalid_json_is_refused_with_the_path(tmp_path: Path) -> None:
    manifest_path(tmp_path).write_text("{ not json", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_a_manifest_naming_an_absent_file_is_refused(tmp_path: Path) -> None:
    """The manifest and the directory must agree, or the run is not the run."""
    manifest_path(tmp_path).write_text(json.dumps(manifest(entry())), encoding="utf-8")

    with pytest.raises(ManifestError, match="not present"):
        load_manifest(tmp_path)


def test_file_presence_can_be_waived_for_inspection(tmp_path: Path) -> None:
    manifest_path(tmp_path).write_text(json.dumps(manifest(entry())), encoding="utf-8")

    assert len(load_manifest(tmp_path, require_files=False)) == 1


def test_a_manifest_whose_files_exist_loads(tmp_path: Path) -> None:
    directory = tmp_path / "landscape-detail"
    directory.mkdir()
    (directory / "landscape-detail.jpg").write_bytes(b"not really a jpeg")
    manifest_path(tmp_path).write_text(json.dumps(manifest(entry())), encoding="utf-8")

    assert len(load_manifest(tmp_path)) == 1


# ---------------------------------------------------------- denoise matrix


def test_the_matrix_covers_every_image_at_every_value() -> None:
    entries = parse_manifest(
        manifest(
            entry(),
            entry(category="portrait-skin", filename="portrait-skin/portrait-skin.jpg"),
        )
    )

    matrix = build_matrix(entries)

    assert len(matrix) == 2 * len(DENOISE_VALUES)
    assert {run.denoise for run in matrix} == set(DENOISE_VALUES)


def test_the_matrix_is_image_major() -> None:
    """A partial run should still leave complete arms for the images it reached."""
    entries = parse_manifest(
        manifest(
            entry(),
            entry(category="portrait-skin", filename="portrait-skin/portrait-skin.jpg"),
        )
    )

    matrix = build_matrix(entries)

    assert [run.image for run in matrix[: len(DENOISE_VALUES)]] == [entries[0].filename] * len(
        DENOISE_VALUES
    )


def test_only_denoise_varies_across_the_matrix() -> None:
    """The experiment is one variable. Everything else is pinned."""
    matrix = build_matrix(parse_manifest(manifest(entry())))

    configs = {run.config for run in matrix}
    assert len(configs) == 1
    only = configs.pop()
    assert only.model == MODEL_ID
    assert only.scale == 4
    assert only.sharpen_strength == 0.0
    # PNG, so the encoder does not put its own quantisation between the model
    # and the metrics.
    assert only.output_format == "png"


def test_both_endpoints_are_measured() -> None:
    assert min(DENOISE_VALUES) == 0.0
    assert max(DENOISE_VALUES) == 1.0
    assert BASELINE_DENOISE in DENOISE_VALUES


def test_the_values_are_ordered_and_unique() -> None:
    assert list(DENOISE_VALUES) == sorted(set(DENOISE_VALUES))


def test_arm_names_are_stable_and_sort_in_order() -> None:
    names = [arm_name(value) for value in DENOISE_VALUES]

    assert names == sorted(names)
    assert names[0] == "denoise-0.00"
    assert names[-1] == "denoise-1.00"


def test_the_baseline_arm_is_identifiable() -> None:
    matrix = build_matrix(parse_manifest(manifest(entry())))
    baselines = [run for run in matrix if run.is_baseline]

    assert len(baselines) == 1
    assert baselines[0].denoise == BASELINE_DENOISE


def test_the_settings_payload_carries_denoise_and_nothing_else_by_default() -> None:
    run = PlannedRun(arm="denoise-0.50", denoise=0.5, image="x.jpg", category="landscape-detail")

    assert settings_payload(run) == {"denoiseStrength": 0.5}


def test_sharpening_reaches_the_payload_only_when_configured() -> None:
    run = PlannedRun(
        arm="denoise-0.50",
        denoise=0.5,
        image="x.jpg",
        category="landscape-detail",
        config=BenchmarkConfig(sharpen_strength=0.4),
    )

    assert settings_payload(run)["sharpenStrength"] == 0.4


def test_a_custom_value_set_is_honoured() -> None:
    matrix = build_matrix(parse_manifest(manifest(entry())), values=(0.0, 1.0))

    assert [run.denoise for run in matrix] == [0.0, 1.0]


def test_an_empty_corpus_yields_an_empty_matrix() -> None:
    assert build_matrix([]) == []
