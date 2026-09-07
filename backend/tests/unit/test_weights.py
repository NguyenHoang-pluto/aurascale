"""Checkpoint reading, architecture construction and DNI blending.

These use small synthetic checkpoints written to disk in the real release
formats, so the unwrapping logic is tested against the shapes it will actually
meet without downloading 64 MB. The real files are loaded in
tests/integration/test_real_inference.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.core.exceptions import ModelLoadError
from app.inference.weights import (
    blend_state_dicts,
    build_module,
    extract_state_dict,
    load_into,
    load_state_dict,
)
from app.services.model_service import ModelEntry


def entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "test",
        "name": "Test model",
        "description": "",
        "arch": "SRVGGNetCompact",
        "scale": 4,
        "file": "test.pth",
        "url": "https://example.invalid/test.pth",
        "arch_params": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 8,
            "num_conv": 1,
            "upscale": 4,
            "act_type": "prelu",
        },
    }
    defaults.update(overrides)
    return ModelEntry.model_validate(defaults)


# ------------------------------------------------------------- architectures


def test_both_architectures_can_be_built_from_manifest_parameters() -> None:
    compact = build_module(entry())
    rrdb = build_module(
        entry(
            arch="RRDBNet",
            arch_params={
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": 8,
                "num_block": 1,
                "num_grow_ch": 4,
                "scale": 4,
            },
        )
    )

    assert type(compact).__name__ == "SRVGGNetCompact"
    assert type(rrdb).__name__ == "RRDBNet"


def test_an_unknown_architecture_is_a_clear_error_not_an_import_failure() -> None:
    with pytest.raises(ModelLoadError) as caught:
        build_module(entry(arch="SwinIR"))

    assert caught.value.code.value == "model_load_failed"
    assert "architecture" in caught.value.message
    assert caught.value.technical is not None and "SwinIR" in caught.value.technical


def test_manifest_parameters_that_do_not_match_the_architecture_are_reported() -> None:
    with pytest.raises(ModelLoadError, match="does not match its architecture"):
        build_module(entry(arch_params={"nonsense": 1}))


# ---------------------------------------------------------- checkpoint shapes


def test_the_ema_weights_are_preferred_when_a_checkpoint_has_them() -> None:
    """The ESRGAN releases ship `params_ema`; using `params` instead would run
    the non-averaged generator and quietly produce worse output."""
    payload = {
        "params_ema": {"conv.weight": torch.ones(1)},
        "params": {"conv.weight": torch.zeros(1)},
    }

    state = extract_state_dict(payload, source="test.pth")

    assert torch.equal(state["conv.weight"], torch.ones(1))


def test_the_v3_format_stores_weights_under_params() -> None:
    payload = {"params": {"conv.weight": torch.ones(1)}}

    assert "conv.weight" in extract_state_dict(payload, source="test.pth")


def test_a_bare_state_dict_is_accepted() -> None:
    payload = {"conv.weight": torch.ones(1), "conv.bias": torch.zeros(1)}

    assert set(extract_state_dict(payload, source="test.pth")) == {"conv.weight", "conv.bias"}


def test_a_dataparallel_prefix_is_stripped() -> None:
    payload = {"module.conv.weight": torch.ones(1), "module.conv.bias": torch.zeros(1)}

    assert set(extract_state_dict(payload, source="test.pth")) == {"conv.weight", "conv.bias"}


def test_a_file_with_no_recognisable_weights_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="recognisable weights"):
        extract_state_dict({"optimizer": {"lr": 1}, "epoch": 3}, source="test.pth")


def test_something_that_is_not_a_mapping_is_rejected() -> None:
    with pytest.raises(ModelLoadError, match="not in a format"):
        extract_state_dict([1, 2, 3], source="test.pth")


# ------------------------------------------------------------------ on disk


def test_missing_weights_say_so_rather_than_raising_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError) as caught:
        load_state_dict(tmp_path / "absent.pth")

    assert "not downloaded yet" in caught.value.message


def test_a_truncated_checkpoint_suggests_downloading_it_again(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"PK\x03\x04 not really a checkpoint")

    with pytest.raises(ModelLoadError) as caught:
        load_state_dict(corrupt)

    assert "download it again" in caught.value.message


def test_a_real_checkpoint_round_trips(tmp_path: Path) -> None:
    module = build_module(entry())
    path = tmp_path / "weights.pth"
    torch.save({"params": module.state_dict()}, path)

    loaded = load_state_dict(path)

    assert set(loaded) == set(module.state_dict())


# ------------------------------------------------------------- strict loading


def test_weights_from_a_different_model_are_refused(tmp_path: Path) -> None:
    """`strict=True` is what stops a mismatched download producing noise."""
    module = build_module(entry())
    wrong = {name: torch.zeros(1) for name in module.state_dict()}

    with pytest.raises(ModelLoadError) as caught:
        load_into(module, wrong, model_id="test")

    assert "do not match the architecture" in caught.value.message


def test_matching_weights_load_cleanly() -> None:
    module = build_module(entry())

    result = load_into(module, module.state_dict(), model_id="test")

    assert result is module


# ------------------------------------------------------------------- DNI


def test_blending_at_one_returns_the_primary_weights_exactly() -> None:
    """The endpoints have to be exact, or "no denoising" would still alter the
    model."""
    primary = {"w": torch.tensor([1.0, 2.0])}
    secondary = {"w": torch.tensor([5.0, 9.0])}

    blended = blend_state_dicts(primary, secondary, 1.0)

    assert torch.equal(blended["w"], primary["w"])


def test_blending_at_zero_returns_the_secondary_weights_exactly() -> None:
    primary = {"w": torch.tensor([1.0, 2.0])}
    secondary = {"w": torch.tensor([5.0, 9.0])}

    blended = blend_state_dicts(primary, secondary, 0.0)

    assert torch.equal(blended["w"], secondary["w"])


def test_a_blend_is_the_weighted_sum_of_both_networks() -> None:
    primary = {"w": torch.tensor([0.0, 10.0])}
    secondary = {"w": torch.tensor([10.0, 0.0])}

    blended = blend_state_dicts(primary, secondary, 0.25)

    assert torch.allclose(blended["w"], torch.tensor([7.5, 2.5]))


def test_mismatched_parameter_names_are_refused() -> None:
    with pytest.raises(ModelLoadError, match="do not match"):
        blend_state_dicts({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(ModelLoadError, match="do not match"):
        blend_state_dicts({"w": torch.ones(2)}, {"w": torch.ones(3)}, 0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_an_out_of_range_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        blend_state_dicts({"w": torch.ones(1)}, {"w": torch.ones(1)}, alpha)


def test_a_blended_network_still_loads_strictly() -> None:
    """The blend must remain a valid state dict, not just matching arithmetic."""
    first = build_module(entry())
    second = build_module(entry())

    blended = blend_state_dicts(first.state_dict(), second.state_dict(), 0.5)

    assert load_into(build_module(entry()), blended, model_id="test") is not None
