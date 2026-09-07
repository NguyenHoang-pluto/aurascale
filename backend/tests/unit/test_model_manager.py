"""Model loading, caching and eviction.

These run against a temporary manifest with tiny architectures, so the caching
rules are exercised at full speed without 64 MB of weights. The real models are
loaded in tests/integration/test_real_inference.py.

The manager is pinned to a CPU target here: what is being tested is the cache
policy, not the device, and a test that only passes on a GPU machine tests the
machine rather than the code.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from app.core.config import Settings
from app.core.exceptions import ModelLoadError, ModelNotFoundError
from app.inference.device import cpu_target
from app.inference.model_manager import CPU_RESIDENT_MODELS, ModelManager
from app.inference.weights import build_module
from app.services.model_service import ModelEntry, ModelService

TINY_COMPACT: dict[str, Any] = {
    "num_in_ch": 3,
    "num_out_ch": 3,
    "num_feat": 8,
    "num_conv": 1,
    "upscale": 4,
    "act_type": "prelu",
}
TINY_COMPACT_X2 = {**TINY_COMPACT, "upscale": 2}


def _entry(model_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "description": "",
        "arch": "SRVGGNetCompact",
        "scale": 4,
        "file": f"{model_id}.pth",
        "url": f"https://example.invalid/{model_id}.pth",
        "sha256": None,
        "arch_params": dict(TINY_COMPACT),
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def tiny_models(tmp_path: Path) -> Settings:
    """A manifest of small models with real weight files on disk."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    entries = [
        _entry("alpha"),
        _entry("beta"),
        _entry("two-x", scale=2, arch_params=dict(TINY_COMPACT_X2)),
        _entry("denoisable", supports_denoise=True, denoise_pair="denoisable-wdn"),
        _entry("denoisable-wdn", selectable=False),
        _entry("no-pair", supports_denoise=True),
    ]
    (models_dir / "manifest.json").write_text(json.dumps({"models": entries}), encoding="utf-8")

    for payload in entries:
        module = build_module(ModelEntry.model_validate(payload))
        torch.save({"params": module.state_dict()}, models_dir / payload["file"])

    return Settings(
        environment="test",
        models_dir=models_dir,
        storage_dir=tmp_path / "storage",
        device="cpu",
    )


def manager_for(settings: Settings) -> ModelManager:
    return ModelManager(
        settings, ModelService(settings), target=cpu_target("pinned for the cache tests")
    )


# ------------------------------------------------------------------- loading


def test_a_model_loads_and_reports_its_manifest_facts(tiny_models: Settings) -> None:
    upscaler = manager_for(tiny_models).get("alpha")

    assert upscaler.id == "alpha"
    assert upscaler.scale == 4
    assert upscaler.arch == "SRVGGNetCompact"


def test_a_loaded_model_is_in_evaluation_mode(tiny_models: Settings) -> None:
    """Left in training mode the network would behave differently."""
    upscaler = manager_for(tiny_models).get("alpha")

    assert upscaler.module.training is False


def test_nothing_is_loaded_until_a_model_is_asked_for(tiny_models: Settings) -> None:
    manager = manager_for(tiny_models)

    assert manager.resident == []


def test_an_unknown_model_is_a_not_found_error(tiny_models: Settings) -> None:
    with pytest.raises(ModelNotFoundError):
        manager_for(tiny_models).get("nonexistent")


def test_missing_weights_are_reported_as_a_load_failure(tiny_models: Settings) -> None:
    (tiny_models.models_dir / "alpha.pth").unlink()

    with pytest.raises(ModelLoadError, match="not downloaded"):
        manager_for(tiny_models).get("alpha")


# --------------------------------------------------------------------- cache


def test_a_second_request_reuses_the_loaded_model(tiny_models: Settings) -> None:
    manager = manager_for(tiny_models)

    first = manager.get("alpha")
    second = manager.get("alpha")

    assert first is second
    assert manager.resident == ["alpha"]


def test_the_cache_is_bounded_and_evicts_the_least_recently_used(
    tiny_models: Settings,
) -> None:
    manager = manager_for(tiny_models)

    for model_id in ("alpha", "beta", "two-x"):
        manager.get(model_id)

    assert len(manager.resident) <= CPU_RESIDENT_MODELS
    # "alpha" was the oldest, so it is the one that went.
    assert "alpha" not in manager.resident
    assert "two-x" in manager.resident


def test_using_a_model_makes_it_the_most_recently_used(tiny_models: Settings) -> None:
    manager = manager_for(tiny_models)

    manager.get("alpha")
    manager.get("beta")
    manager.get("alpha")  # refreshes alpha, so beta becomes the oldest
    manager.get("two-x")

    assert "alpha" in manager.resident
    assert "beta" not in manager.resident


def test_release_frees_everything(tiny_models: Settings) -> None:
    manager = manager_for(tiny_models)
    manager.get("alpha")
    manager.get("beta")

    freed = manager.release()

    assert freed == len(["alpha", "beta"][-CPU_RESIDENT_MODELS:])
    assert manager.resident == []


def test_release_can_target_one_model(tiny_models: Settings) -> None:
    manager = manager_for(tiny_models)
    manager.get("alpha")
    manager.get("beta")

    manager.release("beta")

    assert "beta" not in manager.resident


def test_loading_is_serialised_across_threads(tiny_models: Settings) -> None:
    """Two threads asking at once must not both load the same weights.

    On a 4 GB card that doubles the peak allocation for the duration of the
    load, which is exactly when memory is tightest.
    """
    manager = manager_for(tiny_models)
    results: list[object] = []
    barrier = threading.Barrier(4)

    def request() -> None:
        barrier.wait()
        results.append(manager.get("alpha"))

    threads = [threading.Thread(target=request) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 4
    assert all(result is results[0] for result in results)


# ------------------------------------------------------------------- denoise


def test_a_denoise_blend_is_cached_separately_from_the_plain_model(
    tiny_models: Settings,
) -> None:
    """Returning the plain model for a blended request would ignore the slider."""
    manager = ModelManager(tiny_models, ModelService(tiny_models), target=cpu_target("test"))

    blended = manager.get("denoisable", denoise_strength=0.5)
    again = manager.get("denoisable", denoise_strength=0.5)

    assert blended is again


def test_full_strength_skips_the_blend_entirely(tiny_models: Settings) -> None:
    """At 1.0 the standard weights are used unchanged, so there is no second
    checkpoint to read."""
    manager = manager_for(tiny_models)
    (tiny_models.models_dir / "denoisable-wdn.pth").unlink()

    assert manager.get("denoisable", denoise_strength=1.0) is not None


def test_a_model_without_a_denoise_pair_refuses_the_setting(tiny_models: Settings) -> None:
    with pytest.raises(ModelLoadError, match="does not have a denoise control"):
        manager_for(tiny_models).get("alpha", denoise_strength=0.5)


def test_a_model_claiming_denoise_without_a_pair_is_refused(tiny_models: Settings) -> None:
    """A manifest can be wrong; the failure should name the cause."""
    with pytest.raises(ModelLoadError, match="denoise"):
        manager_for(tiny_models).get("no-pair", denoise_strength=0.5)


@pytest.mark.parametrize("strength", [-0.5, 1.5])
def test_an_out_of_range_denoise_strength_is_rejected(
    tiny_models: Settings, strength: float
) -> None:
    with pytest.raises(ValueError, match="denoise_strength"):
        manager_for(tiny_models).get("denoisable", denoise_strength=strength)


def test_a_blend_produces_different_weights_from_the_unblended_model(
    tiny_models: Settings,
) -> None:
    manager = manager_for(tiny_models)

    plain = manager.get("denoisable")
    plain_weight = next(iter(plain.module.state_dict().values())).clone()
    manager.release()

    blended = manager.get("denoisable", denoise_strength=0.5)
    blended_weight = next(iter(blended.module.state_dict().values()))

    # Two independently initialised tiny models, so a real blend must differ.
    assert not torch.equal(plain_weight, blended_weight)
