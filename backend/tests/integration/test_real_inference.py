"""Real Real-ESRGAN weights, on real images.

Nothing here is stubbed: these load the official checkpoints from the models
directory and run them. They are marked `slow` so they can be selected or
skipped, and they skip rather than fail when the weights are not downloaded, so
a fresh clone still has a green test run.

    .venv/Scripts/python -m pytest                  # everything, these included
    .venv/Scripts/python -m pytest -m slow          # only these
    .venv/Scripts/python -m pytest -m "not slow"    # the fast suite

What they establish, and what nothing else can:

  * the vendored architectures match the released checkpoints exactly, because
    `strict=True` loading would raise otherwise;
  * a real image comes back larger, and looks like the input rather than noise;
  * tiling a real 23-block network produces the same picture as one pass, which
    is the seam guarantee on the actual model rather than a stand-in;
  * DNI blending changes the network in the direction it claims to.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.inference.model_manager import ModelManager
from app.services.enhancement_service import EnhancementRequest, EnhancementService
from app.services.model_service import ModelService

pytestmark = [pytest.mark.slow, pytest.mark.integration]

ALL_MODELS = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-general-x4v3",
]
# The smallest real model, used where the test is about the pipeline rather
# than about a particular set of weights.
FAST_MODEL = "realesr-general-x4v3"


def require(model_id: str) -> None:
    """Skip when the weights are not downloaded."""
    status = ModelService(get_settings()).get(model_id)
    if not status.downloaded:
        pytest.skip(f"{model_id} is not downloaded; run scripts/download_models.py")


@pytest.fixture
def settings() -> Settings:
    """Real models directory, small tiles so tiling actually happens."""
    return Settings(environment="test", tile_size=64, tile_pad=16)


@pytest.fixture
def photo() -> np.ndarray[Any, Any]:
    """A synthetic photograph: gradients, hard edges and grain.

    Flat colour would let a broken model look correct, and real photographs
    cannot be committed to the repository.
    """
    generator = np.random.default_rng(4)
    height, width = 96, 128
    rows, columns = np.mgrid[0:height, 0:width]

    picture = np.dstack(
        [
            columns / width * 255,
            rows / height * 255,
            np.full((height, width), 150.0),
        ]
    )
    picture[30:70, 40:90] = 235
    picture[40:55, 50:80] = 20
    picture += generator.normal(0, 6, picture.shape)

    return np.clip(picture, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_every_released_checkpoint_loads_into_its_vendored_architecture(
    settings: Settings, model_id: str
) -> None:
    """`strict=True` means this fails loudly if a layer drifted from upstream."""
    require(model_id)
    manager = ModelManager(settings)

    upscaler = manager.get(model_id)

    assert upscaler.id == model_id
    assert upscaler.module is not None
    manager.release()


@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_a_real_image_through_a_real_model(
    settings: Settings, photo: np.ndarray[Any, Any], model_id: str
) -> None:
    require(model_id)
    service = EnhancementService(settings)
    native = ModelService(settings).get(model_id).entry.scale
    progress: list[float] = []

    result = service.enhance(
        photo, EnhancementRequest(model_id, native), on_progress=progress.append
    )

    height, width = photo.shape[:2]
    assert result.image.shape == (height * native, width * native, 3)
    assert result.image.dtype == np.uint8

    # Progress is measured per tile, and this image is larger than one tile.
    assert len(progress) > 1
    assert progress[-1] == pytest.approx(1.0)

    # The output must resemble the input rather than being noise or a blank:
    # downsampling it back should land near the original.
    import cv2

    shrunk = cv2.resize(result.image, (width, height), interpolation=cv2.INTER_AREA)
    difference = np.abs(shrunk.astype(int) - photo.astype(int)).mean()
    assert difference < 12, f"result does not resemble the input (mean |diff| {difference:.1f})"

    service.release()


def test_tiling_a_real_model_produces_the_same_picture_as_one_pass(
    photo: np.ndarray[Any, Any],
) -> None:
    """The seam guarantee on real weights.

    Exact equality is not available here - fp16 and different tile groupings
    round differently - but a visible seam would be worth far more than a few
    levels, so a tight bound on the mean is the honest assertion.
    """
    require(FAST_MODEL)
    request = EnhancementRequest(FAST_MODEL, 4)

    tiled_service = EnhancementService(Settings(environment="test", tile_size=64, tile_pad=16))
    tiled = tiled_service.enhance(photo, request)
    assert tiled.reports[-1].tiles > 1
    tiled_service.release()

    whole_service = EnhancementService(Settings(environment="test", tile_size=0, tile_pad=0))
    whole = whole_service.enhance(photo, request)
    assert whole.reports[-1].tiles == 1
    whole_service.release()

    difference = np.abs(tiled.image.astype(int) - whole.image.astype(int))
    assert difference.mean() < 0.5
    assert difference.max() <= 8


def test_eight_times_runs_two_real_passes(settings: Settings, photo: np.ndarray[Any, Any]) -> None:
    require("RealESRGAN_x4plus")
    require("RealESRGAN_x2plus")
    service = EnhancementService(settings)
    small = photo[:48, :64]

    result = service.enhance(small, EnhancementRequest("RealESRGAN_x4plus", 8))

    assert result.image.shape == (48 * 8, 64 * 8, 3)
    assert result.passes == ["RealESRGAN_x4plus", "RealESRGAN_x2plus"]
    service.release()


def test_denoise_blending_changes_the_network_and_the_result(
    photo: np.ndarray[Any, Any],
) -> None:
    """DNI is real interpolation, and higher strength really is smoother.

    The direction is worth asserting rather than assuming: the naming is
    counter-intuitive, since the `wdn` counterpart is the one that preserves
    noise (see ModelManager.get).
    """
    require(FAST_MODEL)
    require("realesr-general-wdn-x4v3")

    generator = np.random.default_rng(9)
    noisy = np.clip(photo.astype(int) + generator.normal(0, 25, photo.shape), 0, 255).astype(
        np.uint8
    )[:64, :64]

    service = EnhancementService(Settings(environment="test", tile_size=0, tile_pad=0))

    def roughness(strength: float) -> float:
        result = service.enhance(
            noisy, EnhancementRequest(FAST_MODEL, 4, denoise_strength=strength)
        )
        # Mean absolute vertical gradient: noise raises it, smoothing lowers it.
        return float(np.abs(np.diff(result.image[..., 0].astype(int), axis=0)).mean())

    strongest = roughness(1.0)
    middle = roughness(0.5)
    weakest = roughness(0.0)

    assert strongest < middle < weakest
    service.release()


def test_alpha_survives_a_real_pass_without_going_through_the_model(
    settings: Settings, photo: np.ndarray[Any, Any]
) -> None:
    require(FAST_MODEL)
    service = EnhancementService(settings)
    alpha = np.full(photo.shape[:2], 180, dtype=np.uint8)
    rgba = np.dstack([photo, alpha])

    result = service.enhance(rgba, EnhancementRequest(FAST_MODEL, 4))

    assert result.image.shape[2] == 4
    # A flat mask stays flat. A model-generated alpha would not.
    assert set(np.unique(result.image[..., 3])) == {180}
    service.release()


def test_the_cache_holds_one_model_on_a_gpu_and_frees_the_previous_one(
    settings: Settings,
) -> None:
    """Two 17M-parameter networks do not fit alongside their activations."""
    require("RealESRGAN_x4plus")
    require(FAST_MODEL)
    manager = ModelManager(settings)

    manager.get("RealESRGAN_x4plus")
    manager.get(FAST_MODEL)

    assert len(manager.resident) <= manager.max_resident
    assert FAST_MODEL in manager.resident
    manager.release()
    assert manager.resident == []


@pytest.mark.gpu
def test_repeated_jobs_do_not_leak_gpu_memory(
    settings: Settings, photo: np.ndarray[Any, Any]
) -> None:
    """A leak here is invisible in one job and fatal by the fourth."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    require(FAST_MODEL)

    service = EnhancementService(settings)
    request = EnhancementRequest(FAST_MODEL, 4)

    def used_mb() -> int:
        free, total = torch.cuda.mem_get_info()
        return int((total - free) // (1024 * 1024))

    service.enhance(photo, request)  # first run pays for the model and the cache
    settled = used_mb()

    for _ in range(3):
        service.enhance(photo, request)

    assert used_mb() <= settled

    service.release()
    torch.cuda.empty_cache()
    assert used_mb() <= settled
