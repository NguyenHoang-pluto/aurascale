"""Ownership of loaded models.

The rules this exists to enforce (docs/architecture.md § 7):

  * **Lazy** - nothing loads at startup. The first job that needs a model pays
    for it, so a CPU-only machine never pays for CUDA initialisation and the
    health endpoint answers immediately after boot.
  * **Cached** - a loaded model stays resident between jobs. Loading
    `RealESRGAN_x4plus` costs seconds; per-request loading would dominate the
    runtime of every small job.
  * **Bounded** - on a 4 GB card two 17M-parameter networks plus activations do
    not fit. The cache holds `max_resident` models (one on CUDA); loading past
    that evicts the least recently used and frees its VRAM first.
  * **Thread-safe** - the worker pool can ask for a model while another thread
    is loading one. A lock around loading stops two threads both allocating the
    same weights and doubling the VRAM spike.

Eviction order matters on a small GPU: the old module is dropped and the cache
emptied *before* the new weights are moved onto the device, so peak usage is one
model rather than two.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger
from app.inference.device import ExecutionTarget, free_vram_mb, release_cuda_memory, select_target
from app.inference.upscaler import RealEsrganUpscaler
from app.inference.weights import blend_state_dicts, build_module, load_into, load_state_dict
from app.services.model_service import ModelService, ModelStatus

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = get_logger(__name__)

# Only one large network fits alongside its activations on a 4 GB card.
CUDA_RESIDENT_MODELS = 1
# CPU inference is bounded by system RAM, which is roomier; two lets a two-pass
# 8x job keep both models without reloading between passes.
CPU_RESIDENT_MODELS = 2

# DNI blending is only meaningful within a rounding error of the endpoints.
BLEND_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class CacheKey:
    """What makes two loaded models different.

    The denoise strength is part of the identity: a DNI blend at 0.3 is a
    different network from the same weights at 0.7, and returning the wrong one
    would silently ignore the user's setting.
    """

    model_id: str
    denoise: float | None = None


class ModelManager:
    """Loads, caches and evicts upscalers."""

    def __init__(
        self,
        settings: Settings,
        models: ModelService | None = None,
        *,
        target: ExecutionTarget | None = None,
    ) -> None:
        self._settings = settings
        self._models = models or ModelService(settings)
        self._target = target
        self._cache: OrderedDict[CacheKey, RealEsrganUpscaler] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ device

    @property
    def target(self) -> ExecutionTarget:
        """Where models load. Resolved once, then reused.

        Resolution is deferred to first use rather than done in __init__ so
        constructing a manager never imports torch.
        """
        with self._lock:
            if self._target is None:
                self._target = select_target(self._settings)
                logger.info("inference target selected", extra=self._target.describe())
            return self._target

    @property
    def max_resident(self) -> int:
        return CUDA_RESIDENT_MODELS if self.target.is_cuda else CPU_RESIDENT_MODELS

    # ------------------------------------------------------------------- cache

    @property
    def resident(self) -> list[str]:
        """Ids currently in VRAM or RAM, least recently used first."""
        with self._lock:
            return [key.model_id for key in self._cache]

    def get(self, model_id: str, *, denoise_strength: float | None = None) -> RealEsrganUpscaler:
        """A ready-to-run upscaler, loading it if it is not resident.

        `denoise_strength` is accepted only by models that declare a denoise
        pair. It is the weight given to the standard checkpoint and matches
        upstream's `--denoise_strength` exactly, so the same value produces the
        same network here as it does with the reference CLI.

        The naming is a known trap worth stating plainly: 1.0 keeps the
        standard `x4v3` weights, which denoise the most, and 0.0 blends fully
        to the `wdn` counterpart, which preserves noise. Higher therefore means
        smoother - measured, not assumed, in tests/integration.
        """
        status = self._models.get(model_id)
        blend = self._resolve_blend(status, denoise_strength)
        key = CacheKey(model_id=model_id, denoise=blend)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                logger.debug("model cache hit", extra={"model": model_id})
                return cached

            upscaler = self._load(status, blend)
            self._cache[key] = upscaler
            self._evict_to_limit()
            return upscaler

    def release(self, model_id: str | None = None) -> int:
        """Drop cached models and free their memory.

        Called between jobs by the worker and on shutdown. Returns how many
        were evicted.
        """
        with self._lock:
            keys = [key for key in self._cache if model_id is None or key.model_id == model_id]
            for key in keys:
                self._drop(key)

            if keys:
                release_cuda_memory()
            return len(keys)

    # --------------------------------------------------------------- internals

    def _resolve_blend(self, status: ModelStatus, denoise_strength: float | None) -> float | None:
        """Validate the denoise request against what the model actually supports."""
        if denoise_strength is None:
            return None

        if not 0.0 <= denoise_strength <= 1.0:
            raise ValueError(f"denoise_strength must be within [0, 1], got {denoise_strength}")

        if not status.entry.supports_denoise or status.entry.denoise_pair is None:
            raise ModelLoadError(
                f"{status.entry.name} does not have a denoise control.",
                technical=f"{status.entry.id} declares no denoise_pair",
                context={"model": status.entry.id},
            )

        # 1.0 is the standard weights unchanged, so there is nothing to blend
        # and no reason to read the second checkpoint.
        if abs(denoise_strength - 1.0) < BLEND_EPSILON:
            return None

        return denoise_strength

    def _load(self, status: ModelStatus, blend: float | None) -> RealEsrganUpscaler:
        entry = status.entry
        vram_before = free_vram_mb() if self.target.is_cuda else None

        state = load_state_dict(status.path)

        if blend is not None:
            pair_id = entry.denoise_pair
            assert pair_id is not None  # guaranteed by _resolve_blend
            pair = self._models.get(pair_id)
            state = blend_state_dicts(state, load_state_dict(pair.path), blend)
            logger.info(
                "blended denoise weights",
                extra={"model": entry.id, "pair": pair_id, "alpha": blend},
            )

        module = load_into(build_module(entry), state, model_id=entry.id)
        del state

        upscaler = RealEsrganUpscaler(
            id=entry.id,
            scale=entry.scale,
            arch=entry.arch,
            module=self._to_device(module, entry.id),
            target=self.target,
        )

        logger.info(
            "model loaded",
            extra={
                "model": entry.id,
                "arch": entry.arch,
                "scale": entry.scale,
                "device": self.target.device_type.value,
                "fp16": self.target.fp16,
                "vram_free_before_mb": vram_before,
                "vram_free_after_mb": free_vram_mb() if self.target.is_cuda else None,
            },
        )
        return upscaler

    def _to_device(self, module: torch.nn.Module, model_id: str) -> torch.nn.Module:
        """Move a network onto the target device in the target precision.

        A CUDA allocation failure here is the one case where the OOM ladder
        cannot help: there is no tile size at which a model that does not fit
        will fit.
        """
        import torch

        target = self.target

        # Make room first, so peak usage is one model rather than two.
        self._evict_to_limit(headroom=1)

        try:
            module = module.to(device=target.torch_device, dtype=target.dtype())
        except torch.cuda.OutOfMemoryError as exc:
            release_cuda_memory()
            raise ModelLoadError(
                "There is not enough GPU memory to load this model. Close other "
                "GPU applications, or set DEVICE=cpu to run on the processor.",
                technical=f"{type(exc).__name__}: {str(exc)[:300]}",
                context={"model": model_id, "vramFreeMb": free_vram_mb()},
            ) from exc
        except RuntimeError as exc:
            release_cuda_memory()
            raise ModelLoadError(
                "The model could not be moved onto the selected device.",
                technical=f"{type(exc).__name__}: {str(exc)[:300]}",
                context={"model": model_id},
            ) from exc

        return module.eval()

    def _evict_to_limit(self, *, headroom: int = 0) -> None:
        """Drop least-recently-used models until the cache is within its bound."""
        limit = max(0, self.max_resident - headroom)

        while len(self._cache) > limit:
            oldest = next(iter(self._cache))
            logger.info("evicting model", extra={"model": oldest.model_id})
            self._drop(oldest)
            release_cuda_memory()

    def _drop(self, key: CacheKey) -> None:
        upscaler = self._cache.pop(key, None)
        if upscaler is None:  # pragma: no cover - defensive
            return
        # Drop the reference to the module explicitly: the allocator only
        # returns the VRAM once nothing points at the tensors.
        upscaler.module = None  # type: ignore[assignment]
