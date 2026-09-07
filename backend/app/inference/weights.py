"""Reading Real-ESRGAN checkpoints and turning them into modules.

Three things happen here and nowhere else:

  * the released `.pth` files are unwrapped — they store the state dict under
    `params_ema` or `params` depending on which release produced them;
  * the manifest's `arch` and `arch_params` are turned into a real module;
  * two checkpoints of the same architecture are blended by DNI, which is how
    the denoise control is a genuine network rather than a mix of two outputs.

Weights are loaded with `strict=True`. If a vendored architecture ever drifted
from the released checkpoints, that raises here instead of silently producing a
partly-initialised network that outputs noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger
from app.services.model_service import ModelEntry

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import nn

logger = get_logger(__name__)

# Checkpoint keys the Real-ESRGAN releases use, in the order they are tried.
STATE_DICT_KEYS = ("params_ema", "params", "state_dict")

ARCH_RRDBNET = "RRDBNet"
ARCH_SRVGGNET = "SRVGGNetCompact"
SUPPORTED_ARCHS = frozenset({ARCH_RRDBNET, ARCH_SRVGGNET})


def build_module(entry: ModelEntry) -> nn.Module:
    """Instantiate the architecture a manifest entry names.

    The parameters come from the manifest rather than from code, so adding a
    variant of an existing architecture is a manifest edit.
    """
    if entry.arch == ARCH_RRDBNET:
        from app.inference.arch.rrdbnet import RRDBNet

        try:
            # The vendored architectures are excluded from mypy so they stay
            # diffable against upstream, which makes their constructors untyped.
            return RRDBNet(**entry.arch_params)  # type: ignore[no-untyped-call]
        except TypeError as exc:
            raise ModelLoadError(
                f"The manifest entry for {entry.id!r} does not match its architecture.",
                technical=f"RRDBNet(**{sorted(entry.arch_params)}): {exc}",
                context={"model": entry.id},
            ) from exc

    if entry.arch == ARCH_SRVGGNET:
        from app.inference.arch.srvggnet import SRVGGNetCompact

        try:
            return SRVGGNetCompact(**entry.arch_params)  # type: ignore[no-untyped-call]
        except TypeError as exc:
            raise ModelLoadError(
                f"The manifest entry for {entry.id!r} does not match its architecture.",
                technical=f"SRVGGNetCompact(**{sorted(entry.arch_params)}): {exc}",
                context={"model": entry.id},
            ) from exc

    raise ModelLoadError(
        f"{entry.name} uses an architecture this build does not know how to run.",
        technical=f"arch={entry.arch!r}, supported: {', '.join(sorted(SUPPORTED_ARCHS))}",
        context={"model": entry.id},
    )


def extract_state_dict(payload: Any, *, source: str) -> dict[str, torch.Tensor]:
    """Unwrap whatever the release format wrapped the weights in.

    `params_ema` is the exponential moving average of the generator weights and
    is what the ESRGAN releases ship; the v3 general models use `params`. A file
    that is already a bare state dict is accepted as-is.
    """
    if not isinstance(payload, Mapping):
        raise ModelLoadError(
            "The model weights file is not in a format this build understands.",
            technical=f"{source}: expected a mapping, got {type(payload).__name__}",
        )

    for key in STATE_DICT_KEYS:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return _strip_module_prefix(nested)

    # A bare state dict: every value is a tensor.
    if payload and all(hasattr(value, "shape") for value in payload.values()):
        return _strip_module_prefix(payload)

    raise ModelLoadError(
        "The model weights file does not contain any recognisable weights.",
        technical=f"{source}: top-level keys {sorted(str(k) for k in payload)[:8]}",
    )


def _strip_module_prefix(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Drop the `module.` prefix left by DataParallel training runs."""
    if all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return dict(state)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Read one checkpoint from disk.

    `weights_only=True` restricts unpickling to tensors and plain containers, so
    a tampered checkpoint cannot execute code on load.
    """
    import torch

    if not path.is_file():
        raise ModelLoadError(
            "The model weights are not downloaded yet.",
            technical=f"missing file {path.name}",
            context={"file": path.name},
        )

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelLoadError(
            "The model weights could not be read. The file may be incomplete; "
            "delete it and download it again.",
            technical=f"{type(exc).__name__}: {exc}",
            context={"file": path.name},
        ) from exc

    return extract_state_dict(payload, source=path.name)


def blend_state_dicts(
    primary: Mapping[str, torch.Tensor],
    secondary: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Deep Network Interpolation between two checkpoints.

        theta = alpha * primary + (1 - alpha) * secondary

    Valid only for two networks trained from the same initialisation, which
    `realesr-general-x4v3` and its `wdn` counterpart are. The result is one
    network whose weights lie between them — not a blend of two outputs, and
    not an approximation: at alpha 1 and 0 the originals come back exactly.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")

    missing = set(primary) ^ set(secondary)
    if missing:
        raise ModelLoadError(
            "The denoise weights do not match the model they pair with.",
            technical=f"parameter names differ: {sorted(missing)[:6]}",
        )

    blended: dict[str, torch.Tensor] = {}
    for key, value in primary.items():
        other = secondary[key]
        if value.shape != other.shape:
            raise ModelLoadError(
                "The denoise weights do not match the model they pair with.",
                technical=f"{key}: {tuple(value.shape)} vs {tuple(other.shape)}",
            )
        # Both operands are float tensors on CPU at this point; the arithmetic
        # is the interpolation itself, so it must not happen in half precision.
        blended[key] = value.float() * alpha + other.float() * (1.0 - alpha)

    return blended


def load_into(module: nn.Module, state: Mapping[str, torch.Tensor], *, model_id: str) -> nn.Module:
    """Load weights strictly, translating a mismatch into a clear error."""
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ModelLoadError(
            "The model weights do not match the architecture they are for. "
            "The download may be corrupt or from a different model.",
            technical=f"{type(exc).__name__}: {str(exc)[:400]}",
            context={"model": model_id},
        ) from exc

    return module
