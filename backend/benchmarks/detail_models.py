"""Phase 3C candidates: which restoration network produces believable detail.

Definitions and loading only. Nothing here is imported by `app/`, and nothing
here modifies production behaviour, defaults, the manifest or the API.

The product problem this phase addresses is narrow and specific: outputs are
larger and cleaner, but **eyes, skin, hair and fine texture still do not look
real**. That is a perceptual complaint, so this phase is a perceptual
benchmark. Metrics are recorded, and they are recorded second.

How a non-manifest model is benchmarked without touching production
-------------------------------------------------------------------

`app/inference/weights.py` builds only `RRDBNet` and `SRVGGNetCompact`, and
`models/manifest.json` is off limits, so HAT cannot be loaded the way a
production model is. But `RealEsrganUpscaler` is a plain dataclass over
`(id, scale, arch, module, target)` - `ModelManager._load` does nothing more
than construct one. So an arbitrary `nn.Module` built here can be wrapped in
the **production upscaler** and run through the **production tiling path**:
the same OOM ladder, the same fp16 policy, the same `UpscaleReport`.

That matters for fairness. A candidate measured with its own bespoke tiling
would be measured under a different safety configuration from the baseline, and
the comparison would be worthless.

Excluded candidates, and why
----------------------------

**GFPGAN / CodeFormer / RestoreFormer (face restoration).** These are the
industry-standard answer to "faces do not look real", and they are excluded on
purpose. A face-restoration GAN reconstructs a face from a learned *prior*: it
does not recover the pixels that were there, it synthesises plausible ones. It
will invent eyelashes, invent pores, and move identity. AuraScale's stated goal
is faithful restoration with identity preserved, so a model whose mechanism is
generative substitution is disqualified by the goal rather than by a
measurement. CodeFormer and RestoreFormer additionally ship under
non-commercial research licences, which would block product use regardless.

**HAT / DRCT / SwinIR classical checkpoints.** Trained on bicubic-downsampled
inputs. Real photographs carry sensor noise, compression and lens softness, and
a network trained on clean bicubic degradation handles none of it. Benchmarking
one here would measure a degradation mismatch and read as an architecture
verdict, which would be misleading. Only `Real_HAT_GAN_SRx4` - trained with
Real-ESRGAN-style degradations - is photographically comparable.

**Community RRDBNet fine-tunes** (4x-UltraSharp, Remacri, NMKD-Siax and
relatives). Architecturally these are the cheapest possible candidates: they
load through the existing `RRDBNet` builder with the same `arch_params` as
`RealESRGAN_x4plus`. They are excluded because their licences are commonly
CC BY-NC-SA or simply unstated, and a non-commercial licence blocks product
use. Including one would require verifying its licence at the source first;
that verification was not done, so no such model is benchmarked and none is
recommended.

**`RealESRGAN_x4plus_anime_6B`.** Already settled in the earlier Phase 3 model
comparison: it does not merely smooth photographic skin, it restyles it into
painterly strokes. Re-running it would re-derive a known answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"

#: The HAT checkpoint under test. Real-world GAN variant, not the classical
#: bicubic one - see the module docstring for why that distinction decides
#: whether a measurement means anything.
HAT_CHECKPOINT = CHECKPOINTS / "Real_HAT_GAN_SRx4.pth"

#: Verified after download. Recorded so a rerun can prove it used the same
#: weights, and so the third-party mirror cannot silently serve something else.
HAT_SHA256 = "f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7"

#: Derived from the checkpoint's own tensor shapes rather than copied from a
#: config file: embed_dim from `conv_first`, 6 RHAG groups of 6 blocks from the
#: key names, 6 heads and window 16 from the relative-position table (961 =
#: (2*16-1)^2), overlap 0.5 from the OCA index width (576 = 24^2), mlp_ratio 2
#: from `mlp.fc1`, compress_ratio 3 and squeeze_factor 30 from the CAB shapes.
HAT_ARCH_PARAMS: dict[str, Any] = {
    "upscale": 4,
    "in_chans": 3,
    "img_size": 64,
    "window_size": 16,
    "compress_ratio": 3,
    "squeeze_factor": 30,
    "conv_scale": 0.01,
    "overlap_ratio": 0.5,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffle",
    "resi_connection": "1conv",
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One arm, with the provenance a reviewer needs to judge it."""

    arm: str
    model_id: str
    architecture: str
    scale: int
    #: DNI blend, for the one model that has a denoise pair. None elsewhere.
    denoise: float | None
    source: str
    licence: str
    checkpoint: str
    trained_for: str
    #: Restoration-oriented or generative. The distinction this phase turns on.
    behaviour: str
    notes: str = ""
    #: True for models the production manifest already knows.
    in_manifest: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        arm="x4plus",
        model_id="RealESRGAN_x4plus",
        architecture="RRDBNet, 23 blocks, 16.70 M",
        scale=4,
        denoise=None,
        source="https://github.com/xinntao/Real-ESRGAN",
        licence="BSD-3-Clause",
        checkpoint="RealESRGAN_x4plus.pth (in manifest)",
        trained_for="real-world photographs (Real-ESRGAN degradation pipeline)",
        behaviour="restoration-oriented GAN; no face prior",
        notes="The mandated baseline and the current Standard model.",
    ),
    Candidate(
        arm="v3-dn0",
        model_id="realesr-general-x4v3",
        architecture="SRVGGNetCompact, 1.21 M",
        scale=4,
        denoise=0.0,
        source="https://github.com/xinntao/Real-ESRGAN",
        licence="BSD-3-Clause",
        checkpoint="realesr-general-x4v3.pth (in manifest)",
        trained_for="real-world photographs",
        behaviour="restoration-oriented; DNI blend toward the wdn weights",
        notes="Like-for-like reference: least-denoised state this model can produce.",
    ),
    Candidate(
        arm="v3-dn1",
        model_id="realesr-general-x4v3",
        architecture="SRVGGNetCompact, 1.21 M",
        scale=4,
        denoise=1.0,
        source="https://github.com/xinntao/Real-ESRGAN",
        licence="BSD-3-Clause",
        checkpoint="realesr-general-x4v3.pth (in manifest)",
        trained_for="real-world photographs",
        behaviour="restoration-oriented; plain x4v3 weights (no blend)",
        notes="What a user selecting Creative actually gets at DEFAULT_DENOISE.",
    ),
    Candidate(
        arm="hat",
        model_id="Real_HAT_GAN_SRx4",
        architecture="HAT (Hybrid Attention Transformer), 20.77 M",
        scale=4,
        denoise=None,
        source="https://github.com/XPixelGroup/HAT",
        licence="Apache-2.0",
        checkpoint=f"Real_HAT_GAN_SRx4.pth, sha256 {HAT_SHA256}",
        trained_for="real-world photographs (GAN variant, not the bicubic classical one)",
        behaviour="restoration-oriented GAN; no face prior",
        notes=(
            "Architecture vendored to benchmarks/arch/hat.py, research only. "
            "Weights obtained from a third-party HuggingFace mirror because the "
            "official release is Google Drive only; the sha256 above was "
            "verified against an independent listing after download."
        ),
        in_manifest=False,
    ),
)


def candidate(arm: str) -> Candidate:
    for entry in CANDIDATES:
        if entry.arm == arm:
            return entry
    raise KeyError(f"no such arm: {arm}")


# --------------------------------------------------------------- verification


def verify_hat_rearrange() -> None:
    """Check the one substitution in the vendored architecture that could bite.

    `benchmarks/arch/hat.py` replaces a single `einops.rearrange` with native
    tensor ops. einops is not installed, so the check is against an independent
    reference written directly from the pattern string rather than against
    einops itself - which is weaker, and is stated as such, but does catch a
    transposed axis, which is the realistic failure.

    Raises rather than returns, and is called before any measurement, so a
    silent mis-shuffle cannot be reported as a model's visual character.
    """
    import torch

    b, nc, ch, ow, nw = 2, 2, 5, 3, 7
    packed = torch.arange(b * nc * ch * ow * ow * nw, dtype=torch.float64).reshape(
        b, nc * ch * ow * ow, nw
    )

    # Reference: read the pattern 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch'
    # literally, one index at a time.
    reference = torch.empty(nc, b * nw, ow * ow, ch, dtype=torch.float64)
    for bi in range(b):
        for ci in range(nc):
            for hi in range(ch):
                for yi in range(ow):
                    for xi in range(ow):
                        for wi in range(nw):
                            flat = ((ci * ch + hi) * ow + yi) * ow + xi
                            reference[ci, bi * nw + wi, yi * ow + xi, hi] = packed[bi, flat, wi]

    # What the vendored file does.
    got = packed.view(b, 2, ch, ow * ow, nw).permute(1, 0, 4, 3, 2).reshape(2, b * nw, ow * ow, ch)

    if not torch.equal(got, reference):
        raise AssertionError(
            "the vendored HAT rearrange substitution does not match the reference; "
            "benchmarks/arch/hat.py OCAB.forward is wrong and no measurement is valid"
        )


def verify_hat_checkpoint() -> str:
    """Confirm the weights on disk are the ones this phase claims to test."""
    import hashlib

    if not HAT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"HAT checkpoint not present at {HAT_CHECKPOINT}")

    digest = hashlib.sha256(HAT_CHECKPOINT.read_bytes()).hexdigest()
    if digest != HAT_SHA256:
        raise ValueError(f"checkpoint sha256 mismatch: got {digest}, expected {HAT_SHA256}")
    return digest


# -------------------------------------------------------------------- loading


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A candidate wrapped in the production upscaler, ready to run."""

    arm: str
    upscaler: Any
    params_millions: float
    vram_after_load_mb: float | None
    extras: dict[str, Any] = field(default_factory=dict)


def _make_window_padded(module: Any, window_size: int, scale: int) -> Any:
    """Wrap HAT so it accepts tiles whose size is not a multiple of its window.

    HAT's `forward` has no `check_image_size`: it assumes the input is already a
    multiple of `window_size` (16 here) and raises a reshape error otherwise.
    The production tiler emits edge tiles of whatever is left over - 1307 wide
    at tile 256 leaves a 27 px column - so most images fail on their first edge
    tile.

    Upstream handles this outside the network, in `HATModel.pre_process` /
    `post_process`: reflect-pad the bottom and right up to the next multiple,
    run, then crop `mod_pad * scale` off the result. This reproduces that
    exactly, so HAT is run the way its authors run it rather than in a
    configuration they never intended.

    A real `nn.Module` holding HAT as a submodule, not a proxy object: the
    upscaler moves, evals and inspects what it is given, and a proxy would have
    to fake all of that. Built inside a function because importing torch at
    module import time is what the rest of this package carefully avoids.
    """
    import torch.nn as nn
    import torch.nn.functional as functional

    class WindowPadded(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = module
            self.window = window_size
            self.scale = scale

        def forward(self, x: Any) -> Any:
            _, _, height, width = x.shape
            pad_h = (self.window - height % self.window) % self.window
            pad_w = (self.window - width % self.window) % self.window

            if pad_h or pad_w:
                x = functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

            out = self.inner(x)

            if pad_h or pad_w:
                out = out[
                    :,
                    :,
                    : out.shape[2] - pad_h * self.scale,
                    : out.shape[3] - pad_w * self.scale,
                ]
            return out

    return WindowPadded()


#: HAT must run in fp32. Measured, not assumed: in fp16 it returns **100 %
#: NaN** - all 786 432 values of a 512x512 output - while fp32 on the identical
#: input returns a clean image. The overflow is in the attention path and no
#: tile size avoids it.
#:
#: This is a real asymmetry and the report must not bury it. Production policy
#: is fp16 and the Real-ESRGAN baselines are measured there, so HAT is being
#: compared at a *different safety configuration*: roughly double the activation
#: memory of its own fp16 path, and 2 590 MiB peak at the production tile of 256
#: against x4plus's measured 584 MiB.
HAT_REQUIRES_FP32 = True


def load_hat(target: Any) -> LoadedModel:
    """Build HAT, load the checkpoint strictly, wrap it in the production upscaler.

    `strict=True` on purpose: a missing or unexpected key would mean the
    architecture parameters inferred from the checkpoint are wrong, and a model
    that quietly loads 90 % of its weights produces plausible-looking garbage
    rather than an error.

    The target is forced to fp32 - see `HAT_REQUIRES_FP32`. The returned model
    records the substitution so no measurement can silently claim parity with
    the fp16 baselines.
    """
    import dataclasses

    import torch

    from app.inference.device import free_vram_mb
    from app.inference.upscaler import RealEsrganUpscaler
    from benchmarks.arch.hat import HAT

    verify_hat_rearrange()
    verify_hat_checkpoint()

    hat_target = dataclasses.replace(target, fp16=False) if target.fp16 else target

    module = HAT(**HAT_ARCH_PARAMS)  # type: ignore[no-untyped-call]
    payload = torch.load(HAT_CHECKPOINT, map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload))
    module.load_state_dict(state, strict=True)
    module.eval()

    params = sum(p.numel() for p in module.parameters()) / 1e6

    module = module.to(device=hat_target.torch_device, dtype=hat_target.dtype())
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    upscaler = RealEsrganUpscaler(
        id="Real_HAT_GAN_SRx4",
        scale=4,
        arch="HAT",
        module=_make_window_padded(module, HAT_ARCH_PARAMS["window_size"], 4),
        target=hat_target,
    )

    return LoadedModel(
        arm="hat",
        upscaler=upscaler,
        params_millions=params,
        vram_after_load_mb=(
            None if not hat_target.is_cuda else float(torch.cuda.memory_allocated()) / 1024 / 1024
        ),
        extras={
            "free_vram_after_load_mb": free_vram_mb() if hat_target.is_cuda else None,
            "fp16_requested": target.fp16,
            "fp16_used": hat_target.fp16,
            "precision_forced": target.fp16 != hat_target.fp16,
            "precision_reason": "fp16 yields 100% NaN output; measured, not assumed",
        },
    )
