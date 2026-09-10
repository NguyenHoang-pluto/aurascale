# Phase 3C — real-world detail / restoration model research

**Status: pilot complete, nothing applied.** No production behaviour changed.
`backend/app/**`, `frontend/src/**`, `models/**`, `DEFAULT_DENOISE`,
`CREATIVE_DENOISE`, `MAX_OUTPUT_PIXELS`, scale behaviour, target-resolution
behaviour, the API and the database are all untouched. No new dependency was
installed. Nothing was committed or pushed.

**Headline: `RealESRGAN_x4plus` remains the best production choice.**
`Real_HAT_GAN_SRx4` is not visibly better on the things the product is weak at —
eyes, skin, hair — is *softer* on every detail metric, costs **5.35× the runtime
and 3.98× the VRAM**, cannot run at the production precision, and produces a
**visible tile seam** in the configuration a 4 GB card forces it into.

---

## A. Matrix

**20/20 cells completed, 0 failed, 0 skipped.**

| Arm | Model | Architecture | Denoise | Cells |
|---|---|---|---|---|
| `x4plus` *(baseline)* | `RealESRGAN_x4plus` | RRDBNet-23, 16.70 M | — | 5/5 |
| `v3-dn0` | `realesr-general-x4v3` | SRVGGNetCompact, 1.21 M | 0.0 | 5/5 |
| `v3-dn1` | `realesr-general-x4v3` | SRVGGNetCompact, 1.21 M | 1.0 | 5/5 |
| `hat` | `Real_HAT_GAN_SRx4` | HAT transformer, 20.77 M | — | 5/5 |

Corpus: the existing five photographs, unchanged, manifest-validated on every
run. Config pinned across all arms: **4x, PNG, sharpening 0.0, production
`RealEsrganUpscaler.upscale`, production tile 256 / pad 16, same inputs.**

Candidates excluded, with reasons, are in §L and in
`benchmarks/detail_models.py`.

## B. Runtime

| Arm | median | min | max | vs baseline |
|---|---|---|---|---|
| `v3-dn0` | 2.3 s | 2.1 | 5.3 | 0.06× |
| `v3-dn1` | 5.0 s | 2.1 | 6.5 | 0.14× |
| **`x4plus`** | **36.7 s** | 30.0 | 43.7 | 1.00× |
| **`hat`** | **196.7 s** | 189.6 | 213.5 | **5.35×** |

## C. VRAM

| Arm | peak | vs baseline |
|---|---|---|
| `v3-dn0` / `v3-dn1` | 43 MiB | 0.06× |
| **`x4plus`** | **731 MiB** | 1.00× |
| **`hat`** | **2907 MiB** | **3.98×** |

On a 4096 MiB card, HAT's 2907 MiB peak leaves ~1.2 GiB of headroom **with no
backend resident**. In production the FastAPI process and its CUDA context also
occupy the card.

## D. Tile size, reduction, precision

| Arm | tile | reduced | fp16 |
|---|---|---|---|
| `v3-dn0` / `v3-dn1` | 256 | 0/5 | yes |
| `x4plus` | 256 and 128 | **4/5** | yes |
| `hat` | **128 only** | **5/5** | **no — fp32 forced** |

Two findings here matter more than the numbers.

**`x4plus` itself OOM-ladders on this hardware.** Only `landscape-detail` kept
tile 256; the other four fell to 128. So the honest runtime comparison is
36.7 s against 196.7 s, not the 26 s figure the isolated probe suggested.

**HAT cannot run at the production precision.** In fp16 it returns **100 % NaN**
— all 786 432 values of a 512×512 output — while fp32 on the identical input
returns a clean image. Measured, not inferred; no tile size avoids it. HAT is
therefore being compared **at a different safety configuration from every other
arm**, which is recorded per-run as `precision_forced` and must not be glossed
over.

## E. Metrics

Median delta against `x4plus` across the five photographs.

| Metric | `v3-dn0` | `v3-dn1` | **`hat`** |
|---|---|---|---|
| high-frequency ratio | +17.6 % | −29.0 % | **−46.1 %** |
| local contrast | +2.3 % | +1.4 % | **−15.6 %** |
| sobel p95 (edge strength) | +0.6 % | −8.9 % | **−22.4 %** |
| edge overshoot (halo proxy) | −3.8 % | −19.9 % | **−33.6 %** |
| flat noise | +75.2 % | −35.8 % | −1.6 % |
| chroma sigma | +45.9 % | −19.5 % | +4.6 % |
| PNG size | +22.7 % | −17.9 % | −3.9 % |

**HAT is uniformly softer than the baseline**, not sharper: less
high-frequency energy, less local contrast, weaker edges. Its one metric
advantage is −33.6 % overshoot, i.e. fewer halos — consistent with a smoother
result rather than a more detailed one.

There is therefore **no metric case for HAT either**. This is worth stating
plainly because the usual failure mode of a benchmark like this is a model that
wins on sharpness metrics and loses on the eye. HAT loses on both.

## F. Blind visual evaluation

**Protocol.** Panels were shuffled per region with the mapping written to a
file that was *not read* until after each verdict was written. Descriptions
below were formed panel-by-position first, then mapped. Inspection at 1:1,
normal viewing size, and ~12× zoom.

A validity check on the protocol: the blind read independently reproduced a
result established in the earlier Phase 3 report — that `v3` at denoise 1.0
flattens skin pores — without that being known at the time of looking.

### Eyes — the special gate

| Panel | Blind description | Arm |
|---|---|---|
| P3 | Lashes clearly separated, fine tips, iris radial fibres visible, clean | **`hat`** |
| P4 | Lashes similarly separated, slightly crisper, but **speckle in the brow** and grainier lid skin | **`x4plus`** |
| P1 | Lashes moderately defined, somewhat merged | `v3-dn1` |
| P2 | Softest, lashes merged into a mass, iris flattest | `v3-dn0` |

Blind ranking **P4 ≈ P3 > P1 > P2**, i.e. `x4plus ≈ hat`, both ahead of `v3`.

Eye realism classification — **all four `NATURAL`**. No arm invented eyelashes,
no arm produced a synthetic-looking iris, catchlights are preserved in all
four, and no arm produced a "too perfect" eye. HAT is *cleaner*; `x4plus` is
*grainier*. Neither is more believable than the other.

**`hat` vs `x4plus` on eyes: EQUIVALENT.**

### Skin

| Panel | Blind description | Arm |
|---|---|---|
| P1 | Visible micro-texture, mottled pore structure, natural | **`hat`** |
| P3 | Similar, mottling marginally more defined | `v3-dn0` |
| P2 | Slightly softer than P1 but still textured | **`x4plus`** |
| P4 | **Noticeably flatter**, pore texture reduced, waxy | `v3-dn1` |

HAT's skin is natural and shows **no plastic-skin failure** — a genuine point
in its favour, and the failure mode Phase 3A found with NLM. `x4plus` is a
little softer here than HAT.

**`hat` vs `x4plus` on skin: SLIGHTLY BETTER for `hat`.**

### Hair

| Panel | Blind description | Arm |
|---|---|---|
| P3 | **Best strand separation**, distinct strands within the dark mass, crisp isolated hairs | **`x4plus`** |
| P4 | Good separation, close to P3, but a **blocky discontinuity** in the lower-left dark region | **`hat`** |
| P2 | Softer | `v3-dn0` |
| P1 | Softest, strands merged | `v3-dn1` |

**`hat` vs `x4plus` on hair: SLIGHTLY WORSE for `hat`** — and the blocky
artifact turned out to be the important finding (§G).

### Lips and identity

Face shape, nose shape, eye shape and mouth are **identical** between `x4plus`
and `hat` at normal viewing size. **Neither model alters identity**; neither
invents facial features. Both pass the identity gate. `x4plus` renders
under-eye wrinkles with marginally more contrast, `hat` marginally smoother.

**Identity preservation: both PASS. At normal viewing size the two are very
close and I would struggle to pick a winner.**

## G. The tile seam — HAT's disqualifying artifact

While inspecting hair I saw a rectangular tonal discontinuity in the HAT panel.
Verified three ways:

1. **Numerically.** Row-difference profile on the hair crop peaks at **y = 351
   with a ratio of 2.18** against the local median; `x4plus` peaks at 1.15. The
   column peaks (x ≈ 115, ratio ~2.5–2.9) are present in *both* arms and are a
   real image edge, not a seam.
2. **Visually at 5×.** A **perfectly straight horizontal tone step** crosses the
   frame: skin above is lighter and pinker, below darker and more saturated,
   and **hair strands are cut and offset in tone as they pass through it**. No
   natural skin feature is a perfectly horizontal line that changes tone on both
   sides while interrupting overlying hair.
3. **Position.** It coincides with the numerically detected row.

**Root cause.** HAT ran at tile 128 with the production `tile_pad` of 16. That
pad is tuned for RRDBNet's small receptive field. HAT's window attention
(window 16) plus overlapping cross-attention (overlap ratio 0.5, i.e. 24 px
windows) across six RHAG groups has a far larger effective receptive field, so
16 px of context is not enough and each tile reconstructs its border from
different information. Larger tiles would help — and do not fit in 4 GB.

**Severity.** Clearly visible at 4–5× inspection zoom; faint but present at 1:1
once you know where to look; not obvious at normal viewing size. So it is not
catastrophic — but it is a *structured, non-natural* artifact, the kind that
becomes obvious on smooth gradients such as sky and skin, and it does not occur
with `x4plus`.

## H. Foliage, text, low-light, landscape

| Region | Observation | `hat` vs `x4plus` |
|---|---|---|
| **Text/signage** | HAT marginally cleaner but lettering **less legible**; both keep the hard band edge crisp | SLIGHTLY WORSE |
| **Foliage (saturated colour)** | Near-identical; saturated greens preserved in both; **no colour bleeding, no desaturation** in either; HAT marginally softer on moss texture | EQUIVALENT |
| **Low-light / noisy** | HAT slightly smoother in dark water, `x4plus` slightly grainier | EQUIVALENT to SLIGHTLY BETTER |
| **Landscape fine texture** | `x4plus` retains visibly more granular rock texture; **HAT visibly loses it** — matches hf −57.2 % on this image | CLEARLY WORSE |

No arm produced ringing, colour halos, checkerboard patterns, repeated
textures, or GAN hallucination on any region. The only structured artifact
found in the entire pilot is HAT's tile seam.

## I. Metric-vs-eye disagreements

1. **Two automatic seam detectors missed the seam I could plainly see.** A
   gradient-based detector ranked HAT *second lowest* for seams (median 2.44 vs
   `x4plus` 2.64) and a low-frequency step detector separated nothing (every arm
   ≈31). Both were dominated by the images' own structure. Only the targeted
   per-row profile on the affected crop, plus the eye, found it. **The eye found
   the phase's most important artifact; two metrics designed to find it did
   not.**
2. **HAT's −33.6 % overshoot reads as "fewer halos", which sounds good.** It is
   really a symptom of a softer image, and pairs with −46.1 % hf and −15.6 %
   local contrast. Taken alone it would have flattered HAT.
3. **`v3-dn0`'s +17.6 % hf and +75.2 % flat noise** look like more detail; the
   eye reads its eyes and hair as the *softest* of the four. Its high-frequency
   energy is largely noise.
4. **Direction of agreement, for once:** on landscape, HAT's −57.2 % hf matched
   the visible loss of rock texture. Metrics and eye agreed here.

## J. Quality vs cost

| Arm | Runtime | VRAM | Visual verdict vs `x4plus` |
|---|---|---|---|
| `v3-dn0` | **0.06×** | **0.06×** | Worse on eyes/hair; noisier |
| `v3-dn1` | 0.14× | 0.06× | Worse; flattens skin pores |
| `hat` | **5.35×** | **3.98×** | Eyes equivalent, skin slightly better, hair slightly worse, landscape clearly worse, **plus a tile seam** |

HAT costs **5.35× the time and 3.98× the memory** of the baseline, requires a
precision the production pipeline does not use, and returns no net visual gain.

## K. Recommendation

### **REJECT** — keep `RealESRGAN_x4plus`.

Against the production-change gate:

| Question | Answer |
|---|---|
| Improvement visible at normal viewing size? | **No** — at normal size HAT and `x4plus` are very close |
| Repeatable across photographs? | **No** — better on skin, worse on hair and landscape, equivalent elsewhere |
| Improves eyes/skin/hair realistically? | **Partly** — skin slightly better, eyes equivalent, hair slightly worse |
| Preserves identity? | **Yes** — both do |
| Avoids hallucination? | **Yes** — both do |
| Avoids major artifacts? | **No** — visible tile seam at the forced tile size |
| Fits RTX 3050 4 GB reasonably? | **Marginally** — 2907 MiB peak, fp32 only, tile forced to 128 |
| Improvement large enough to justify the complexity? | **No** |

Six of eight are negative or marginal. Integrating HAT would mean vendoring a
new architecture into `app/inference/arch/`, adding a manifest entry, a
precision exception for one model, a larger `tile_pad` exception for one model,
and a 5× slower path — to obtain an image users would not identify as better.

**This is a successful research result, not a failure.** The prior Phase 3
report declined HAT on *reasoning* and flagged it as explicitly unmeasured.
Phase 3C measured it. The conclusion is the same, and it is now evidence.

**What would change the answer:** a GPU with ≥ 8 GB, so HAT could run at tile
256+ where the seam does not arise and the runtime penalty shrinks; or a HAT
variant that is fp16-stable. Neither is available on the target hardware.

**Model routing (portrait → A, landscape → B): not justified.** The differences
between arms are too small and too inconsistent to support routing, and routing
would add a classification step and a second resident model to a card that
already OOM-ladders on `x4plus`. Keep the simpler architecture.

## L. Limitations and confidence

**Confidence: moderate-to-high for the rejection, lower for any positive claim.**

1. **One portrait, one face.** Every eye/skin/hair/identity conclusion rests on a
   single subject. A second portrait — different skin tone, different lighting,
   different age — could change the skin and eye findings. This is the largest
   limitation.
2. **One region per feature.** The seam was found in the hair crop; I did not
   exhaustively scan every HAT output for further seams, so its *frequency* is
   unquantified. Its existence is certain.
3. **HAT ran at a different precision and tile size** from every other arm. This
   is unavoidable — fp16 gives NaN, and tile 256 does not fit — but it means the
   comparison is model-plus-configuration, not model alone. Part of HAT's
   softness and its seam may be the configuration rather than the network.
4. **`x4plus` also OOM-laddered** on 4/5 images, so the baseline is not running
   at its own best configuration either. Both arms are constrained by the 4 GB
   card.
5. **Community RRDBNet fine-tunes were not tested** — they are architecturally
   free (drop-in through the existing loader) and are the most likely place a
   real quality gain exists, but their licences are commonly CC BY-NC-SA or
   unstated, and I did not verify one. That is a real gap, not a settled
   question.
6. **Five photographs.** The corpus limitation carried from Phase 2.5 onward.
7. **4x only**, on ~2 MP inputs. No 8x/16x or target-resolution runs.
8. **Automatic artifact detection proved unreliable** (§I), so absence of other
   artifacts is weaker evidence than the presence of the one that was found.

---

## Reproducibility

| | |
|---|---|
| Date | 2026-09-09 (UTC) |
| Measurements | `phase3c_measurements.json` (30 rows: 20 pilot + 10 strips) |
| Runner | `benchmarks/phase3c.py` |
| Candidates | `benchmarks/detail_models.py` |
| HAT architecture | `benchmarks/arch/hat.py` — vendored, Apache-2.0, research-only |
| HAT checkpoint | `Real_HAT_GAN_SRx4.pth`, sha256 `f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7`, verified before every load |
| Crops | `phase3c-crops/` — 40 per-arm crops + 10 blind strips + attribution |
| Corpus | 5 photographs, `corpus/manifest.json` |
| Scale / format / sharpening | 4x / PNG lossless / 0.0 |
| Tiling | production defaults, tile 256 pad 16; reductions recorded per run |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, Windows 11 |
| Python / NumPy / OpenCV / PyTorch | 3.11.9 / 2.4.6 / 5.0.0 / 2.7.1+cu118 |

Rerun: `python -m benchmarks.phase3c --pilot --strips`.

### Vendored HAT — the four changes from upstream

Documented in the file header and kept minimal so it stays diffable:
the architecture-registry import and decorator dropped; `to_2tuple` and
`trunc_normal_` inlined; the single `einops.rearrange` replaced with native
tensor ops **and verified against an independent reference before any
measurement** (`verify_hat_rearrange`); and the attention mask moved to
`x.dtype` as well as `x.device`, without which fp16 raises immediately.

A fifth adaptation lives outside the architecture, in `detail_models.py`:
`_WindowPadded` reflect-pads input to a multiple of the window size and crops
the result, exactly as upstream's own `HATModel.pre_process`/`post_process`
does. Without it HAT raises on any tile whose size is not a multiple of 16,
which is most edge tiles.

### Experiments not performed

- **GFPGAN / CodeFormer / RestoreFormer** — excluded by decision. A
  face-restoration GAN synthesises a face from a learned prior rather than
  recovering the pixels that were there; it invents eyelashes and pores and
  moves identity, which the goal of faithful restoration disqualifies.
  CodeFormer and RestoreFormer additionally carry non-commercial research
  licences. GFPGAN would also have required `basicsr` + `facexlib`.
- **HAT classical / DRCT / SwinIR classical checkpoints** — trained on
  bicubic-downsampled input. On real photographs they would measure a
  degradation mismatch that would read as an architecture verdict.
- **Community RRDBNet fine-tunes** — licence not verified; see §L.5.
- **`RealESRGAN_x4plus_anime_6B`** — the earlier Phase 3 report already showed
  it restyles photographic skin into painterly strokes.
- **A dedicated large-image safety probe for HAT** — the pilot itself is the
  probe: HAT ran the full 4x pipeline on all five photographs at up to 32 MP
  output through the production tiler, hitting and correctly surviving the OOM
  ladder on every one. Nothing in this phase bypasses `MAX_OUTPUT_PIXELS`; the
  benchmark calls `RealEsrganUpscaler.upscale` directly and never submits jobs.

No number in this report was estimated or extrapolated. Every visual claim comes
from an image in `phase3c-crops/` that was actually viewed.
