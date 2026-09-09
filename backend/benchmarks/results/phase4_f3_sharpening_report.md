# Phase 4 · F3 — does AuraScale's sharpening improve real photographs?

**Research only. The production sharpening default was NOT changed** — it
remains `sharpenStrength: 0`. `DEFAULT_DENOISE` remains 1.0 and
`CREATIVE_DENOISE` remains 0.25. No file under `backend/app`, `frontend/src`,
`models/` or `pyproject.toml` was modified. Nothing committed, nothing pushed.
F4 and F5 were not started.

---

## 1. Objective

`unsharp_mask` is the most carefully built post-process in the codebase and
**no benchmark has ever measured it**. Every phase from 2 onward pinned
`sharpen_strength = 0.0` — correctly, to isolate whatever else it studied — so
the one detail-recovery lever the pipeline already owns is both disabled by
default and unevaluated. The Phase 4 consolidation named it the largest untested
component of the current pipeline.

The governing rule for this phase: **an unsharp mask raises high-frequency
energy by construction. That is what it does, and it is therefore not evidence
that it helped.**

## 2. Exact matrix

**5 photographs × 6 strengths = 30 cells. 30 completed, 0 failed, 0 skipped.**

| | |
|---|---|
| Model | `RealESRGAN_x4plus` |
| Strengths | **0.00, 0.15, 0.25, 0.35, 0.50, 0.75** |
| Scale | 4x, single native pass |
| Output | PNG, lossless |
| Denoise | **structurally absent** — see below |
| Target resolution | none |
| Tiling | production defaults, tile 256 / pad 16 |
| Precision | fp16, production policy |
| Baseline arm | `sh0.00` |

**Sharpening is the only variable, and denoise cannot confound it.**
`RealESRGAN_x4plus` declares no `denoise_pair`, so `_denoise_for` drops any
denoise value and no denoising stage runs at all. Denoise is not "held
constant" here — it does not exist, which is stronger.

**The neural pass ran once per photograph** and its output was cached in
memory; all six strengths were applied to that identical array. Five SR passes,
not thirty, and no arm can differ by model variance, tiling or precision.

## 3. Existing sharpening semantics

Read from `app/services/enhancement_service.py`; nothing was modified.

| | |
|---|---|
| Contract | `strength ∈ [0, 1]`, validated; out of range raises `ValidationError` |
| Amount | `strength × SHARPEN_MAX_AMOUNT` (1.5), so 0.75 → 1.125 |
| Radius | `sigma = clamp(0.75 × scale, 1.0, 6.0)` = **3.0 at 4x** |
| Detail | luma − GaussianBlur(luma, sigma), on Rec.709 luma |
| Shaping | soft threshold at ±2.0 (dead zone), then clip at ±10.0 (ceiling) |
| Application | correction added **equally to R, G and B**, so no colour fringe |
| Clipping | float clip to 0–255 before the cast, so highlights cannot wrap |
| Memory | strip-processed with a 64 MiB budget and a `ceil(4σ)+2` margin |
| Zero | `strength == 0` returns the caller's array unchanged — a true no-op |

The dead zone and ceiling are the design's two defences: the first keeps flat
regions quiet, the second bounds the rim on a hard edge. Whether they work is
one of the things this phase tests.

Existing tests (`tests/unit/test_enhancement_service.py`) already lock the
contract — range refusal, determinism, highlight clipping, local contrast
rising, and the clamped radius at 16x. This benchmark neither duplicates nor
contradicts them.

## 4. Methodology

Metrics reuse `benchmarks/metrics.py` unchanged; chroma statistics and the
shared-flat-mask correction come from the tracked `benchmarks/research_utils.py`.
The flat-block mask is taken once from the unsharpened arm and reused across
the other five — sharpening changes which blocks are flattest, so a per-arm
mask would compare different regions.

**Visual evaluation ran first and blind**, with panel order shuffled
independently per region so position carries no information. Metrics were read
afterwards.

Twelve inspection regions: the ten shared with Phase 3C, plus two added for
this phase because sharpening's characteristic failure is at strong edges and
no existing region was a maximally hard boundary —
`landscape-detail/snow-rock-edge` (dark rock silhouetted against sunlit snow)
and `low-light-noise/lamp-edge` (a string of point lamps against dark
structure). Both were checked against the sources before the sweep ran.

## 5. Runtime and memory

The sweep's own `sharpen_ms` figures are **unreliable** — they range 890–4422 ms
for identical work, because the timer competed with 40 MB PNG encodes and metric
computation. Re-measured cleanly on a 32 MP array, three runs each:

| arm | median | runs | RSS delta |
|---|---|---|---|
| sh0.00 | **0.0 ms** | 0, 0, 0 | +0 MB |
| sh0.15 | 991 ms | 991, 944, 1017 | +95 MB |
| sh0.25 | 953 ms | 1041, 910, 953 | +96 MB |
| sh0.50 | 956 ms | 956, 988, 951 | +97 MB |
| sh0.75 | 949 ms | 949, 954, 930 | +96 MB |

Cost is **flat across strength** — the work is identical regardless of amount —
and ~950 ms against a 30.5 s neural pass is about **3 % overhead**. RSS grows by
one strip-budget-plus-output, ~96 MB, independent of strength. `sh0.00` is a
genuine zero-cost no-op.

The neural pass: median 30.5 s, tile 256 on one image and 128 on four (the OOM
ladder fired), fp16, peak 731 MiB. Same for every arm of a photograph by
construction.

**Cost plays no part in the decision.**

## 6. Metric results — median delta vs `sh0.00`

| arm | hf ratio | local contrast | sobel p95 | flat noise | overshoot | PNG |
|---|---|---|---|---|---|---|
| sh0.15 | +17.3 % | +4.4 % | +7.5 % | +1.5 % | +9.2 % | +2.6 % |
| sh0.25 | +25.0 % | +6.4 % | +10.9 % | +1.5 % | +13.6 % | +3.7 % |
| sh0.35 | +39.0 % | +9.7 % | +15.7 % | +1.9 % | +21.1 % | +5.1 % |
| sh0.50 | +55.6 % | +13.5 % | +21.7 % | +2.0 % | +29.6 % | +6.7 % |
| sh0.75 | **+89.6 %** | +20.9 % | +33.5 % | +2.1 % | **+46.3 %** | +9.5 % |

Every quantity rises monotonically. **None of that is evidence of improvement** —
it is what an unsharp mask does.

> **Higher high-frequency energy does NOT imply better perceptual quality.**
> The arm with the most high-frequency energy in this sweep, `sh0.75` at
> +89.6 %, is the one the eye judged worst on three of five photographs. Every
> conclusion below is anchored to §8, not to this table.

### There is no metric knee

Overshoot gained per unit of detail gained, `Δovershoot / Δhf`:

| image | sh0.15 | sh0.25 | sh0.35 | sh0.50 | sh0.75 |
|---|---|---|---|---|---|
| foliage-texture | 0.77 | 0.78 | 0.76 | 0.75 | 0.73 |
| landscape-detail | 0.54 | 0.54 | 0.54 | 0.53 | 0.50 |
| low-light-noise | 0.53 | 0.54 | 0.53 | 0.53 | 0.51 |
| portrait-skin | 0.49 | 0.46 | 0.44 | 0.43 | 0.40 |
| text-signage | 0.49 | 0.50 | 0.51 | 0.50 | 0.48 |

**Essentially flat, and very slightly falling.** The ratio of halo to detail
does not deteriorate with strength — sharpening simply does more of everything.
So the metrics cannot identify a threshold. **The choice of default is
perceptual or it is nothing**, which is the single most useful thing this table
says.

## 7. Per-image results

`flat_noise` barely moves anywhere: +0.5 % on landscape at every strength,
+2.0 % on text at every strength, +4.5 % worst case (portrait at 0.75). The
dead zone is doing its job in flat regions. §10 shows why that is not the whole
story.

`hf_ratio` rises most on portrait-skin (+121.3 % at 0.75) simply because its
baseline is the lowest in the corpus (0.00072) — a small absolute change on a
tiny denominator.

## 8. Blind visual results

Panel order was shuffled independently per region. That decision earned its
keep immediately.

### The shuffle caught me reading position instead of pixels

On the first four regions I inspected — `snow-rock-edge`, `lamp-edge`,
`skin-cheek`, `text` — I described a smooth progression from soft at P1 to
sharpest at P6. The mappings were:

| region | P1..P6 |
|---|---|
| snow-rock-edge | 0.50, **0.00**, 0.35, 0.15, 0.25, 0.75 |
| lamp-edge | **0.00**, 0.50, 0.15, 0.25, 0.35, 0.75 |
| skin-cheek | 0.15, 0.25, **0.75**, **0.00**, 0.50, 0.35 |
| text | 0.25, 0.50, **0.00**, 0.35, 0.15, 0.75 |

I identified **sh0.75** as the extreme in three of four. Everything else in my
ranking was confabulated: on `snow-rock-edge` I called P1 (0.50) soft and P4/P5
(0.15/0.25) sharp; on `skin-cheek` I called P3 — which was 0.75, the maximum —
merely "more defined". **On edge-dominated regions I could not reliably
distinguish 0.00 from 0.15, 0.25, 0.35 or 0.50.**

### On texture the ladder *is* discriminable, and the blind read was exact

`landscape-detail/fine-texture`, P1..P6 = **0.15, 0.25, 0.35, 0.75, 0.50, 0.00**.

Written before the reveal: *"P4 ≈ P5 most defined; P6 softest; P1 lowest of the
sharpened ones; P2/P3 middle."*

- P6 softest = **sh0.00** ✓
- P4 most defined = **sh0.75** ✓, P5 = **sh0.50** ✓
- P1 = **sh0.15** ✓, P2/P3 = **0.25/0.35** ✓

The ordering was **exactly right**, blind. So the benefit is real and
perceptible — including at 0.15 — but it lives on *texture*, not on edges.

### Labelled confirmation of direction

| region | 0.00 → 0.25 | 0.25 → 0.75 |
|---|---|---|
| text-signage | slightly crisper, no artefact | **clear dark undershoot rim** below the pale band; dark area speckled |
| snow-rock-edge | marginally crisper | **bright halo** hugging the rock silhouette; internal texture over-contrasted |
| lamp-edge | slightly more defined | **dark structure visibly grainy/crunchy** |
| landscape rock | **visibly more defined texture** | over-done; slope reads as noise rather than rock |
| foliage | crisper leaf edges, more surface texture | edge rim appearing on the dark/light boundary |
| skin | subtle, natural | mottling becomes etched, not plastic |

### Viewing size decides the product question

| scale | 0.00 vs 0.25 vs 0.75 |
|---|---|
| fit-to-screen (32 MP → ~240 px panel) | **indistinguishable at every strength** |
| **1:1** | 0.25 visibly better than 0.00; 0.75 visibly over-sharpened |

For an upscaler, 1:1 is the honest viewing scale — it is where the added
resolution is consumed, and a user who upscales 4x is not going to view the
result fit-to-screen. At that scale the benefit is real. At fit-to-screen the
whole question is moot.

### Classification

| image | sh0.15 | sh0.25 | sh0.35 | sh0.50 | sh0.75 |
|---|---|---|---|---|---|
| landscape-detail | better | **better** | better | better | worse |
| foliage-texture | better | **better** | better | equivalent | worse |
| portrait-skin | equivalent | equivalent | equivalent | equivalent | worse |
| text-signage | equivalent | equivalent | equivalent | equivalent | **worse** |
| low-light-noise | equivalent | equivalent | equivalent | worse | **worse** |

## 9. Metric-vs-eye disagreements

1. **`flat_noise` says sharpening barely amplifies noise (+1.5–2.1 % median).
   The eye says the low-light dark structure gets visibly crunchy.** Both are
   right about different things — `flat_noise` samples only the *flattest*
   blocks, and the grain appears in dark *textured* areas. §10 quantifies what
   the metric structurally cannot see. **This is the phase's most important
   metric failure.**
2. **`hf_ratio` rises 17–90 % at every strength, monotonically.** It is highest
   where the picture is worst (0.75). Read alone it would have recommended the
   most damaged arm.
3. **The overshoot/detail ratio is flat and slightly *falling* with strength**,
   which would suggest 0.75 is the cleanest trade. Visually 0.75 is the only arm
   with unmistakable ringing. A ratio computed over large deltas flatters
   aggressive settings — the same trap F2 hit.
4. **I misread four of five blind regions** (§8). Blind evaluation reduces
   confirmation bias; it does not make the observer infallible, and the shuffle
   is what exposed it rather than hiding it.

## 10. Low-light noise — the special test

`flat_noise` measures the flattest blocks. Sharpening's grain cost lands in dark
*textured* areas, which those blocks exclude by construction. Measuring
high-frequency energy restricted to the darkest 35 % of pixels instead:

| region | sh0.15 | sh0.25 | sh0.35 | sh0.50 | sh0.75 |
|---|---|---|---|---|---|
| low-light / dark-noisy | +17.3 % | +21.6 % | +27.5 % | +35.7 % | **+53.6 %** |
| low-light / lamp-edge | +18.7 % | +24.2 % | +33.1 % | +43.8 % | **+64.3 %** |
| text-signage / text | +9.2 % | +10.2 % | +12.0 % | +14.0 % | +17.9 % |
| landscape / fine-texture | +14.0 % | +18.7 % | +24.7 % | +33.7 % | +50.0 % |

Against `flat_noise`'s +0.7 % to +3.1 % on the same image. **Sharpening does
amplify noise substantially in dark regions — by an order of magnitude more than
the flat-region metric reports** — and the effect is present from the lowest
strength tested.

Visually this is objectionable only at 0.50 and above on the low-light image.
At 0.25 the +21.6 % is measurable but did not read as grain in inspection.

## 11. Artifact analysis

| Artefact | Found? |
|---|---|
| Halo / bright ringing | **Yes at 0.75** — bright fringe around the rock silhouette |
| Edge undershoot | **Yes at 0.75** — dark rim below the pale sign band, visible from ~0.35 |
| Grain amplification | **Yes, from 0.15**, measurable; visible from ~0.50 in dark regions |
| Crunchy texture | Yes at 0.75 — landscape slope reads as noise |
| Artificial microcontrast | Yes at 0.75 on skin — etched rather than natural |
| Plastic skin | **No** — sharpening is the opposite failure; skin never went waxy |
| Colour fringing | **No** at any strength — the luma-only design does what it claims |
| Stair-stepping | No |
| Highlight wrap | No — the float clip holds; no arm exceeded 0–255 |
| Repeated/artificial texture | No |

The two design defences behave as documented: the luma-only correction produced
no colour fringing anywhere, and the ceiling kept even 0.75's rim bounded rather
than blown. The dead zone protects flat regions specifically — it does not, and
cannot, protect dark textured ones.

## 12. Aggregate conclusion

- Sharpening produces a **real, blind-validated increase in perceived texture
  detail**, visible from 0.15 on textured subjects at 1:1.
- Its costs are **ringing and halo at hard edges** (visible at 0.75, faint from
  ~0.35) and **dark-region grain amplification** (measurable from 0.15, visible
  from ~0.50).
- The two do not trade off smoothly in any metric — there is no knee — so the
  boundary is perceptual.
- At **fit-to-screen no strength is distinguishable**; at **1:1 the difference
  is real**.
- Cost is ~3 % of the neural pass and constant across strength.

The band where benefit is visible and harm is not is roughly **0.15–0.35**.

## 13. Recommendation

### **C — 0.25 is a defensible default**, with the caveat below.

Against the evidence:

| Criterion | 0.25 |
|---|---|
| Visible benefit | **Yes** at 1:1 on textured subjects; blind-validated ordering |
| Ringing / halo | **None visible** on the hardest edges in the corpus |
| Grain amplification | +21.6 % dark-region HF — measurable, not visually objectionable |
| Skin | equivalent to off; no etching, no plastic |
| Text | equivalent to off; no undershoot |
| Cost | ~950 ms per 32 MP, ~3 % of the pass, +96 MB |
| Metric support | +25 % hf, +6.4 % contrast, +1.5 % flat noise |

0.15 is the more cautious choice and is also defensible — it carries visible
benefit on texture with the smallest dark-region cost (+17.3 %). **0.50 and
above should not be a default**: 0.50 is visibly worse on low-light, and 0.75 is
clearly worse on three of five photographs.

### The caveat, stated plainly

This recommends a **value**, not an action. Whether sharpening should be *on* by
default is a product decision with the same shape as F2's: it changes what every
user gets, the benefit is invisible at fit-to-screen, and low-light photographs
pay a measurable grain cost for a benefit that is subject-dependent. F3 supplies
the evidence for that decision; it does not make it, and **the production
default was not changed.**

If a single number is wanted for a future integration, it is **0.25**. If the
priority is to never make any image worse, it is **0.15**.

## 14. Confidence and limitations

**Moderate-to-high** that 0.75 is too strong — ringing, halo and grain are all
visible and consistent.

**Moderate** that 0.25 is a good default. It is clearly *safe* on this corpus and
clearly *beneficial* on two of five images. Whether it beats 0.15 is not
resolved.

**Low** on anything portrait-specific — one face, one subject.

1. **Five photographs, one observer.** Every visual verdict is mine, and §8
   documents that I got four of five blind regions wrong.
2. **One region per feature** per image was inspected in depth.
3. **4x only**, on ~2 MP inputs. Sigma scales with the factor
   (`0.75 × scale`, clamped at 6.0), so 8x and 16x sharpen at a different radius
   and are **not covered** by this result.
4. **`x4plus` only.** `general-v3` produces a different detail character and may
   want a different strength.
5. **The dark-region metric is new to this phase** and has no prior calibration;
   it is reported as a relative delta only.
6. **Fit-to-screen invisibility cuts both ways** — it weakens the case for
   enabling sharpening as much as it weakens the case against 0.75.

---

## Reproducibility

| | |
|---|---|
| Date | 2026-09-10 |
| HEAD | `dd7f727` |
| Runner | `benchmarks/phase4_f3.py` |
| Shared helpers | `benchmarks/research_utils.py` (tracked) |
| Measurements | `phase4_f3_measurements.json` (42 rows: 30 sweep + 12 strips) |
| Crops | `phase4-f3-crops/` — 72 per-arm crops + 12 blind strips + attribution |
| Model | `RealESRGAN_x4plus`, no denoise pair |
| Scale / format / denoise | 4x / PNG / structurally absent |
| Sharpening | sigma 3.0, amount = strength × 1.5, dead zone ±2.0, ceiling ±10.0 |
| Tiling | production defaults, tile 256 / pad 16 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |
| Python / NumPy / OpenCV / PyTorch | 3.11.9 / 2.4.6 / 5.0.0 / 2.7.1+cu118 |

Rerun with `python -m benchmarks.phase4_f3 --sweep --strips`. The module imports
only tracked benchmark modules; a clean checkout can run it.

**The production sharpening default was NOT changed. `DEFAULT_DENOISE` remains
1.0 and `CREATIVE_DENOISE` remains 0.25.**
