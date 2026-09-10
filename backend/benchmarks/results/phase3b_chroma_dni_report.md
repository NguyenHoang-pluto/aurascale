# Phase 3B — DNI × chroma prefilter interaction

**Status: research complete, nothing applied.** No production behaviour changed.
`DEFAULT_DENOISE`, `CREATIVE_DENOISE`, `default_model`, `MAX_OUTPUT_PIXELS`,
sharpening, 2x/4x/8x/16x, the 2K–16K targets, the model manifest and the API
contracts are all untouched. No dependency added.

**Headline: the Phase 3A result does not survive super-resolution, and where it
is visible at all it makes the output worse.** Chroma prefiltering removed
7.5–37.8 % of chroma noise measured *on the input*. Measured on the 4x *output*
across 110 cells, the chroma-noise benefit is 0.4 % in the wrong direction, and
on the one photograph where the difference is visible to the eye the filtered
result is **grainier**, not cleaner.

---

## 1. Objective

Phase 3A recommended chroma-only prefiltering as its single promising result
but explicitly never ran it together with DNI. This phase closes that gap and
answers three questions:

1. does chroma prefiltering add anything on top of DNI (C vs A)?
2. does the benefit change as DNI strengthens?
3. does it behave differently between Standard's model and Creative's?

## 2. Two structural facts that shape everything

Both were verified in code and empirically, not assumed.

**`RealESRGAN_x4plus` has no `denoise_pair`.** `_denoise_for` drops any denoise
value handed to it and `_resolve_blend` would refuse one. Standard therefore
has **no DNI at all**, so arms A (DNI only) and C (chroma + DNI) do not exist
for it. The four-arm comparison is only measurable on `realesr-general-x4v3`.

**For v3, "no DNI" is not the un-denoised state.** `_resolve_blend` returns
`None` — no blend, plain x4v3 weights — for `denoise_strength=None` *and* for
`1.0`. Verified empirically: the two outputs are **byte-identical** (max abs
diff 0). Since Phase 2.5 established that higher values denoise *harder*, the
plain weights are the strongest-denoising state. The least-denoised output v3
can make is `dni=0.00` (fully wdn weights), and that is what **D** means here.
Calling `None` "no denoise" would have made the baseline the most-denoised arm
and inverted every conclusion in this report.

## 3. Interaction matrix

110 cells, all completed.

| Config | Chroma rungs | Cells |
|---|---|---|
| v3, dni 0.00 (**D reference**) | off, weak, medium, strong | 20 |
| v3, dni 0.25 (`CREATIVE_DENOISE`) | off, weak, medium, strong | 20 |
| v3, dni 0.50 | off, weak, medium, strong | 20 |
| v3, dni 0.75 | off, weak, medium, strong | 20 |
| v3, dni 1.00 (`DEFAULT_DENOISE`) | off, weak, medium, strong | 20 |
| x4plus, no DNI | off, medium | 10 |

The four named arms, on v3:

| Arm | Chroma | DNI |
|---|---|---|
| **D** | off | 0.00 |
| **A** | off | 0.25 |
| **B** | chroma-medium | 0.00 |
| **C** | chroma-medium | 0.25 |

**Scope reduction, stated plainly.** x4plus was run at `off` and
`chroma-medium` only, not the full ladder. Two measured reasons: its cells cost
26–54 s against v3's ~1.8 s, and it hits the OOM ladder on a 4 GB card (§11),
which makes its runtime figures unusable for comparison anyway. It has no DNI,
so it takes no part in the interaction; `off` vs the candidate rung answers the
only question it can answer. The full ladder is measured on v3.

## 4. Methodology fix carried from Phase 3A

Phase 3A recomputed the flat-block mask on each filtered image. Filters that
change luma change which blocks are flattest, so before and after sampled
*different blocks*, and Phase 3A had to withhold its NLM and bilateral chroma
figures as confounded.

Here the mask is computed **once from each image's `chroma=off` output** and
the identical block indices are reused for all four chroma arms of that image.
Every chroma comparison below is over the same regions. Chroma statistics are
sampled over the same `metrics.tile_boxes` the luma metrics use, so both
families read identical pixels.

## 5. Dataset

The Phase 3A corpus, unchanged, re-validated through `load_manifest()` on every
run: landscape-detail (1732×1154), portrait-skin (1307×1530), text-signage
(1632×1224), foliage-texture (1224×1632), low-light-noise (1892×1057). Centre
crops, never resampled. Licences in `corpus/manifest.json` and
`phase3b-crops/ATTRIBUTION.md`.

## 6. Metrics

Luma metrics come from `benchmarks/metrics.py`, unchanged. Chroma metrics are
new in this phase and built from `prefilter`'s primitives:

| Metric | Reads as |
|---|---|
| `sigma_luma` | luma noise in flat blocks — the control; a chroma-only filter must not move it |
| `sigma_chroma` | chroma noise in flat blocks — what the filter is meant to lower |
| `chroma_gradient` | Sobel on Cr/Cb — colour-boundary definition. **Falls both when chroma noise is removed and when colour edges are smeared; it cannot tell them apart.** |
| `high_frequency_ratio`, `local_contrast`, `sobel_p95`, `flat_noise`, `edge_overshoot` | as in earlier phases |
| `png_bytes` | lossless encoded size — an independent read on surviving detail |

## 7. Quantitative results

### 7.1 The interaction — chroma-medium's effect at each DNI rung (v3, median of 5)

| DNI | Δ chroma sigma | Δ luma sigma | Δ chroma gradient | Δ hf ratio |
|---|---|---|---|---|
| 0.00 | −0.7 % | +1.5 % | −4.3 % | +1.6 % |
| 0.25 | +0.4 % | +0.3 % | −4.4 % | +2.5 % |
| 0.50 | −0.1 % | +0.5 % | −4.2 % | +3.3 % |
| 0.75 | −0.6 % | +0.2 % | −4.8 % | +3.8 % |
| 1.00 | +0.4 % | +1.0 % | −6.0 % | +6.8 % |

**No interaction.** The chroma effect is flat across the entire DNI range —
it does not shrink as DNI strengthens, which is what an overlap would look
like, and it does not grow. It is simply ~0 everywhere.

The chroma-noise benefit the filter exists to deliver is **within ±0.7 % and
not consistently signed**. What *is* consistent is `chroma_gradient` falling
4.2–6.0 % at every rung — a measurable cost in colour-boundary definition with
no matching benefit.

### 7.2 D / A / B / C (v3, median of 5 photographs)

| Metric | C vs A | B vs D | A vs D |
|---|---|---|---|
| chroma sigma | **+0.4 %** | −0.7 % | **−4.6 %** |
| luma sigma | +0.3 % | +1.5 % | +19.9 % |
| chroma gradient | **−4.4 %** | −4.3 % | −1.0 % |
| hf ratio | +2.5 % | +1.6 % | −14.1 % |
| local contrast | +0.1 % | +0.1 % | −0.1 % |

**DNI alone (A vs D) reduces chroma noise by 4.6 %. The dedicated chroma
prefilter reduces it by 0.7 % (B vs D) and makes it 0.4 % *worse* on top of DNI
(C vs A).** DNI — a luma-oriented weight blend that was never designed to touch
colour — is roughly six times more effective at reducing post-SR chroma noise
than the filter built specifically for the job.

### 7.3 Per-image chroma gradient — where the cost concentrates

| Image | C vs A | B vs D |
|---|---|---|
| low-light-noise | **−18.1 %** | **−17.7 %** |
| landscape-detail | −4.9 % | −5.4 % |
| foliage-texture | −4.4 % | −4.3 % |
| portrait-skin | −3.1 % | −3.1 % |
| text-signage | −0.8 % | −0.8 % |

The image with the most chroma noise loses the most colour-boundary definition.
`chroma_gradient` alone cannot say whether that is noise removed or edges
smeared — §8 answers it by looking.

### 7.4 Where the difference actually lands — luma, not chroma

Per-pixel difference between arms, decomposed by channel:

| Image | pair | mean ΔY | mean ΔCr | mean ΔCb | p99 ΔY | p99 ΔCr | p99 ΔCb |
|---|---|---|---|---|---|---|---|
| low-light-noise | D−B | **2.094** | 1.025 | 1.440 | **12** | 6 | 9 |
| low-light-noise | A−C | **1.733** | 0.980 | 1.398 | **11** | 6 | 9 |
| landscape-detail | D−B | 0.621 | 0.218 | 0.217 | 6 | 2 | 2 |
| text-signage | D−B | 0.652 | 0.319 | 0.328 | 5 | 2 | 2 |
| foliage-texture | D−B | 0.563 | 0.329 | 0.448 | 3 | 2 | 4 |
| portrait-skin | D−B | 0.450 | 0.439 | 0.357 | 2 | 2 | 2 |

**A chroma-only prefilter produces an output difference that is larger in luma
than in chroma.** The filter never touches Y; the change arrives because the
network reconstructs differently from a chroma-altered input. This is the
mechanism behind the whole result: the prefilter is not delivering cleaner
colour downstream, it is perturbing the reconstruction.

### 7.5 Difference magnitude — how much changes at all

| Image | pair | mean abs | % pixels >2/255 | % >8/255 |
|---|---|---|---|---|
| low-light-noise | D−B | 2.567 | **56.10 %** | **11.02 %** |
| text-signage | D−B | 0.727 | 10.54 % | 0.52 % |
| foliage-texture | D−B | 0.713 | 10.19 % | 0.55 % |
| landscape-detail | D−B | 0.655 | 9.45 % | 0.75 % |
| portrait-skin | D−B | 0.618 | 5.83 % | **0.00 %** |

Only low-light-noise changes materially. On portrait-skin no pixel differs by
more than 9/255 and none by more than 8/255 — the filter is very nearly a
no-op there.

### 7.6 Runtime

| Arm | filter, median | min | max |
|---|---|---|---|
| chroma-weak | 14.0 ms | 12.0 | 98.8 |
| chroma-medium | **25.2 ms** | 21.3 | 347.6 |
| chroma-strong | 33.2 ms | 28.7 | 78.1 |

Against a v3 SR pass of ~1.8 s that is **~1.4 % overhead**; against an x4plus
pass of 26–54 s it is under 0.1 %. Cost is not the problem with this filter.

## 8. Human visual evaluation

Metrics said "equivalent" with a consistent `chroma_gradient` cost that could
mean either noise removal or edge smearing. Only looking could settle it. Phase
3A's decisive finding — NLM rendering skin plastic — was invisible to every
metric, so this pass was treated as the arbiter, not a formality.

**Method.** Crops are 4x SR output, identical coordinates across arms,
nearest-neighbour zoom so nothing is resampled between model and eye. For each
photograph the largest-difference 32 px block was located numerically and the
crop centred there — so every judgement below is of the *strongest case the
filter has*, not a random region. Differences were described by panel position
before being mapped back to arms. Both a normal view (1:1 on the 4x output) and
an inspection zoom (≈12x effective) were used.

### 8.1 low-light-noise — the target case, and the only visible difference

At ≈12x on the peak-difference region, the two panels are clearly different.
One shows lamp highlights that are **softer and more blended**, an amorphous
bright cluster, and a smooth dark background. The other shows the **same
highlights crisper and more separated** — the star-shaped lamp resolves into
distinct points, the cluster breaks into individual bright spots — but the dark
foliage and background carry **visibly more grain and speckle**.

Mapping back: the softer, cleaner panel is **D** (baseline). The crisper,
grainier panel is **B** (chroma prefilter).

At **normal viewing** (1:1 on the 4x output) the difference survives, and the
reading is unambiguous: **D looks cleaner; B looks noisier.** The extra grain in
the dark regions is more objectionable than the marginally crisper highlights
are attractive.

The A vs C pair shows the **same pattern in the same direction**: A (DNI only)
smoother and cleaner, C (chroma + DNI) grainier. The effect is therefore
repeatable across both DNI conditions on this image.

This also explains §7.1's rising `hf_ratio` (+10.1 % B vs D, +13.8 % C vs A on
this image). Read as a number it looks like recovered detail. Looked at, it is
**grain**. That is precisely the trap the visual pass exists to catch, and here
the metric and the eye disagree about the sign of the same measurement.

### 8.2 portrait-skin — indistinguishable, and no plastic skin

At ≈12x on the peak-difference region the two panels are **indistinguishable**.
Identical skin texture, identical subtle mottling, identical colour. No
desaturation, no colour shift, no loss of natural variation, no waxiness.

Worth recording explicitly: the chroma filter does **not** reproduce Phase 3A's
NLM plastic-skin failure. It is harmless here — and equally, does nothing.

### 8.3 foliage-texture — indistinguishable, saturated colour intact

A dark stem crossing bright saturated green, the colour-boundary stress case.
**Indistinguishable.** Same green saturation, same stem-edge definition, no
colour bleeding across the boundary, no desaturation, no artificial uniformity
in the leaves.

### 8.4 text-signage — indistinguishable, no fringing

A hard high-contrast edge. **Indistinguishable.** The edge is equally crisp in
both; a faint warm tint at the transition is present *identically* in both, so
no colour fringing is introduced and no colour edge definition is lost. The
−0.8 % `chroma_gradient` on this image is not visible.

### 8.5 landscape-detail — fine texture, essentially indistinguishable

Snow edge against dark rock. Essentially **indistinguishable**, with a faint
hint of the same "filtered version slightly grainier" direction seen on
low-light, far weaker and at the limit of perception even at 12x.

### 8.6 Visual matrix — D vs B

| Image | Visible winner | Magnitude | Normal-view noticeable? | Main reason |
|---|---|---|---|---|
| low-light-noise | **D** | Clearly worse for B | **Yes** | B visibly grainier in dark regions |
| landscape-detail | D (marginal) | Slight | No | Faint extra grain in rock |
| portrait-skin | — | Indistinguishable | No | No visible difference at all |
| foliage-texture | — | Indistinguishable | No | Colour boundaries unchanged |
| text-signage | — | Indistinguishable | No | Edge and fringing unchanged |

### 8.7 Visual matrix — A vs C

| Image | Visible winner | Magnitude | Normal-view noticeable? | Main reason |
|---|---|---|---|---|
| low-light-noise | **A** | Clearly worse for C | **Yes** | C visibly grainier, same as B vs D |
| landscape-detail | A (marginal) | Slight | No | Same faint direction |
| portrait-skin | — | Indistinguishable | No | Difference below 9/255 everywhere |
| foliage-texture | — | Indistinguishable | No | No colour bleed either way |
| text-signage | — | Indistinguishable | No | No fringing either way |

### 8.8 The critical question

*"If a normal AuraScale user uploads a noisy photograph, would enabling chroma
prefilter produce a visibly better final upscaled image?"*

**NO — visually worse.**

On the noisiest photograph in the corpus — the exact case the filter was
proposed for — the filtered output is visibly grainier at normal viewing size.
On the other four photographs it is indistinguishable. There is no image class
in this corpus where it produces a visible improvement.

## 9. Model interaction

**`realesr-general-x4v3` (Creative).** The full four-arm comparison exists here.
Chroma-medium is numerically flat against `off` at every DNI rung (±0.7 %
chroma sigma), costs 4.2–6.0 % of chroma gradient, and is visually neutral on
four photographs and visibly harmful on the fifth. DNI alone outperforms it by
about 6× on the metric the filter targets.

**`RealESRGAN_x4plus` (Standard).** It has **no DNI**, so it did not and cannot
participate in the interaction — any claim about DNI × chroma on x4plus would
be fabricated. The only question it can answer is whether chroma prefiltering
helps on its own:

| Image | Δ chroma sigma | Δ chroma gradient |
|---|---|---|
| landscape-detail | **+2.2 %** | −9.0 % |
| foliage-texture | +0.4 % | −5.1 % |
| low-light-noise | +0.0 % | −19.1 % |
| portrait-skin | −1.5 % | −2.7 % |
| text-signage | −2.4 % | −0.8 % |
| **median** | **0.0 %** | **−5.1 %** |

**It does not help x4plus.** Median chroma-noise change is exactly zero, in
exchange for a 5.1 % median loss of colour-boundary definition and −19.1 % on
the low-light image. The pattern matches v3's: no benefit, a consistent cost.

There is no model for which chroma prefiltering is useful, so the "helps one
model, hurts the other" hazard does not arise — it simply helps neither.

## 10. Strength sensitivity

| DNI | weak | medium | strong |
|---|---|---|---|
| 0.00 | −1.0 % | −0.7 % | −0.7 % |
| 0.25 | −0.0 % | +0.4 % | +0.3 % |
| 0.50 | +3.5 % | −0.1 % | −0.4 % |
| 0.75 | +6.8 % | −0.6 % | −0.1 % |
| 1.00 | +7.4 % | +0.4 % | −4.0 % |

Δ chroma sigma vs `off`. No rung produces a consistent benefit, and the ladder
is not monotone — `chroma-weak` frequently makes post-SR chroma noise *worse*
(up to +7.4 %), which is not physically meaningful for a smoothing filter and
indicates the measurement is dominated by the network's own regenerated chroma
rather than by the input's. Meanwhile `chroma_gradient` falls monotonically
with strength (−1.3 % weak → −8.2 % strong): the cost scales with the setting
while the benefit never appears. **There is no strength worth choosing.**

## 11. Runtime, memory and large-image safety

**The prefilter is safe and cheap. That was never the problem.**

- **uint8 throughout.** Asserted per cell across all 110 cells; no assertion
  fired. Input dtype and shape are preserved exactly.
- **Bounded memory.** Phase 3A's completed probe measured ~46 MB RSS at 16 MP
  for every chroma rung — one uint8 RGB copy of a 16 MP image — with runtime
  linear in size (2/8/16 MP all ≈6–17 ms/MP).
- **No full-output float32 buffer.** The filter runs on the **input**, capped at
  `max_input_pixels` = 16 MP, versus `max_output_pixels` = 200 MP. That is a
  12.5× smaller ceiling, on uint8 rather than float32 (a further 4×). The
  previous full-resolution `GaussianBlur` failure was on the output side and is
  structurally unreachable in this position.
- **No prefilter OOM** in any cell.
- **PNG output size** is unchanged by the filter (±0.7 % across all rungs),
  consistent with it neither adding nor removing meaningful detail.

**x4plus OOM ladder — a real finding, unrelated to the prefilter.** 9 of 110
cells fell back from `tile=256` (35 tiles) to `tile=128` (~130–140 tiles), all
of them x4plus, with SR times of 44–54 s against 26 s unreduced. This is
x4plus's RRDBNet memory footprint on a 4 GB card under VRAM pressure. It occurs
with the prefilter `off` as well as on — 4 of the 5 `off` cells were also
reduced — so it is a property of the model and the hardware, not of this
experiment. It does mean **x4plus runtime figures in this phase are not
comparable between arms**, and none of the conclusions rest on them.

## 12. Failure cases and limitations

1. **The chroma-noise benefit does not survive SR.** Input-side −7.5…−37.8 %
   (Phase 3A) becomes +0.4 % (C vs A) on the output.
2. **The one visible effect is negative** — grainier output on the noisiest
   image, at normal viewing size, in both DNI conditions.
3. **A chroma-only filter produces a mostly-luma change** downstream (§7.4),
   which is not what the design intended and undermines the premise.
4. **`chroma-weak` is non-monotone**, sometimes raising post-SR chroma noise by
   up to +7.4 %.
5. **`chroma_gradient` is ambiguous by construction** — it cannot separate noise
   removal from edge smearing. This is why §8 was decisive rather than §7.
6. **x4plus runtime is confounded** by tile fallback (§11); its quality figures
   are unaffected.
7. **x4plus was not run at weak/strong** — a deliberate, stated scope reduction.
   Given medium shows no benefit and strong costs more, this is unlikely to
   change the conclusion, but it is untested.
8. **Corpus limits, unchanged from Phase 3A.** Five already-denoised web JPEGs,
   only one genuinely noisy. Every conclusion about noisy images rests on
   low-light-noise. A camera-original high-ISO frame could behave differently.
9. **One region per photograph** was visually inspected — the peak-difference
   block. Other regions were not exhaustively examined.
10. **4x only.** No 8x/16x or target-resolution runs. A prefilter acts once on
    the input regardless of output scale, so this is unlikely to matter, but it
    is not measured.

## 13. Recommendation

### **C — Reject the chroma prefilter.**

> **Chroma prefilter does not provide a meaningful post-SR quality benefit and
> should not enter production.**

The evidence is consistent across every axis measured:

- **Numerically**: +0.4 % chroma noise on top of DNI — the wrong direction —
  against a consistent 4.2–6.0 % loss of colour-boundary definition. DNI alone
  is ~6× more effective at the filter's own job (−4.6 % vs −0.7 %).
- **Visually**: no visible improvement on any of five photographs; a visible
  *degradation* on the one that matters most, at normal viewing size, repeated
  in both DNI conditions.
- **Mechanistically**: the change it produces downstream is larger in luma than
  in chroma, so it is perturbing the reconstruction rather than cleaning colour.
- **Across models**: no benefit on v3 at any of five DNI rungs; no benefit on
  x4plus, where median chroma-noise change is exactly 0.0 %.
- **Across strengths**: no rung helps; cost scales with strength while benefit
  never appears.

Cost and safety are *not* the reason for rejection — at 25 ms and 46 MB it is
one of the cheapest things the pipeline could do. It is rejected because it
does not work.

**Not B (research-only):** parking it implies an open question. There isn't
one — 110 cells and a visual pass across five photographs, two models and five
DNI rungs all agree.

**Not D (another experiment):** the remaining uncertainties (§12.8–12.10) are
about corpus breadth, not about this filter's effect size. Nothing plausible
about a wider corpus turns "+0.4 % and visibly grainier" into a product win.

**What Phase 3A got right and wrong.** Right: the input-side measurement, and
the caution that it needed confirming. Wrong: assuming an input-side gain would
survive the network. The lesson generalises — **for a preprocessing stage, the
only measurement that counts is taken on the output**, and Phase 3A's own
recommendation to confirm before shipping is what prevented a bad integration.

## 14. Production integration proposal

**None.** No integration is proposed and none should be built.

The Phase 3A pipeline sketch (`estimate_noise` → chroma prefilter → neural
passes) should be considered withdrawn. `estimate_noise` remains a validated,
reusable noise estimator — Phase 3A showed it monotone and content-independent
against injected noise — but it now has no consumer.

**Real-ESRGAN DNI stays the sole denoiser**, unchanged, at its shipped values.

## 15. Remaining uncertainty

Low, for the question asked. The result is consistent across 110 cells, two
models, five DNI rungs, three filter strengths and five photographs, and the
visual pass agrees with the numbers on direction while correcting their sign on
`hf_ratio`.

What is genuinely still open, and is *not* about this filter:

- **Standard has no denoiser at all.** x4plus cannot denoise by any mechanism
  currently in the product. This phase shows chroma prefiltering is not the
  answer; it does not show what is.
- **The corpus still lacks a camera-original high-ISO photograph.** Every
  noisy-image conclusion since Phase 2.5 rests on one Wikimedia night shot.
- **DNI's own ceiling is unmeasured** on genuinely noisy input.

---

## Reproducibility

| | |
|---|---|
| Benchmark date | 2026-09-09 (UTC) |
| Raw measurements | `phase3b_measurements.json` (115 rows: 110 matrix + 5 crops) |
| Runner | `benchmarks/phase3b.py` |
| Definitions reused | `benchmarks/prefilter.py` (unmodified), `metrics.py`, `corpus.py` |
| Crops | `phase3b-crops/` — 5 four-panel strips + 10 pairwise peak-difference crops |
| Model A | `realesr-general-x4v3`, dni 0.00 / 0.25 / 0.50 / 0.75 / 1.00 |
| Model B | `RealESRGAN_x4plus`, no DNI (no `denoise_pair`) |
| Chroma arms | off / weak / strong / medium — `cv2.bilateralFilter` on Cr and Cb only |
| Scale | 4x, single native pass |
| Output | PNG, lossless; metrics from the array, size from the encode |
| Tiling | production defaults (tile 256, pad 16); 9 cells fell back to 128 |
| Sharpening | 0.0 |
| Metric tiles | 16 × 512 px, seed 20260908; flat mask from the `off` arm, shared |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11 |
| Python / NumPy / OpenCV / PyTorch | 3.11.9 / 2.4.6 / 5.0.0 / 2.7.1+cu118 |

Rerun with `python -m benchmarks.phase3b --all`.

### Experiments not performed

- **x4plus at chroma weak/strong** — scope reduction, §3, reasons stated.
- **8x / 16x / target-resolution runs** — a prefilter acts once on the input;
  not measured.
- **Phase 3B's own scaling probe** — Phase 3A's completed probe covers the same
  filters at 2/8/16 MP and is cited rather than duplicated.
- **Per-cell peak VRAM** — CUDA peak is process-wide and cannot be attributed to
  a cell. Not reported rather than reported wrongly.

No result in this report was estimated or extrapolated. Every number comes from
`phase3b_measurements.json` or from analysis run directly against it, and every
visual claim comes from an image in `phase3b-crops/` that was actually viewed.
