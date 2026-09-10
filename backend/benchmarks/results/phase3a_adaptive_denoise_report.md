# Phase 3A — Adaptive noise reduction research

**Status: research complete, nothing applied.** No production behaviour
changed. `DEFAULT_DENOISE`, `default_model`, `MAX_OUTPUT_PIXELS`, Standard,
Creative, 8x/16x, 16K, sharpening and the frontend are all untouched. No
dependency was added.

**Headline: the central hypothesis was refuted.** A denoise strength driven by
an image's noise level does not work. The estimator that measures noise is
sound and validated; the mapping from that measurement to a filter strength is
not, and this report recommends against building it.

---

## 1. Objective

Determine the best practical noise-reduction strategy for real photographs
*before* super-resolution, and decide between four architectures:

- **A** keep Real-ESRGAN DNI/denoise only;
- **B** add fixed CPU preprocessing denoise before SR;
- **C** add adaptive preprocessing denoise before SR;
- **D** a hybrid.

Phase 2.5 motivated the question. It found that a single global denoise value
cannot be right for every image: on the low-light photograph, denoising cost
0.22–0.45 units of high-frequency detail per unit of noise removed; on the
portrait, where there was little noise to begin with, 0.25 cost **6.7×** more
detail than the noise it gained back. That is an argument for adaptivity. This
phase tests whether adaptivity can actually be implemented.

## 2. Existing pipeline

Confirmed by reading `app/services/enhancement_service.py` and
`app/inference/model_manager.py`, not assumed:

```
split_alpha
  → [neural pass 1 … neural pass N]      ← denoise happens *inside* here
  → resize_to_target   (target-resolution jobs only)
  → unsharp_mask       (sharpen_strength > 0 only)
  → attach_alpha
```

**There is no preprocessing stage at all today.** Denoising is not a filter; it
is DNI weight interpolation, performed once at model load by
`ModelManager._load`, which blends the `realesr-general-x4v3` weights toward
their `realesr-general-wdn-x4v3` counterpart at alpha = `denoise_strength`.

Two consequences shape everything below:

- it is **free at inference**. Phase 2.5 measured processing time flat across
  all five denoise values, because the blend happens once at load;
- it reaches **exactly one model**. `realesr-general-x4v3` is the only registry
  entry with a `denoise_pair`, so Standard (`RealESRGAN_x4plus`) has no
  denoising stage whatsoever, and `_denoise_for` drops the setting for it.

Creative today is `realesr-general-x4v3` at a fixed denoise of 0.25 — one
model, one constant, chosen in Phase 2.5 and not adapted to anything.

## 3. Noise problem definition

Task 2 asks that six things be told apart. What this phase can and cannot
distinguish:

| Class | Addressed? | How |
|---|---|---|
| Sensor / high-ISO noise | Yes | The `low-light-noise` photograph, and the injected-gaussian ladder |
| JPEG / compression noise | **Partly** | Present in every corpus image (all are JPEG), but never isolated as a variable. No arm targets blocking or mosquito noise specifically. |
| Chroma noise | Yes | Measured separately in Cr/Cb; one candidate (`chroma-*`) targets it exclusively |
| Luminance noise | Yes | The primary axis of `flat_noise` and `sigma_luma` |
| Fine texture that must **not** be removed | Yes | `foliage-texture` and `portrait-skin`; `high_frequency_ratio` and `local_contrast`; visual crops |
| Edges that must **not** be blurred | Yes | `text-signage`; `sobel_p95` and `edge_overshoot`; visual crops |

### Metric limitations, stated up front

These matter for reading everything below.

1. **Every metric is reference-free and is a proxy.** There is no ground truth;
   only deltas between arms over the same image mean anything.
2. **All six metrics are computed on Rec.709 luma.** They are therefore
   *structurally blind to chroma-only processing*. The `chroma-*` arms show
   ≈0 % change on every luma metric by construction, not by failing. Their
   effect was measured separately (§7.4) and inspected visually.
3. **`flat_noise` selects the flattest windows of the image it is given.**
   When a filter changes which windows are flattest, before/after comparison
   silently changes the sample. This confounds the chroma-channel figures for
   `nlm-*` and `bilateral-*` in §7.4 and they are reported as unreliable.
4. **The cost ratio `|Δhf| / |Δnoise|` is normalised by each image's own
   baseline.** An image with very little high-frequency energy to begin with —
   the portrait, at `hf = 0.00015`, 230× below the foliage — shows a large
   *percentage* change from a small absolute one. Comparisons of this ratio
   *between* images are therefore weaker than comparisons within one image.
   This is a real limitation and it affects §9.
5. **`edge_overshoot` is a halo proxy**, not a measurement of ringing.
6. No metric here models perception. The visual pass in §8 is not decoration;
   it caught the single most important failure in the phase.

## 4. Dataset

The existing real-photo corpus, unchanged and re-validated through
`corpus.load_manifest()` on every run so a run cannot measure a corpus other
than the one it reports.

| Category | Dimensions | Licence | Author |
|---|---|---|---|
| landscape-detail | 1732×1154 | CC BY 4.0 | Vyacheslav Argenberg |
| portrait-skin | 1307×1530 | CC BY 4.0 | not stated on Commons |
| text-signage | 1632×1224 | CC BY-SA 4.0 | PortlandAppraisalBlog |
| foliage-texture | 1224×1632 | CC BY-SA 4.0 | Derk29 |
| low-light-noise | 1892×1057 | CC0 | Wilfredor |

Centre crops of the originals, **never resampled** — downscaling averages
sensor noise away, and sensor noise is the subject. Images are not committed;
`corpus/manifest.json` makes the corpus reproducible.

Synthetic material was used **only** where a known answer is the point:
gaussian noise injected at known sigma into these same photographs, so the
content stays real while the noise becomes checkable. No production conclusion
rests on a synthetic image.

### Measured noise of the corpus

Combined sigma (`sigma_luma + 0.35 × sigma_chroma`), from `estimate_noise`:

| Category | sigma_luma | sigma_chroma | combined |
|---|---|---|---|
| text-signage | 0.300 | 0.004 | 0.302 |
| portrait-skin | 0.367 | 0.087 | 0.398 |
| foliage-texture | 0.377 | 0.140 | 0.426 |
| landscape-detail | 0.428 | 0.067 | 0.451 |
| low-light-noise | **0.674** | **0.169** | **0.734** |

**The corpus is a poor noise dataset.** Its noisiest image is only 2.4× its
cleanest, and all five are clean in absolute terms — they are web JPEGs that
have already been through a denoiser somewhere. This is the single biggest
limitation of the phase and it is why the ladder in §9 exists.

## 5. Algorithms tested

Fifteen arms across six families, all OpenCV or NumPy — both already required,
so **no dependency was added**. `scikit-image`, `scipy` and BM3D were
considered and rejected on that basis: adding a dependency is an engineering
decision, not a benchmark arm.

| Family | Arms | Rationale |
|---|---|---|
| **A** none | `none` | Baseline. Every delta is measured against it. |
| **B** Gaussian | `gaussian-{weak,medium,strong}` σ 0.5/1.0/1.8 | The edge-blind control. Not a serious candidate — included because every edge-aware method must beat it. |
| **C** Bilateral | `bilateral-{weak,medium,strong}` d 5/7/9, σ_c 15/35/75 | The classic edge-preserving filter. |
| **C2** Chroma-only bilateral | `chroma-{weak,medium,strong}` | Filters Cr/Cb only, leaves Y untouched. Designed around the noise *taxonomy* rather than around an algorithm. |
| **D** Non-local means | `nlm-{weak,medium,strong}` h 3/6/10 | The quality reference. Exploits patch self-similarity, which photographs have and noise does not. |
| — Median | `median-3`, `median-5` | Benchmarked to be ruled out with evidence rather than by assertion. |
| **E** Real-ESRGAN DNI | `dni-0.25`, `dni-1.00` | The incumbent, measured in the same run so it is directly comparable. |

Not benchmarked, with reasons: **BM3D** (no dependency-free implementation),
**guided filter** (`cv2.ximgproc` is not in this OpenCV build — verified, not
assumed), **wavelet shrinkage** (needs `scipy`), **learned denoisers**
(a new model, explicitly out of scope).

## 6. Benchmark methodology

Five experiments, in `benchmarks/phase3a.py`. Separated because they answer
different questions at very different costs.

| # | Experiment | What it answers | Cells |
|---|---|---|---|
| 1 | `--filters` | What each filter does to the input | 5 images × 15 arms = 75 |
| 2 | `--ladder` | Does the estimator track *known* noise? | 5 × 5 sigmas = 25 |
| 3 | `--sr` | What the model produces from a filtered input | 5 × 7 arms = 35 |
| 4 | `--scaling` | Runtime and memory to the 16 MP input ceiling | 3 sizes × 14 arms = 42 |
| 5 | `--crops` | Visual inspection | 5 strips |

Experiment 3 is the one that decides anything. A prefilter is not judged by
what it does to the input — the user never sees the input — but by what
Real-ESRGAN then produces from it, and the network was trained on degraded
inputs, so pre-smoothing is not guaranteed to help merely because the
intermediate looks cleaner.

In experiment 3 the DNI blend is pinned to **0.0** for every prefilter arm, so
the prefilter is the only variable; the `dni-*` arms then carry no prefilter.
The two denoising strategies are therefore measured against the same baseline
and are directly comparable.

Metrics come from the existing `benchmarks/metrics.py`, unchanged: 16
deterministic seeded 512 px tiles per image, Rec.709 luma, float64.

## 7. Quantitative results

### 7.1 Composite table — SR output, median across the five photographs

Delta against `none`. `ratio` = `|Δhf| / |Δnoise|`, **lower is better**: it is
the detail paid per unit of noise removed.

| arm | Δ noise | Δ hf | Δ contrast | Δ overshoot | **ratio** |
|---|---|---|---|---|---|
| `gaussian-medium` | −30.8 % | −54.2 % | −28.3 % | −58.3 % | 1.96 |
| `bilateral-medium` | −51.9 % | −56.4 % | −20.1 % | −15.1 % | 0.89 |
| `chroma-medium` | −0.5 % | +1.6 % | +0.1 % | +1.4 % | (n/a — see §7.4) |
| `nlm-medium` | −54.1 % | −57.8 % | −22.6 % | −21.6 % | 0.87 |
| `dni-0.25` | −12.9 % | −14.1 % | −0.1 % | −3.7 % | 1.36 |
| **`dni-1.00`** | −46.4 % | −49.9 % | −9.0 % | −16.7 % | **0.65** |

**The incumbent wins the median.** DNI at 1.00 has the best detail-per-noise
ratio of every arm tested, and the best local-contrast preservation of any arm
that removes a comparable amount of noise. No fixed prefilter beat it.

That is the finding that rules out architecture **B**.

Gaussian is disqualified as expected: it removes the least noise of any
non-trivial arm while destroying the most detail, and its −58.3 % overshoot is
the signature of an edge-blind operator flattening edges outright.

### 7.2 The adaptivity argument, quantified

The same filter, at the same parameters, on the noisiest and cleanest images:

| arm | low-light-noise | portrait-skin | swing |
|---|---|---|---|
| `nlm-medium` | **0.16** | **1.16** | **7.3×** |
| `bilateral-medium` | 0.32 | 1.60 | 5.0× |
| `dni-1.00` | 0.22 | 1.41 | 6.4× |

On the noisy image `nlm-medium` removes 51.5 % of the noise for 8.0 % of the
high-frequency energy. On the clean portrait the same call removes 54.1 % of
the noise for **62.7 %** of the detail.

This is a real and large effect, and it is why a fixed prefilter is the wrong
architecture. It is *not*, however, evidence that a noise-driven mapping can
exploit it — see §9, where that is tested directly and fails.

### 7.3 Per-image detail

`low-light-noise` — the image denoising exists for:

| arm | Δ noise | Δ hf | ratio |
|---|---|---|---|
| `gaussian-medium` | −38.7 % | −75.8 % | 1.96 |
| `bilateral-medium` | −51.9 % | −16.4 % | 0.32 |
| `chroma-medium` | −3.4 % | **+10.1 %** | — |
| **`nlm-medium`** | −51.5 % | **−8.0 %** | **0.16** |
| `dni-1.00` | −46.4 % | −10.2 % | 0.22 |

`portrait-skin` — the image denoising should leave alone:

| arm | Δ noise | Δ hf | ratio |
|---|---|---|---|
| `gaussian-medium` | −20.1 % | −54.2 % | 2.70 |
| `bilateral-medium` | −38.6 % | −61.7 % | 1.60 |
| `chroma-medium` | −1.2 % | +1.6 % | — |
| `nlm-medium` | −54.1 % | −62.7 % | 1.16 |
| `dni-0.25` | −2.1 % | −14.1 % | 6.71 |

Median filtering is ruled out on evidence: `median-3` on the low-light input
removed 23 % of the noise for 50 % of the high-frequency energy — a worse trade
than bilateral at a fraction of the quality.

### 7.4 Chroma, measured directly

The luma metrics cannot see chroma-only filtering, so chroma sigma was measured
in Cr/Cb separately.

| Image | Δ chroma sigma | Δ luma sigma |
|---|---|---|
| landscape-detail | **−34.5 %** | **0.0 %** |
| portrait-skin | −15.5 % | +0.1 % |
| foliage-texture | −7.5 % | −0.3 % |
| low-light-noise | **−37.8 %** | **0.0 %** |
| text-signage | +155.7 % † | 0.0 % |

`chroma-medium` does exactly what it claims: it removes 7.5–37.8 % of chroma
noise at **0.0 %** luma cost. The zero in the right-hand column is not an
approximation — the Y channel is passed through untouched.

† text-signage has essentially no chroma noise to begin with
(`sigma_chroma = 0.004`, an order of magnitude below every other image), so its
relative change is a large percentage of nearly nothing. The same near-zero
baseline problem Phase 2.5 hit with foliage.

**The chroma figures for `nlm-*` and `bilateral-*` are unreliable** and are
omitted: those filters change the luma channel, which changes which blocks
qualify as flat, so the before/after samples differ. Reporting the apparent
+115 % to +721 % as a finding would be wrong.

## 8. Visual observations

Task 6 is explicit that metrics are not sufficient, and it was right: the most
important result in this phase is visual and no metric flagged it clearly.

Crops in `phase3a-crops/`, five arms at 2×, nearest-neighbour so nothing is
resampled between filter and eye. Regions were chosen by hand — Phase 2.5's
automatic highest-variance selection landed on dark foliage instead of the
lettering it meant to inspect, and that gap is closed here.

**`portrait-skin` — the decisive crop.** `none` shows pores, fine wrinkle
lines and surface structure. `chroma-medium` is visually indistinguishable from
it. `bilateral-medium` is noticeably smoothed. **`nlm-medium` renders the skin
plastic** — pores gone, fine lines mostly gone, a waxy surface. It looks worse
than the Gaussian panel beside it. This is the classic over-denoised-skin
failure, produced on an image that is already clean, by the arm that scores
best on the noisy image.

**`low-light-noise`.** `none` is visibly grainy. `gaussian-medium` destroys the
chain-link fence and the brick texture entirely. `bilateral-medium` keeps
structure but shows patchy flattening in the water. `chroma-medium` is
structurally identical to `none` with the colour mottle reduced.
`nlm-medium` is the cleanest while keeping the fence resolvable and the water
ripples intact — the arm earning its place.

**`text-signage` — edges and ringing.** No arm introduced a visible halo;
these are all smoothing operators, and the `edge_overshoot` column agrees.
`nlm-medium` preserved the hard letter edges sharply while cleaning the flat
dark sign face. `gaussian-medium` visibly softened the letter edges.

**`foliage-texture`.** `nlm-medium` keeps stem structure but smooths the fine
bark speckle; `bilateral-medium` flattens more; `gaussian-medium` erases it.

**Conclusion from the eye:** NLM's failure is specifically on *low-contrast
fine texture* (skin), not on edges, and it is severe. Chroma-only filtering
introduces no visible artefact on any image in the corpus.

## 9. Noise estimation experiments

`estimate_noise` in `benchmarks/prefilter.py`. Three steps, each inspectable:

1. split into 32 px blocks, keep only the flattest **10 %** — texture lives in
   the rest, and counting it would be the exact failure Task 2 names;
2. estimate sigma on those blocks by Immerkær's Laplacian-response method,
   separately for luma and chroma, taking the **median** across blocks so a few
   flat-but-edged blocks cannot drag it up;
3. stretch `sigma_luma + 0.35 × sigma_chroma` between two anchors, clamp to
   [0, 1].

### The estimator works

Gaussian noise at known sigma, injected into the five real photographs:

| Injected σ | 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| landscape-detail | 0.43 | 1.52 | 2.81 | 5.30 | 9.11 |
| portrait-skin | 0.37 | 1.44 | 2.71 | 5.26 | 10.22 |
| text-signage | 0.30 | 1.41 | 2.68 | 5.02 | 8.45 |
| foliage-texture | 0.38 | 1.42 | 2.63 | 5.02 | 8.96 |
| low-light-noise | 0.67 | 1.56 | 2.78 | 5.24 | 9.17 |

Monotone in every row, and — the important part — **nearly identical across
five photographs with wildly different texture**. A dense fern and a smooth
portrait return the same reading at the same injected noise. That is strong
evidence the flat-block restriction is doing its job: the estimator is
measuring noise, not content.

It under-reads by a consistent ≈0.55×. Most of that is expected: independent
per-channel RGB noise becomes luma noise scaled by √(0.299²+0.587²+0.114²) =
0.669. The remainder is the flat-block restriction and the median, both of
which bias downward — which is the intended direction for something that
decides how hard to filter.

### The anchors are half-calibrated

`SIGMA_CLEAN = 0.45` is constrained by the real corpus. It sits above
text-signage (0.302) and portrait-skin (0.398) — the two images Phase 2.5
independently showed denoising *harms* — and at landscape-detail's level.

`SIGMA_NOISY = 3.0` is **not constrained by the real corpus**, because the
corpus contains no genuinely noisy photograph. It comes from the injection
ladder. It is also demonstrably too low: the score saturates at 1.000 by
injected σ 4, so across most of the tested range the score cannot discriminate
at all. This is the weakest number in the phase.

## 10. Adaptive mapping experiments — **the hypothesis failed**

The mapping rests on one premise: *a high noise score means denoising is cheap;
a low one means it is expensive.* Tested directly — inject known noise into the
real photographs, score each, measure what `nlm-medium` then costs.

25 image–noise pairs:

```
Pearson r(noise score, cost ratio) = +0.077
```

No relationship, and the sign is the opposite of the one predicted.

The cost is instead a property of the **content**, essentially constant across
every noise level from clean to σ 16:

| Image | cost ratio, σ 0 → 16 |
|---|---|
| low-light-noise | 0.02 → 0.02 |
| foliage-texture | 0.05 → 0.16 |
| text-signage | 0.05 → 0.37 |
| landscape-detail | 0.23 → 0.26 |
| **portrait-skin** | **0.87 → 0.88** |

**The portrait is the failure case.** Injected at σ 16 it scores a saturated
1.000, so `adaptive_strength` proposes 0.6 — and the visual crop at that
setting is the plastic skin of §8. A mapping from noise level alone will
over-filter a noisy portrait, because *how much noise an image carries says
nothing about how much fine texture is at risk*.

Task 7 warned against assuming one statistic is sufficient. It is not, and this
is the measurement that shows it.

Caveat, stated because it partly softens the result: the cost ratio is
normalised by each image's own baseline `hf`, and the portrait's baseline is
230× below the foliage's (§3, limitation 4). Some of the between-image spread
is that normalisation rather than real difference. **The visual evidence is not
subject to that caveat**, and it points the same way.

## 11. Memory and runtime

### Scaling to the 16 MP input ceiling

`max_input_pixels` is 16 MP, so this is the worst case a prefilter would face.

| arm | 2 MP | 8 MP | 16 MP | ms/MP | RSS Δ at 16 MP |
|---|---|---|---|---|---|
| `gaussian-medium` | 1.3 ms | 6.3 ms | 11.6 ms | 0.7 | 45.3 MB |
| `median-3` | 1.6 ms | 7.3 ms | 13.1 ms | 0.8 | 45.8 MB |
| `bilateral-medium` | 19.4 ms | 68.4 ms | 153.1 ms | 9.6 | 45.6 MB |
| `chroma-medium` | 20.2 ms | 86.4 ms | **187.1 ms** | 11.7 | **45.9 MB** |
| `nlm-medium` | 1402 ms | 5460 ms | **11741 ms** | 733.9 | 46.0 MB |

**Memory is flat at ≈46 MB for every filter, including NLM.** That is one uint8
RGB copy of a 16 MP image (16 M × 3 = 48 MB). No candidate allocates a
full-resolution float32 buffer, and none allocates per-tile scratch that grows
with image size.

This is the direct answer to Task 9's concern. The out-of-memory failure this
project hit previously was a full-resolution `GaussianBlur` on the **output**
side, where the array can be 200 MP. A prefilter runs on the **input**, capped
at 16 MP — a 12.5× smaller ceiling — and on uint8 rather than float32, a
further 4×. The dangerous configuration is structurally not reachable here.

Runtime scales linearly (ms/MP flat within measurement noise across an 8×
size range), so nothing pathological appears at scale.

### Cost in context

NLM at 16 MP is **11.7 seconds** — roughly 78× bilateral and 1000× Gaussian.
For a typical 2 MP upload it is 1.4 s against an SR pass of ≈1.8 s, so about
**+78 %** wall-clock. Chroma-only at the same 2 MP is 20 ms, about **+1 %**.

`strip_rows`-style strip processing was **not** implemented or measured: with
peak memory flat at one input-sized copy, there is nothing for it to solve.
Recorded as not needed rather than as done.

### A timing artefact

The first SR cell of experiment 3 (`landscape-detail` / `none`) reports
10 934 ms against ≈1 800 ms for every subsequent cell. That is model load and
CUDA warm-up, not the arm. It is excluded from all timing conclusions; only the
prefilter timings, measured separately and repeatedly, are used.

## 12. Failure cases

1. **The adaptive mapping (§10).** The headline failure. r = +0.077.
2. **NLM on clean skin (§8).** Plastic surface at `medium` on an image that did
   not need denoising. Visually worse than Gaussian.
3. **No fixed prefilter beat the incumbent (§7.1).** DNI 1.00's median ratio of
   0.65 was not matched by any prefilter arm.
4. **Score saturation (§9).** `SIGMA_NOISY = 3.0` saturates the score by
   injected σ 4, so most of the tested range is indiscriminable.
5. **`text-signage` chroma is a near-zero baseline (§7.4)** — the same class of
   artefact that made Phase 2.5 exclude foliage. Its +155.7 % is not a finding.
6. **Chroma figures for NLM and bilateral are confounded (§7.4)** by flat-mask
   shift and are withheld rather than reported.
7. **The corpus is inadequate for this question (§4).** Five images, 2.4×
   noise spread, all already-denoised web JPEGs, and none is a genuinely
   high-ISO frame. Every conclusion about noisy images rests on one photograph
   plus injected noise.
8. **JPEG/compression noise was never isolated** as a variable, despite being
   one of the six classes Task 2 names.

## 13. Recommendation

### **D — Hybrid**, in a narrow and specific form.

Concretely: **keep Real-ESRGAN DNI as the luma denoiser, and pursue chroma-only
prefiltering as a separate, targeted addition. Do not build adaptive luma
preprocessing.**

The reasoning, arm by arm:

- **Not B (fixed preprocessing).** Directly refuted. No fixed prefilter beat
  DNI 1.00's median ratio of 0.65, and DNI is free at inference where NLM costs
  11.7 s at 16 MP.
- **Not C (adaptive preprocessing), yet.** The estimator works; the mapping
  does not. Shipping a mapping with r = +0.077 to its own premise would be
  shipping a number that looks principled and is not — and its concrete failure
  mode is destroying skin texture on noisy portraits, which is a *worse* user
  outcome than the fixed default it would replace.
- **Not pure A either.** DNI has two structural gaps that no amount of tuning
  closes: it reaches only `realesr-general-x4v3`, leaving Standard with no
  denoising at all; and it is a luma-dominant operator that leaves chroma
  mottle largely intact — measured at −37.8 % chroma sigma achievable, which
  DNI does not deliver.

**Chroma-only prefiltering is the one clear win**, and it is unusually
low-risk:

| | |
|---|---|
| Quality | −7.5 % to −37.8 % chroma noise |
| Luma cost | **0.0 %** — the Y channel is not touched |
| Detail cost | none measurable; visually indistinguishable from unfiltered |
| Runtime | 20 ms at 2 MP, 187 ms at 16 MP (≈1 % of an SR pass) |
| Memory | 46 MB at 16 MP, flat — one uint8 copy |
| Model coverage | Works for **every** model, including Standard |
| Dependency | none — `cv2.bilateralFilter` on Cr/Cb |

There is also a suggestive secondary effect: on the two images with real chroma
noise, chroma prefiltering *raised* measured high-frequency energy after SR
(+10.1 % low-light, +11.5 % landscape). A plausible mechanism is that chroma
noise wastes model capacity. **This is n = 2 and is not a claim** — it is a
reason to measure it properly in 3B, not a reason to act.

### If adaptive is pursued later, what it would need

Not a specification — the evidence does not support one — but the shape of the
gap:

- **algorithm**: NLM for luma (best measured trade where noise is real);
  chroma-bilateral for colour;
- **estimator**: `estimate_noise` is sound and reusable as-is;
- **second signal, missing**: a texture-at-risk measure. Noise level alone is
  refuted. Without this there is no safe mapping;
- **score range**: [0, 1], but `SIGMA_NOISY` must be recalibrated — 3.0
  saturates too early; the ladder suggests 8–9;
- **placement**: before the first neural pass, on the input, on uint8;
- **memory strategy**: none needed beyond one input-sized copy (§11);
- **interaction with DNI**: they overlap and must not both be applied at full
  strength — untested, and a prerequisite;
- **Standard**: would gain a denoiser it does not have. That is a behaviour
  change requiring its own decision;
- **Creative**: would replace a fixed 0.25 with a computed value;
- **expected improvement**: unquantified. Deliberately.

## 14. Proposed production architecture

For a **future** phase, not this one:

```
upload
  → inspect (existing)
  → estimate_noise                    ← new, ~40 ms at 2 MP, cheap and safe
  → chroma-only prefilter             ← new, gated on sigma_chroma
  → [neural pass … ]                  ← unchanged; DNI still the luma denoiser
  → resize_to_target                  ← unchanged
  → unsharp_mask                      ← unchanged
  → attach_alpha
```

The luma path is untouched. Only chroma is filtered, only when chroma noise is
actually present, and everything downstream is unchanged.

## 15. What should **not** be implemented yet

- **Any adaptive luma denoise mapping.** Refuted (§10).
- **NLM anywhere in production.** 11.7 s at 16 MP, and it renders clean skin
  plastic. It is the best arm on genuinely noisy images and must not be applied
  without a gate that this phase failed to design.
- **Any change to `DEFAULT_DENOISE`, `default_model`, `MAX_OUTPUT_PIXELS`,
  Standard, Creative, 8x/16x, 16K, or sharpening.**
- **A user-facing denoise slider.**
- **Chroma prefiltering itself** — recommended, but on a five-image corpus with
  one genuinely noisy photograph. It needs a 3B confirmation on a wider corpus
  before it ships.
- **A new model, HAT, DRCT, or any new dependency.**

### Prerequisites for Phase 3B

1. **A better corpus.** 20+ photographs including genuine high-ISO frames and
   camera-original files that have not already been denoised. This blocks
   everything else; the current corpus cannot support a stronger conclusion.
2. Isolate **JPEG/compression noise** as its own variable.
3. Design and test the **texture-at-risk** signal §13 identifies as missing.
4. Recalibrate `SIGMA_NOISY` against real noisy photographs.
5. Test **DNI × prefilter interaction**, never measured here.
6. A **perceptual** check — every metric used is a proxy, and the one finding
   that changed the recommendation came from looking.

---

## Reproducibility

| | |
|---|---|
| Benchmark date | 2026-09-09 (UTC) |
| Report | `benchmarks/results/phase3a_adaptive_denoise_report.md` |
| Raw measurements | `benchmarks/results/phase3a_measurements.json` (182 rows) |
| Definitions | `benchmarks/prefilter.py` |
| Runner | `benchmarks/phase3a.py` |
| Crops | `benchmarks/results/phase3a-crops/` |
| Corpus manifest | `benchmarks/corpus/manifest.json` |
| Model | `realesr-general-x4v3` (the only entry with a denoise pair) |
| Scale | 4x, single native pass |
| Output format | in-memory `uint8` RGB; no encoder between model and metrics |
| Sharpening | 0.0 |
| DNI in prefilter arms | 0.0, so the prefilter is the only variable |
| Tiling / device | defaults — CUDA, tile 256, pad 16 |
| Noise injection seed | 20260909 |
| Metric tiles | 16 × 512 px, seed 20260908 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Windows 11 |
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| OpenCV | 5.0.0 |
| PyTorch | 2.7.1+cu118 |
| Anchors in use | `SIGMA_CLEAN` 0.45, `SIGMA_NOISY` 3.0 |
| Corpus 20th/80th percentile | 0.378 / 0.508 |

Rerun with:

```
python -m benchmarks.phase3a --all
```

### Experiments not performed

- **BM3D**, **guided filter**, **wavelet shrinkage** — would each require a new
  dependency (§5). Not attempted; documented rather than forced.
- **Strip/tile prefiltering** — peak memory is flat at one input-sized copy, so
  there was nothing to solve (§11). Not needed rather than done.
- **DNI combined with a prefilter** — the two were deliberately never applied
  together, so their interaction is unmeasured. A prerequisite for 3B.
- **8x, 16x and target-resolution jobs** — all measurement is at 4x. A
  prefilter runs once on the input regardless of output scale, so the scaling
  probe to 16 MP covers the memory question, but no end-to-end 8x/16x run was
  made.
- **Per-run peak VRAM** — `peak_vram_mb` exists in the harness but CUDA peak is
  process-wide, so it cannot be attributed to a cell. Not reported rather than
  reported wrongly.

No result in this report was estimated, extrapolated or filled in. Every number
comes from `phase3a_measurements.json` or from the analysis scripts run against
it.
