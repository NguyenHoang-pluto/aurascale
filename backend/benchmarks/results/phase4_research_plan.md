# Research consolidation and Phase 4 plan

**Research documentation only.** No production code changed, no dependency
added, no model added, no benchmark run to produce this document. Everything
below is read from artifacts already in the repository, from the production
source, or from three cheap code-level verifications noted as such.

Source of truth: `phase2_5_denoise_report.md`, `phase3_model_report.md`,
`phase3a_adaptive_denoise_report.md`, `phase3b_chroma_dni_report.md`,
`phase3c_detail_model_report.md` and their measurement JSON.

Repository state: HEAD `73e4e06` ("feat: add 16x scale and 16K target").
Working tree clean of tracked changes; only untracked Phase 3A/3B/3C research
artifacts. Tests: backend 713 passed / 2 skipped, frontend 427 passed.

Phase 2 has no standalone report — its synthetic-corpus result is recorded
inside the Phase 2.5 report. Phase 2.75 was a **feature** phase (16x scale, 16K
target), not a benchmark, and correctly has no measurements.

---

## A. What we know

| # | Finding | Phase | Evidence |
|---|---|---|---|
| A1 | Denoise at 1.0 is destructive on real photographs | 2.5 | Median **−53 % high-frequency energy**, −25 % local contrast on text |
| A2 | Denoise cost is **content-dependent by >10×** | 2.5 | Detail-per-noise ratio: low-light **0.22–0.45** (cheap), portrait **6.71** at 0.25 (expensive) |
| A3 | 0.25 is the least destructive setting that still denoises | 2.5 | −15.7 % hf for −15.3 % noise; recommended, **never applied** |
| A4 | DNI costs nothing at inference | 2.5 | Processing time flat across all five values — the blend happens once at load |
| A5 | `x4plus` is the best general model available | 3, 3C | Best skin, no restyling, lowest artifact risk; nothing has beaten it in two independent comparisons |
| A6 | `general-v3` wins on landscape detail, loses badly on skin | 3 | +21.1 % sobel, +41.8 % hf on landscape — but **+182.5 % flat noise** and −73 % hf on portrait |
| A7 | `general-v3` is 4.1× faster and uses 1/17 the VRAM | 3 | 9.6 s vs 38.8 s; 34.8 MiB vs 584 MiB at tile 256 |
| A8 | `anime_6B` restyles photographs | 3 | Replaces pores with painterly strokes; **+28.7 % overshoot**, highest of any arm |
| A9 | A noise **estimator** can be built and is content-independent | 3A | Monotone against injected noise, near-identical response across five very different photographs |
| A10 | Noise level does **not** predict whether denoising is safe | 3A | **Pearson r = +0.077** across 25 image-noise pairs; cost is a property of content (portrait 0.87–0.88 at every noise level) |
| A11 | Prefilter gains measured on the **input** do not survive the network | 3B | Input-side −7.5…−37.8 % chroma noise becomes **+0.4 %** on the output |
| A12 | `RealESRGAN_x4plus` has **no denoise control at all** | 3B | No `denoise_pair`; `_denoise_for` drops any value. Standard has no denoiser |
| A13 | For `general-v3`, `denoise=None` ≡ `denoise=1.00` | 3B | Byte-identical output, max abs diff 0. "No denoise" is the **most**-denoised state |
| A14 | Transformer SR is not viable on 4 GB | 3C | HAT: **5.35× runtime, 3.98× VRAM**, fp32 forced (fp16 = 100 % NaN), tile forced to 128, **visible tile seam** |
| A15 | Sharpness metrics do not track perceived quality | 3B, 3C | 3B: `hf_ratio` rose while the image visibly got grainier. 3C: HAT is *softer* on every metric and still not better |
| A16 | Automatic artifact detection is unreliable | 3C | Two detectors ranked HAT *cleaner* than baseline while a plainly visible tile seam sat in its output |
| A17 | The 4 GB card constrains the **baseline** too | 3C | `x4plus` OOM-laddered to tile 128 on **4 of 5** images |
| A18 | Blind visual evaluation is reproducible | 3C | The blind pass independently re-derived a known Phase 3 result (v3 at denoise 1.0 flattens pores) without that being known at the time |

## B. What we rejected, and why

| Approach | Phase | Verdict | Evidence |
|---|---|---|---|
| Denoise default 1.0 | 2.5 | Rejected as a default | −53 % median hf; the reversible error (under-denoising) is the better default |
| Synthetic-texture corpus | 2 | Disqualified itself | The model correctly erased gaussian "texture"; conclusions could not transfer to photographs |
| `anime_6B` for photographs | 3 | Rejected | Restyles rather than restores |
| Adaptive luma denoise from a noise score | 3A | **Refuted** | r = +0.077 to its own premise; would apply 0.6 strength to a noisy portrait and destroy skin |
| NLM prefilter | 3A | Rejected | Plastic skin on a *clean* portrait; 11.7 s at 16 MP; no fixed prefilter beat DNI's median ratio of 0.65 |
| Gaussian / median prefilter | 3A | Rejected | Gaussian: −30.8 % noise for −54.2 % hf. Median: −23 % noise for −50 % hf |
| Chroma-only prefilter | 3B | **Rejected** | +0.4 % chroma noise on top of DNI, −4.4 % colour-boundary definition, visibly grainier on the noisiest image; the change lands in **luma**, not chroma |
| HAT (`Real_HAT_GAN_SRx4`) | 3C | **Rejected** | See A14; softer on every detail metric; 6 of 8 production-gate questions negative |
| Face-restoration GANs (GFPGAN, CodeFormer, RestoreFormer) | 3C | Excluded by design | Generative face priors invent detail and move identity — disqualified by the faithful-restoration goal. CodeFormer/RestoreFormer also carry non-commercial licences |
| Classical (bicubic-trained) SR checkpoints | 3C | Excluded | Wrong degradation model; would measure a mismatch and read as an architecture verdict |
| Model routing (portrait → A, landscape → B) | 3C | Not justified | Differences too small and inconsistent; would add a classifier and a second resident model to a card that already OOM-ladders |

## C. What currently works

- **`RealESRGAN_x4plus` as Standard.** Survived two independent comparisons.
- **The tiling path and OOM ladder.** Handled `x4plus` reductions and HAT's
  2.9 GiB peak without a single inference failure across 130 benchmark cells.
- **The memory discipline.** Strip-processed sharpening; prefilters bounded at
  one uint8 input copy (~46 MB at 16 MP); input capped at 16 MP against a
  200 MP output ceiling.
- **DNI as the denoiser.** Free at inference, and Phase 3B showed it out-denoises
  a purpose-built chroma filter by ~6×.
- **The benchmark infrastructure itself.** Five-photo manifest-validated corpus,
  reference-free metrics, blind protocol, per-run configuration capture. Phase
  3C reused Phase 3A/3B code unmodified.
- **The unsharp-mask implementation** — scale-aware sigma, luma-only (no colour
  fringing), dead-zone + ceiling shaping, strip-processed. Well designed. See D3.

## D. Current quality bottleneck

The evidence does not point at the model or at preprocessing. Both were tested
and both were rejected. It points at **parameters and wiring that were decided,
implemented, and then never actually reached the output.**

### D1 — The denoise default is still 1.0, and 1.0 is the worst setting

`DEFAULT_DENOISE = 1` in `frontend/src/stores/useEnhancementStore.ts`. Phase 2.5
recommended **0.25** and the change was never applied. Phase 3 measured denoise
1.0 on portrait skin at **−88.4 % hf** against `x4plus`.

### D2 — Enhancement Mode currently has no effect on the output

Verified at code level, not inferred:

- `buildSettings` sends `denoiseStrength` whenever the model supports it, and
  its default is `DEFAULT_DENOISE = 1`;
- the request always carries an explicit `model`;
- `resolve_model(CREATIVE, "RealESRGAN_x4plus")` returns `RealESRGAN_x4plus`
  — the mode is ignored;
- `resolve_denoise(CREATIVE, 1.0, …)` returns `1.0` — `CREATIVE_DENOISE = 0.25`
  is ignored.

Both planners behave correctly; explicit fields are *meant* to win. But the UI
always supplies them, so **from the UI, Creative changes neither the model nor
the denoise.** It is recorded in metadata and shown in history while changing
nothing. `CREATIVE_DENOISE` is effectively unreachable.

This has **not** been confirmed end-to-end through a running job — it is a
static trace of `useEnhanceJob.buildSettings` → `parse_options` →
`resolve_denoise`. Confirming it is the first task of Phase 4.

### D3 — Sharpening is off by default and has never been benchmarked

`sharpenStrength: 0`, and `buildSettings` omits the field entirely at 0. The
implementation is the most carefully designed post-process in the codebase, and
**not one phase has measured it**. Every benchmark from Phase 2 onward pinned
`sharpen_strength = 0.0` — correctly, to isolate other variables, but the
consequence is that the one detail-recovery lever that already exists is both
disabled and unmeasured.

The product complaint is "detail does not look real". The most likely single
cause is D1 (detail is being denoised away), and the most likely unexplored
remedy is D3.

### D4 — Not the bottleneck, on evidence

Model architecture (A5, A14), preprocessing (A10, A11), and chroma handling
(A11) have each been tested and rejected. Further work in those areas is
unlikely to pay.

## E. Next quality opportunities, ranked

Scored 1 (low/good) to 5 (high/bad) for cost and risk.

| # | Direction | Expected gain | Complexity | Runtime | VRAM | Hallucination | Artifact | Licence | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **E1** | Verify D2, then correct the denoise default and mode wiring | **High** | 1 | 1 (none) | 1 (none) | 1 | 1 | 1 | Perfect |
| **E2** | Benchmark sharpening and choose a default | **Medium-high** | 1 | 2 (post-process only) | 1 | 2 | 3 (halos) | 1 | Perfect |
| **E3** | Tile-size quality invariance | Medium (diagnostic) | 1 | 2 | 1 | 1 | 2 | 1 | Perfect |
| **E4** | Licence-audit community RRDBNet fine-tunes | Unknown, possibly high | 2 | 1 (desk work) | 1 | 3 | 3 | **5** | Drop-in |
| **E5** | Output encoding defaults (JPEG q=92) | Low-medium | 1 | 1 | 1 | 1 | 2 | 1 | Perfect |

**E1** is first on every axis: highest expected gain, no new code paths, no new
compute, and the supporting measurements already exist. **E4** is the only
direction where a genuine model gain might still be found — the loader already
builds RRDBNet, so a permissively-licensed photographic fine-tune costs no
architecture change — but licence risk is the blocker and must be resolved by
verification, not assumption.

## F. Phase 4 proposal

Four small hypothesis-driven experiments. **Not a factorial matrix.** Total
compute is roughly one Phase 3C pilot.

---

### F1 — Is Enhancement Mode inert?

- **Hypothesis.** A job submitted through the UI with mode Creative produces
  byte-identical output to the same job with mode Standard, because both the
  model and the denoise are overridden by explicit fields.
- **Justification.** D2, traced statically through the production source.
- **Experiment.** Submit through the **job API** (not the service layer) two
  jobs on one corpus image: `mode=standard` and `mode=creative`, with the exact
  field set `buildSettings` produces. Compare output bytes and the recorded
  `model` / `denoise` in the job row.
- **Inputs.** One photograph — `portrait-skin`. One is enough for a binary
  question.
- **Controls.** A third job with `mode=creative` and **no** `model`/`settings`
  fields, which should select `realesr-general-x4v3` at 0.25.
- **Metrics.** Byte equality; recorded model and denoise per job.
- **Visual.** None needed — this is a wiring question.
- **Success (hypothesis confirmed).** Arms 1 and 2 identical, arm 3 different.
- **Rejection.** Arms 1 and 2 differ → D2 is wrong, and the rest of F1 is
  withdrawn.
- **Cost.** ~3 jobs, under 5 minutes.
- **Productionisable.** Yes — the fix is a frontend change (stop sending
  `denoiseStrength` when the user has not moved the slider, or have `setMode`
  set the model and denoise). **The fix is not part of Phase 4.**

### F2 — What denoise default should ship?

- **Hypothesis.** 0.25 is visibly better than 1.0 on portrait and text, and not
  visibly worse on the low-light image.
- **Justification.** A1–A3. Phase 2.5 recommended 0.25 on metrics and a visual
  pass, but that pass predates the blind protocol established in 3B/3C, and its
  own report flagged that lettering was never visually inspected.
- **Experiment.** `realesr-general-x4v3` at denoise **{0.00, 0.25, 0.50, 1.00}**,
  5 photographs, 4x, PNG, sharpening 0, production tiling. 20 cells.
- **Controls.** `x4plus` (no denoise control) as an external reference.
- **Metrics.** The existing set; read **after** the visual pass.
- **Visual protocol.** The Phase 3C blind protocol verbatim: shuffled panels,
  mapping withheld until each verdict is written, inspection at 1:1, normal
  size and ~12×, on the existing face/foliage/text/low-light/landscape regions.
- **Success.** 0.25 wins or ties on ≥4 of 5 photographs at normal viewing size,
  with no category clearly worse.
- **Rejection.** 0.25 is clearly worse anywhere, or indistinguishable from 1.0
  everywhere — in which case the default is not worth changing.
- **Cost.** ~20 cells, mostly `general-v3` at 2–5 s. Under 15 minutes.
- **Productionisable.** Yes — a one-constant change.

### F3 — Does sharpening recover perceived detail without artifacts?

- **Hypothesis.** A non-zero sharpening default improves perceived micro-detail
  on skin, hair and rock without visible halos, because the implementation
  already has a dead zone (noise floor 2.0) and a ceiling (10.0) designed to
  prevent exactly those failures.
- **Justification.** D3 — the lever exists, is off, and has never been measured.
  It is the only remaining untested component of the current pipeline.
- **Experiment.** `x4plus` at 4x, sharpening **{0.0, 0.25, 0.5, 0.75, 1.0}**,
  5 photographs. 25 cells. Sharpening is a post-process, so the SR pass can be
  computed **once per image** and the five strengths applied to the cached
  result — 5 SR passes, not 25.
- **Controls.** 0.0 is the current default and the baseline for every delta.
- **Metrics.** `edge_overshoot` is the primary guard (halo proxy);
  `local_contrast` and `hf_ratio` as the intended gain; `flat_noise` to detect
  grain amplification.
- **Visual protocol.** Blind, as F2. **Explicitly inspect for**: bright/dark rims
  on the sign lettering and the snow/rock boundary; grain in the low-light dark
  band; over-crisp pores on skin.
- **Success.** A strength exists where local contrast rises visibly on ≥3
  photographs with no visible halo and no visible grain increase.
- **Rejection.** Every strength that helps also produces a visible rim or grain
  → keep 0.0 and record that the default is correct.
- **Cost.** 5 SR passes + 25 cheap post-processes. Under 15 minutes.
- **Productionisable.** Yes — a default change, no new code.

### F4 — Does the OOM tile reduction cost quality?

- **Hypothesis.** `x4plus` output at tile 128 is visually indistinguishable from
  tile 256 on the same image.
- **Justification.** A17 — the baseline silently ran at tile 128 on 4 of 5
  images, so most Phase 3/3C baseline measurements are of the *reduced*
  configuration. And A14 proved tiling **can** cost quality: HAT's seam.
- **Experiment.** One photograph that fits at tile 256 (`landscape-detail`),
  `x4plus`, forced tile ∈ {256, 128, 64}. 3 cells.
- **Controls.** Tile 256 is the reference.
- **Metrics.** Per-row and per-column difference profiles at expected tile
  boundaries — the detector that **did** find HAT's seam. Plus the standard set.
- **Visual.** Direct pairwise inspection at the boundaries, at ~5×.
- **Success (hypothesis confirmed).** No visible seam and metric deltas within
  ±2 % → the OOM ladder is quality-neutral and prior measurements stand.
- **Rejection.** A seam or a systematic delta appears → **prior Phase 3/3C
  baseline numbers are partly a tiling artifact** and must be re-qualified.
- **Cost.** 3 cells, under 5 minutes.
- **Productionisable.** Not directly; it validates or invalidates existing
  evidence, and could motivate a larger `tile_pad`.

### F5 — Licence audit (desk research, no compute)

- **Hypothesis.** At least one photographic RRDBNet-23 fine-tune exists under a
  permissive, commercially usable licence.
- **Justification.** E4. The loader already builds RRDBNet with `x4plus`'s exact
  `arch_params`, so such a model is a **drop-in with zero architecture change** —
  the cheapest possible place a real quality gain could still exist. Phase 3C
  excluded this class on unverified licences, which is a gap, not an answer.
- **Experiment.** No benchmark. For each candidate, record from the **source**:
  licence text and URL, author, architecture and `arch_params`, training data
  and degradation model, and whether the licence permits commercial use.
- **Success.** ≥1 candidate with a verified permissive licence → propose a
  single drop-in benchmark arm for a later phase.
- **Rejection.** All candidates non-commercial or unstated → close the direction
  and record it, so it is not reopened on vibes.
- **Cost.** Desk work. No download, no compute.
- **Productionisable.** Only after a benchmark that does not exist yet.

---

## G. Model strategy

**Keep `x4plus` as primary. Keep `general-v3` as Creative. Pause architecture
research. Pursue only the licence audit (F5).**

- Two independent comparisons (Phase 3, Phase 3C) both concluded `x4plus` is the
  best available. That is a stable result.
- `general-v3` remains the right Creative basis: it is the only arm that beat
  the reference on detail anywhere, at 4.1× the speed and 1/17 the VRAM — but
  **only once F1/F2 make Creative actually do something.**
- Do not investigate another architecture. HAT was the strongest realistic
  candidate and it lost on quality *before* its 5.35× cost was considered. The
  4 GB card is the binding constraint and no architecture research changes that.
- Community fine-tunes are the one open avenue, gated on F5.

## H. Quality pipeline — where to focus

```
input → validation → [neural upscaling] → resize-to-target → [sharpening] → encoding
                            ▲                                      ▲
                     A5/A14: settled                      D3: never measured
```

| Stage | Focus? | Why |
|---|---|---|
| Preprocessing | **No** | Tested and rejected twice (3A, 3B) |
| Neural model | **No** | Settled by two comparisons; pause pending F5 |
| **Adaptive parameter selection** | **Yes — first** | Not "adaptive" in the 3A sense (refuted) but *correct static defaults actually reaching the pipeline*: F1, F2 |
| **Postprocessing (sharpening)** | **Yes — second** | The only unmeasured component of the current pipeline: F3 |
| Artifact suppression | No | No artifacts found in production output across ~150 benchmark cells; the only one seen was HAT's, and HAT is rejected |
| Detail recovery | Indirectly | F2 (stop removing it) and F3 (enhance what survives) *are* the detail-recovery work |
| Output encoding | Later | F5-adjacent; PNG is lossless and JPEG q=92 is untested but low-leverage |
| Tiling | **Diagnostic** | F4 — validates the evidence base rather than improving output |

**The next research should focus on postprocessing and on defaults, not on
models or preprocessing.**

## I. Commercial readiness

Not legal advice. Facts and open questions are separated deliberately.

### Verified

- **Real-ESRGAN** — BSD-3-Clause. All four production weights
  (`x4plus`, `x4plus_anime_6B`, `x2plus`, `general-x4v3` + `wdn` pair) come from
  this project.
- **HAT** — Apache-2.0; **SwinIR** — Apache-2.0; **DRCT** — MIT. Verified from
  the source repositories during Phase 3. None is used in production; HAT exists
  only as a research-only vendored file under `benchmarks/`.
- **CodeFormer / RestoreFormer** — non-commercial research licences. Excluded.
- **Dependencies** — no new production dependency has been added by 3A, 3B or
  3C. `basicsr`, `einops`, `timm`, `facexlib` and `gfpgan` are all absent and
  stay absent.
- **Corpus** — every image carries a recorded Commons licence; two are
  CC BY-SA 4.0, so derived crops inherit share-alike, which the crop
  `ATTRIBUTION.md` files state.

### Still requires verification

1. **The upstream Real-ESRGAN *weights*** are distributed under the repository's
   BSD-3 licence, but the weights' own terms and the licence of their training
   data have not been independently checked. This is the most material open
   question, because these weights are the product.
2. **Any community fine-tune** — F5 exists precisely to resolve this, and no
   such model should be benchmarked before its licence is verified at source.
3. **The vendored `benchmarks/arch/hat.py`** is Apache-2.0 and research-only. If
   HAT were ever adopted, its NOTICE/attribution obligations would need proper
   handling in `app/inference/arch/`. Not applicable while it stays in
   `benchmarks/`.
4. **The HAT checkpoint provenance** — obtained from a third-party HuggingFace
   mirror because the official release is Google-Drive only. The sha256 was
   verified against an independent listing, but the mirror's right to
   redistribute was not. Research-only; would matter if ever productionised.
5. **Corpus images are not committed**, by design. Anyone reproducing the
   benchmarks re-fetches them under their own attribution obligations.

## J. Final recommendation

**Run F1 first, alone, before anything else.**

If Enhancement Mode is inert, then Creative — a shipped, user-visible feature —
changes nothing about the image, and **every denoise-capable job is running at
1.0, the setting Phase 2.5 measured as costing 53 % of high-frequency detail and
Phase 3 measured at −88.4 % hf on portrait skin.**

That would mean the product's "detail does not look real" complaint has a
largely non-neural cause that three phases of model and preprocessing research
could not have found, and that the fix is a small frontend change supported by
measurements already in the repository.

F1 costs three jobs and under five minutes. It either redirects the whole
roadmap or is cheaply eliminated. Nothing else should start until it is
answered.

Then, in order: **F2** (choose the denoise default), **F3** (measure the one
untested lever), **F4** (validate the evidence base), **F5** (desk work, can run
in parallel).

**Do not resume model research.** Two comparisons agree, and the hardware is the
constraint.

---

### Standing methodology, carried forward

Established across 3A–3C and non-negotiable for Phase 4:

1. **Visual evaluation is a first-class gate and comes first.** Phase 3B's
   `hf_ratio` rose while the image got grainier; Phase 3C's automatic detectors
   missed a plainly visible seam. Metrics are read second.
2. **Blind, with the mapping withheld** until each verdict is written.
3. **Every arm runs through the production path** at a recorded configuration.
   Precision, tile size and reduction are recorded per run; a candidate that
   cannot run at production precision is flagged, not quietly compared.
4. **Negative results are results.** Three phases in a row ended in rejection and
   each narrowed the search usefully.
5. **Normal-viewing-size significance.** A difference visible only at 12× is not
   a product improvement.
6. **No fabricated numbers.** Incomplete experiments are reported incomplete.
