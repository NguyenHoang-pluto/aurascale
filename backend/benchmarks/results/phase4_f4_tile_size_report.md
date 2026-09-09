# Phase 4 · F4 — does tile size change the picture?

**Research only. Production was NOT changed** — `tile_size` remains 256,
`tile_pad` remains 16, and the out-of-memory ladder in
`app/inference/upscaler.py` was neither disabled nor bypassed. No file under
`backend/app`, `frontend/src`, `models/` or `pyproject.toml` was modified.
Nothing committed, nothing pushed. F5 was not started.

---

## 1. Objective

Production tiles at 256 and silently reduces that tile when VRAM is short. The
ladder is a reliability feature and it is doing its job, but **no benchmark has
ever asked what it costs the picture.** On a 4 GB card the reduction is not
hypothetical: it is the normal path for large inputs.

The question this phase answers is narrow and it is the one the ladder makes
urgent. When production drops from tile 256 to tile 128, or from 128 to 64, does
the user get a worse image, and if so, worse in what way and by how much?

The governing rule for this phase: **a tile arm is evidence about its tile size
only if the tile it asked for is the tile that ran.** An arm that was clamped or
fell back to CPU is a different experiment wearing the same label.

## 2. Exact matrix

**5 photographs × 3 tile sizes = 15 cells. 15 completed, 0 failed, 0 excluded.**

| | |
|---|---|
| Model | `RealESRGAN_x4plus` |
| Requested tiles | **256, 128, 64** |
| Scale | 4x, single native pass |
| Output | PNG, lossless |
| Sharpening | 0.0 |
| Denoise | **structurally absent** — `x4plus` declares no `denoise_pair` |
| Tile pad | production 16, held constant across every arm |
| Precision | fp16, production policy |
| Path | `RealEsrganUpscaler.upscale`, the same call `enhancement_service` makes |
| Reference arm | `t256`, the production default |

`tile_pad` is deliberately not varied with the tile. The experiment varies tile
size; varying the pad alongside it would confound the two.

**All 15 arms were honoured.** Every cell recorded `requested_tile ==
effective_tile`, `tile_reduced = false`, `fell_back_to_cpu = false`, and
`device = cuda`. No cell needed to be excluded, so every number below is
evidence about the tile size on its label.

## 3. The determinism control

The sweep is uninterpretable without it and the harness refuses to run without
it. Recorded earlier in an identical environment, and preserved unchanged in
the measurements file:

| arm | requested → effective | passes byte-identical | max_abs |
|---|---|---|---|
| `t256` | 256 → 256, 256 | yes | 0 |
| `t128` | 128 → 128, 128 | yes | 0 |
| `t064` | 64 → 64, 64 | yes | 0 |

All three arms are bit-exact across two runs, with `sha256_first ==
sha256_repeat` computed from the same arrays the pixel comparison reads. **Every
difference reported below is therefore caused by the tile size and not by run-to-run
variance.** That is the whole point of the control, and it is what makes the
"differs" verdicts in §6 admissible.

Repeatability required one non-obvious step. `release_cuda_memory` is called
before each pass, because the caching allocator otherwise still holds the
previous pass's blocks and `free_vram_mb` cannot see them as free — which had
silently clamped a second tile-256 pass to a smaller tile and would have been
recorded as a repeatability failure. The helper is production's own; nothing in
`app/` was changed to obtain this.

## 4. Methodology

Every pass goes through the production upscaler, so the OOM ladder, the VRAM
clamp and the stitcher are the real ones. Free VRAM is sampled and recorded
before each pass, after the release, so the tile each arm was granted can be
audited rather than assumed.

Seams are scored by a matched filter for what a tile joint actually is: a
full-height discontinuity. Mean absolute luma step is taken across every column
boundary and reduced to a 1-D profile, then each boundary is scored as a robust
z against its own 24-px neighbourhood, median and MAD rather than mean and
standard deviation so one genuine image edge nearby cannot flatten the score.

The design that makes this readable: the three seam grids are **exclusive**, and
every arm is scored on all three. The `t256` grid is the multiples of 1024, the
`t128` grid the multiples of 512 that are not multiples of 1024, the `t064` grid
the multiples of 256 that are neither. A grid is a real joint for the arm that
owns it and every finer arm, and **ordinary image content for every coarser
one** — so the coarser arms are a built-in control that measures what the metric
reads at coordinates that merely look like seam coordinates.

## 5. Runtime and memory

| photograph | t256 | t128 | t064 |
|---|---|---|---|
| landscape-detail | 30.3 s, 35 tiles | 28.0 s, 140 tiles | 67.1 s, 532 tiles |
| portrait-skin | 27.4 s, 36 tiles | 38.9 s, 132 tiles | 82.1 s, 504 tiles |
| text-signage | 24.9 s, 35 tiles | 48.0 s, 130 tiles | 83.9 s, 520 tiles |
| foliage-texture | 24.9 s, 35 tiles | 48.6 s, 130 tiles | 86.3 s, 520 tiles |
| low-light-noise | 28.1 s, 40 tiles | 49.5 s, 135 tiles | 90.7 s, 510 tiles |
| **median** | **27.4 s** | **48.0 s** | **83.9 s** |
| **peak VRAM** | **767 MB** | **260 MB** | **115 MB** |

Tile 256 is the fastest arm on four of the five photographs and tile 64 costs
roughly three times its median runtime. The exception is `landscape-detail`,
where `t128` finished in 28.0 s against 30.3 s for `t256` — the only cell in the
sweep where a smaller tile was quicker, and small enough to be scheduling noise
rather than a pattern.

Peak VRAM is the only axis on which the smaller tiles win consistently, and it
is the axis the ladder exists to serve. The trade is real and it is not
symmetric: dropping to tile 128 saves 507 MB and costs 75% more time at the
median.

## 6. The arms are not byte-identical

Every arm differs from `t256`. None is a bit-exact match.

| photograph | arm | max_abs | subpixels differing | PSNR | SSIM |
|---|---|---|---|---|---|
| landscape-detail | t128 | 98 | 23.04% | 46.7 dB | 0.9949 |
| landscape-detail | t064 | 101 | 47.52% | 41.9 dB | 0.9864 |
| portrait-skin | t128 | 68 | 32.45% | 45.9 dB | 0.9905 |
| portrait-skin | t064 | 105 | 61.39% | 40.4 dB | 0.9651 |
| text-signage | t128 | 98 | 22.18% | 46.1 dB | 0.9861 |
| text-signage | t064 | 137 | 42.67% | 42.0 dB | 0.9699 |
| foliage-texture | t128 | 45 | 20.62% | 50.4 dB | 0.9983 |
| foliage-texture | t064 | 70 | 46.61% | 45.3 dB | 0.9944 |
| low-light-noise | t128 | 60 | 18.63% | 49.8 dB | 0.9966 |
| low-light-noise | t064 | 80 | 42.44% | 44.6 dB | 0.9920 |

The ordering is consistent across this table: `t064` is further from `t256` than
`t128` is, on all five photographs and on all four of these pixel metrics. That
consistency belongs to *distance from the reference* and does not extend to
every measurement in the phase — the seam z-scores in §7 are not monotonic in
tile size, as §10 records. All 15 outputs have distinct SHA-256 digests.

**The fraction is the misleading number here.** A fifth to a half of all
subpixels differ, which sounds decisive, but PSNR stays at 40–50 dB and SSIM
above 0.965. That combination describes a very large number of very small
differences, which is what a different tile partition produces: the network sees
a different receptive-field context for most output pixels and lands a fraction
of a level away almost everywhere. §8 is where that gets adjudicated.

## 7. Seams exist and the metric locates them exactly

Median robust z per grid. **own** = the grid is a real tile joint for this arm;
**ctl** = the same coordinates are ordinary image content for this arm.

| photograph | arm | t256 grid | t128 grid | t064 grid |
|---|---|---|---|---|
| landscape-detail | t256 | +1.61 own | +0.07 ctl | +0.08 ctl |
| landscape-detail | t128 | +1.85 own | +1.99 own | +0.07 ctl |
| landscape-detail | t064 | +1.58 own | +1.83 own | +1.83 own |
| portrait-skin | t256 | +7.48 own | −0.22 ctl | −0.41 ctl |
| portrait-skin | t128 | +7.66 own | +4.30 own | −0.40 ctl |
| portrait-skin | t064 | +8.15 own | +4.76 own | +6.26 own |
| text-signage | t256 | +5.57 own | +0.56 ctl | −0.11 ctl |
| text-signage | t128 | +5.94 own | +8.02 own | +0.00 ctl |
| text-signage | t064 | +6.53 own | +8.90 own | +4.60 own |
| foliage-texture | t256 | +2.48 own | +0.38 ctl | +0.28 ctl |
| foliage-texture | t128 | +2.26 own | +1.75 own | +0.28 ctl |
| foliage-texture | t064 | +2.38 own | +1.69 own | +1.56 own |
| low-light-noise | t256 | +1.40 own | −0.73 ctl | +0.03 ctl |
| low-light-noise | t128 | +1.34 own | +0.27 own | +0.05 ctl |
| low-light-noise | t064 | +1.54 own | +0.31 own | +1.36 own |

The comparison that carries weight is the **paired** one — same photograph, same
grid, same axis, arms that own the joint against arms for which those columns are
ordinary content. **All 20 such comparisons are clean: every owning arm reads
above every non-owning arm, without exception.** The metric is measuring tile
joints, not image content that happens to live at round coordinates.

Stated globally the separation is not perfect, and the reason is worth keeping.
Across all 90 cells on both axes — 60 own, 30 control — control cells span −0.73
to +0.73 and own cells span +0.27 to +8.90, so **4 of the 60 own cells fall
inside the control range**: `low-light-noise` t128 and t064 on the t128 grid
(+0.27, +0.31, vertical) and `foliage-texture` t128 and t064 on the t256 grid
(+0.71, +0.65, horizontal). These are not metric failures. They are photographs
where that particular joint is genuinely weak, and a weak real seam on one image
can score below an ordinary-content reading on a busier one. Seam strength is a
property of the content, not a constant of the tile size, which is why the
paired form is the one to read.

### The z-scores are significant and the steps are tiny

Statistical significance and visual significance are different questions, and
here they give opposite answers. Converting the same cells to physical units:

| photograph | arm | grid | z_median | joints above 3σ | step at joint | local baseline | **excess** |
|---|---|---|---|---|---|---|---|
| portrait-skin | t256 | t256 | +7.48 | 100% | 2.731 | 1.752 | **+0.98 levels** |
| portrait-skin | t064 | t064 | +6.26 | 90% | 2.513 | 1.666 | **+0.85 levels** |
| text-signage | t064 | t128 | +8.90 | 83% | 1.463 | 1.102 | **+0.36 levels** |
| text-signage | t128 | t128 | +8.02 | 83% | 1.495 | 1.110 | **+0.39 levels** |
| text-signage | t064 | t064 | +4.60 | 77% | 1.656 | 1.247 | **+0.41 levels** |
| landscape-detail | t128 | t128 | +1.99 | 29% | 3.033 | 2.657 | **+0.38 levels** |

Across all own-seam cells the excess step ranges from **+0.23 to +0.98 8-bit
levels**. The largest tile seam in the corpus, measured as a median over its
joints, is **under one level out of 255.**

Portrait-skin scores highest precisely because it is smoothest. Its local
baseline is 1.75 levels, so the MAD is small and a consistent sub-level step
dominates the ratio. **A z of +7.48 on this image is a statement about how flat
the neighbourhood is, not about how big the seam is.** That is a real risk of
reading the z column alone and it is why §8 exists.

## 8. Visual results

Seam crops are placed on the **worst** measured joint of each grid, not a
typical one, and the band along the joint is chosen where the seam crosses the
most structure. Choosing the crop by measurement removes the obvious way to
cheat at this — picking a flat region where no stitcher could fail.

### Tile 64 produces a plainly visible tiling artifact

On `portrait-skin`, arm `t064`, the artifact is not subtle and does not need a
trained eye. A rectangular block is visible: fine skin texture inside it, a
distinct softening outside it, and hard boundaries on **two** axes.

The geometry is exactly where tiling predicts, which is what makes it a tile
artifact rather than image content. The crop origin is (1600, 2114); tile 64 at
4x puts joints every 256 output px; the visible corner falls at absolute
x = 1792 and y = 2304, both exact multiples of 256.

| arm | horizontal step at y=2304 vs neighbours | vertical step at x=1792 vs neighbours |
|---|---|---|
| t256 | 1.03× | 1.35× |
| t128 | 1.04× | 1.37× |
| **t064** | **2.43×** | **2.56×** |

Texture energy confirms the block is anomalous rather than the content being
naturally split there. Below-left of the corner `t064` reads 6.65 against 1.04
immediately across the vertical joint — a 6.4× jump, where `t256` reads 2.3×
and `t128` 2.7× at the identical coordinates. The same block reads 6.65 against
1.93 across the horizontal joint, a 3.5× jump, against 1.4× and 1.3× for the
coarser arms.

Neither coarser arm shows a comparable discontinuity here. Their readings are
not 1.00× — a real image edge does run near this column, which is why the
vertical figures sit at 1.35× and 1.37× rather than at unity — but they stay
within about a third of their own local baseline on the vertical axis and within
4% of it on the horizontal. `t064` is more than double its baseline on both.
**This is a tile-64 defect.**

### Tile 128 leaves a faint seam on flat low-contrast fields

On `text-signage` at the worst `t128` joint, viewed at 1:1, the dark panel shows
a subtle tonal step running the full height of the panel in the `t128` and
`t064` arms. `t256` is continuous across the same columns.

Calling this visible requires qualification. It is detectable on a flat, dark,
low-contrast field when the location is known in advance, in a 160-px-wide strip
of a 6528-px-wide image. I could not find it without the measurement pointing at
it, and at normal display size it is not there. **Measurable, locatable,
and not a defect a user would encounter.**

### Tile size changes synthesized texture, direction depending on content

The clearer `t128` finding on this photograph is not the seam. In the same dark
panel, `t256` renders fine crackle texture across the whole field and `t128`
and `t064` largely smooth it away. This is visible at 1:1 without knowing where
to look, and it is not a boundary effect — it holds on both sides of the joint.

The global metrics agree and quantify it. On `text-signage`, high-frequency
ratio falls from 0.0025 at `t256` to 0.0015 at both smaller arms, a 40% loss,
with `sobel_mean` falling 0.0466 → 0.0437 → 0.0419.

**The direction is content-dependent and reverses.** On `portrait-skin` the
smaller tile *adds* high-frequency energy — 0.0006 → 0.0007 → 0.0010 — because
what it is adding is the artifact documented above, not detail. Smaller tiles do
not uniformly soften; they change what the network invents in ambiguous regions,
and both losing real texture and inventing false structure are failure modes.

### Ordinary content is visually equivalent, on the regions inspected

Away from joints, the arms are indistinguishable to me on the two detail regions
I examined. The `text-signage` crop — roofline, sky gradient, dark panel,
windows — reads identically across `t256`, `t128` and `t064`. The
`portrait-skin` cheek crop is likewise equivalent, with no difference in pore
texture or tonal rendering I can name.

**This covers two of the five detail regions.** The `landscape-detail`,
`foliage-texture` and `low-light-noise` texture crops were written by the
harness and are in the crop directory, but I did not inspect them, so the
finding is not established for those photographs.

This is the honest resolution of §6, with that scope attached. A fifth to a half
of subpixels differ, and on the ordinary content I looked at, none of that
difference is perceptible. Note that this is a judgment reached by looking, not
one PSNR and SSIM could have delivered on their own: those numbers say the two
frames are close, not that a viewer cannot tell them apart, and §8's texture
finding is a case where a high SSIM sits alongside a difference I could see.

## 9. Classification of the differences

The phase was required to separate three causes. All three are separable here
and only one of them is present.

**A — deterministic differences caused by tile partitioning.** This is the
entire result. The determinism control is bit-exact, so every difference in §6
and §7 is attributable to the partition. It divides into two effects: seams at
grid coordinates, sub-level in magnitude except at tile 64, and a global change
in synthesized texture whose sign depends on content.

**B — differences caused by the OOM ladder or a reduced effective tile.**
**None. Zero cells.** All 15 arms were honoured, so no result in this report is
contaminated by a clamp or a CPU fallback, and no arm here is being read as
evidence about a tile size it did not run at.

**C — differences caused by unrelated runtime state.** **None detectable.** The
determinism control returns `max_abs = 0` on all three tile sizes on the
noisiest photograph in the corpus, which is the hardest case for reproducibility.
Run-to-run variance is therefore not merely small on that control, it is zero.
The control is one photograph and two repeats per arm, so this bounds run-to-run
variance rather than proving it can never appear; it is enough to attribute the
sweep's differences to the partition, which is what it was run for.

## 10. Conclusion

**Tile size is not a free parameter, and the ladder is not free.**

- Tile 256 and tile 128 are **visually equivalent in ordinary content**. The
  numerical difference between them is real, deterministic and imperceptible.
- Tile 128 costs **75% more runtime** than tile 256 and leaves a seam that is
  measurable but that I judge not visible in use.
- Tile 128 loses real synthesized texture on low-contrast fields — a 40%
  high-frequency drop on `text-signage`, visible at 1:1. This is the strongest
  argument against the first rung of the ladder, and it is a quality cost, not
  a seam.
- **Tile 64 is a visible defect.** It produces rectangular artifacts with hard
  boundaries on both axes at exactly the tile grid, and it costs three times the
  median runtime of tile 256. It is slower than both coarser arms on every
  photograph, and further from `t256` than `t128` is on every photograph and all
  four pixel metrics. **It is not uniformly worse on the seam metric**: on 10 of
  the 20 shared-grid comparisons `t064`'s median z sits slightly *below*
  `t128`'s, for example `text-signage` horizontal on the t128 grid at +4.03
  against +4.40. The case against tile 64 rests on the visible artifact and the
  runtime, not on a clean sweep of every measurement.

The ladder's first step is a reasonable trade. Its lower rungs are not a
graceful degradation, and tile 64 should be regarded as a last resort that
visibly damages the output rather than as a smaller-but-equivalent path.

**No production change is proposed by this report.** F4 was scoped to measure,
and integration was explicitly out of scope.

## 11. Confidence and limitations

- **Five photographs, one model, one scale, one card.** `RealESRGAN_x4plus` at
  4x on an RTX 3050 Laptop. Nothing here transfers to other models or scales
  without measurement.
- **Tile 256 held on this hardware but not with much room.** Free VRAM before
  the `t256` passes ranged 2191–3299 MB. `recommended_tile_size` drops the
  ceiling to 192 at or below 2048 MB, so the narrowest margin observed was
  **143 MB**. Tile 256 was honoured in all five sweep cells and both determinism
  passes, on an otherwise idle GPU. Another consumer taking half a gigabyte
  would silently clamp it, and the recorded `free_vram_before_mb` is what makes
  that auditable rather than assumed.
- **The visual judgments are mine and are single-observer.** The tile-64
  artifact is unambiguous and geometrically verified. The tile-128 seam call —
  measurable, not visible in use — is a judgment, and it is the one most worth
  a second observer.
- **Seam crops are worst-case by construction.** They are placed on the most
  conspicuous joint of each grid, so they overstate the typical joint. The
  median joint is the §7 table, not the pictures.
- **PSNR is between two arms, neither of which is ground truth.** It is reported
  because readers expect it, not because it is the right tool here.
- **The provenance block describes the sweep run.** The determinism rows were
  produced about three hours earlier and are preserved unchanged; the
  interpreter, torch build, GPU and all five corpus hashes were verified
  identical across both, so the single provenance block is accurate for every
  row in the file.

## Reproducibility

```
cd backend
.venv/Scripts/python -m benchmarks.phase4_f4 --determinism   # already recorded
.venv/Scripts/python -m benchmarks.phase4_f4 --sweep
```

`--sweep` refuses to run unless determinism rows are present. Environment:
Python 3.11.9, torch 2.7.1+cu118, CUDA on, cuDNN deterministic off, benchmark
off, TF32 on, `NVIDIA GeForce RTX 3050 Laptop GPU`. All five corpus SHA-256
digests verified against the manifest at report time.

Measurements: `phase4_f4_measurements.json` — 28 rows, 3 determinism, 15 sweep,
10 seam-crop. Crops: `phase4-f4-crops/`, 45 images.

The three determinism rows serialize `psnr_db` as the bare token `Infinity`,
which is correct — the arms are byte-identical, so RMSE is zero — but is not
standard JSON. Python's `json` reads it back natively and the file already
survived a full load-and-save cycle when the sweep was appended, so neither the
benchmark nor this report is affected. A non-Python consumer such as `jq` or
`JSON.parse` would reject the file. No sweep row is affected, because every
sweep arm differs from its reference.

Tests: `tests/unit/test_benchmark_tile_quality.py`, 52 passed. Full unit suite
639 passed, 2 skipped. `backend/app` and `frontend` unmodified.
