# Phase 5 — can detail be *recovered* after Real-ESRGAN, or only amplified?

**Research only. Production was NOT changed.** No file under `backend/app`,
`frontend/src`, `models/` or `pyproject.toml` was modified. The sharpening
default remains 0, the denoise default is untouched, model routing is untouched,
`tile_size` remains 256. Nothing committed, nothing pushed. Phase 6 was not
started.

---

## 1. Objective

F3 established the lesson this phase is built on: **high-frequency energy can
rise while the picture gets worse.** An unsharp mask raises Sobel, HF ratio and
local contrast whether it is amplifying texture, sensor noise or a JPEG block
edge, so "the metric went up" is not evidence of anything.

The question here is whether a post-process can raise detail *where detail
exists* and leave noise alone. That is a claim about selectivity rather than
gain, and answering it needed two things F3 did not have: a reference against
which "recovered" and "invented" are different measurements, and proof that the
candidates are not the unsharp mask wearing a new name.

## 2. What was tested

Five arms. `A` and `B` are controls rather than proposals — `B` is F3's
recommended value, carried in so new ideas are measured against the best thing
the codebase already has.

| arm | what it is | why it is not `B` |
|---|---|---|
| `A-baseline` | neural output, untouched | — |
| `B-sharpen025` | production `unsharp_mask` at 0.25 | the incumbent |
| `C-guided` | boost the detail layer of a guided-filter decomposition | base is edge-preserving, not Gaussian; signal-dependent through the variance term |
| `D-gated` | `C`'s detail layer, gain multiplied by structure-tensor coherence | spatially varying gain; isotropic signal is gated off |
| `E-dog` | difference-of-Gaussians band boost | occupies a band, not a half-plane; finest scale excluded |

`B` calls production's own `unsharp_mask`, so nothing is reimplemented and any
difference between arms is a difference between operators.

### The arms were calibrated to equal benefit before any photograph was seen

The first characterisation run exposed a problem that would have produced a
flattering and worthless result. At their initial settings `C` and `D` had a
mid-band gain of 1.031 against `B`'s 1.191 — they barely acted. Comparing them
to `B` then would have shown they were "safer", which is trivially true of an
operator that does almost nothing, and would have tested a badly chosen constant
rather than the idea.

Every candidate gain was therefore solved by bisection so that **all four arms
deliver a mid-band gain of 1.191**, matching `B`. With benefit held equal, the
only thing left to differ is cost. The calibration was done on synthetic signals
alone, before any photograph was processed, and no parameter was revised after
seeing a picture result.

The guided filter's `eps` moved from 4.0 to 200 in the same pass, for a reason
worth recording: at `eps = 4` the filter treats ordinary texture as an edge and
keeps it in the base layer, so the detail layer holds almost nothing but noise.

## 3. Exact matrix

**Three tracks. 5 operator signatures, 5 photographs × 5 arms twice = 50 picture
cells. 55 cells completed, 0 failed, 0 excluded.**

| | |
|---|---|
| Model | `RealESRGAN_x4plus` |
| Scale | 4x, single native pass |
| Output | PNG, lossless |
| Sharpening | 0.0 except in arm `B`, where it is the arm |
| Denoise | structurally absent — `x4plus` declares no `denoise_pair` |
| Tiling | production tile 256 / pad 16 |
| Corpus | the tracked five, hashes verified identical to F4 |
| Neural passes | **one per photograph per track**, output cached and shared by all five arms |

Caching the neural pass is not only a saving. Because every arm post-processes
the *same* array, no arm can differ from another by model variance, tiling or
precision. All ten neural passes in the real track were honoured at tile 256
with no reduction and no CPU fallback.

## 4. Are the candidates actually different operators?

Applied to synthetic signals whose correct answer is known. This ran first and
gated the rest: an arm matching `B` on all three axes is a duplicate experiment.

| arm | step overshoot | noise gain | mid gain | fine gain | flat gain | detail per noise |
|---|---|---|---|---|---|---|
| `A-baseline` | 0.00 | 1.000 | 1.000 | 1.000 | 1.0000 | — |
| `B-sharpen025` | 4.00 | 1.254 | 1.191 | 1.175 | 1.0000 | 0.950 |
| `C-guided` | 3.00 | 1.332 | 1.191 | 1.176 | 1.0000 | 0.894 |
| `D-gated` | 3.00 | 1.088 | 1.191 | 1.176 | 1.0000 | 1.095 |
| `E-dog` | 5.00 | 1.039 | 1.192 | 1.076 | 1.0000 | 1.147 |

**No candidate is a duplicate.** At identical mid-band gain the three differ from
`B` and from each other on noise gain, on overshoot and on what they do to the
finest scale. `E` is the only arm that leaves fine detail nearly alone (1.076
against `B`'s 1.175), which is what a band-limited operator should do. No arm
shifts the level of a flat field.

On these signals `D` and `E` looked like genuine improvements on `B`: same
benefit, materially less noise. **§6 shows that prediction does not survive
contact with a photograph**, and §8 explains why.

## 5. The reference track — the only one that can decide

A corpus photograph is the ground truth. It is degraded by a fixed pipeline
(Gaussian blur σ 0.8, 4x area downsample, Gaussian noise σ 2.0, JPEG quality 85,
in that physical order), upscaled 4x by the production path, and each candidate
is applied to that output. Every arm is then scored against the original.

Parameters were fixed before any candidate was compared and were not touched
afterwards. Comparison is on a common crop, never a resample, so no arm is
scored through an extra resampling step.

### Baseline wins fidelity on every photograph

| metric | best arm, out of 5 photographs |
|---|---|
| PSNR | **`A` 5 / 5** |
| SSIM | **`A` 5 / 5** |
| edge preservation | `E` 4 / 5, `A` 1 / 5 |
| texture error | `C` 5 / 5 |

Median change against baseline:

| arm | ΔPSNR | ΔSSIM | Δedge corr. | Δtexture error |
|---|---|---|---|---|
| `B-sharpen025` | −0.305 dB | −0.00962 | −0.00180 | −0.049 |
| `C-guided` | −0.193 dB | −0.01184 | −0.00228 | **−0.212** |
| `D-gated` | −0.172 dB | −0.00941 | −0.00206 | −0.193 |
| `E-dog` | −0.381 dB | −0.00975 | **+0.00184** | +0.412 |

**Every candidate makes the reconstruction less faithful.** Not one improves
PSNR or SSIM on a single photograph.

### The split between texture error and fidelity is the finding

`C` has the **best texture error of any arm on all five photographs** while
having worse PSNR and worse SSIM than baseline on all five. Those two facts
together are the thing this phase was built to detect.

Texture error asks whether the local texture *energy* matches the reference. The
Real-ESRGAN output is too smooth, so adding detail moves that statistic toward
the truth. PSNR and SSIM ask whether the detail is in the *right places*. It is
not. The candidates add roughly the right amount of texture in the wrong
positions, which improves the statistic and degrades the picture.

**That is hallucinated detail, measured rather than asserted.** A phase without
a reference would have seen texture error fall and called it recovery.

## 6. The real-photo track

No reference exists here, so this track reports no-reference metrics and writes
the crops the eye judges. Every candidate raised Sobel, HF ratio and dark-region
HF on every photograph, exactly as F3 warned they would, and none of that is
evidence.

The number that matters is flat-region noise, because a flat region has no
detail to recover and anything gained there is noise by definition.

| photograph | `A` | `B` | `C` | `D` | `E` |
|---|---|---|---|---|---|
| landscape-detail | 0.00188 | 0.00189 | 0.00358 | 0.00342 | 0.00348 |
| portrait-skin | 0.00515 | 0.00526 | 0.00692 | 0.00670 | 0.00667 |
| text-signage | 0.00232 | 0.00235 | 0.00369 | 0.00353 | 0.00361 |
| foliage-texture | 0.00289 | 0.00293 | 0.00410 | 0.00404 | 0.00404 |
| low-light-noise | 0.00378 | 0.00382 | 0.00514 | 0.00494 | 0.00478 |

Median change against baseline:

| arm | flat-region noise | dark-region HF |
|---|---|---|
| `B-sharpen025` | **+1.0%** | +20.2% |
| `C-guided` | +41.7% | +38.4% |
| `D-gated` | +39.8% | +36.9% |
| `E-dog` | +39.9% | +19.6% |

**The production sharpener raises flat-region noise by one percent. All three
candidates raise it by about forty.** The range reaches +90% on
`landscape-detail`. This is the clearest quantitative separation in the phase
and it runs opposite to what §4 predicted.

## 7. Human visual results

Inspected at fit-to-screen and at 1:1, blind first — panel order shuffled and
recorded in the measurements file, never drawn on the image — then labelled.

**At fit-to-screen, blind, I could not order the five panels on any region.**
Consistent with F3's finding at the same viewing size.

At 1:1, by region:

| region | verdict |
|---|---|
| skin (cheek) | **EQUIVALENT.** No waxiness, no etching, no plastic look in any arm. |
| hair | **EQUIVALENT.** Strand separation identical, including in `D`, the arm built for oriented structure. Slightly more grain in the skin behind the strands. |
| eye | **EQUIVALENT.** Iris structure and catchlight unchanged. |
| text / signage | **EQUIVALENT on letterforms.** No deformation, no undershoot. `C` adds visible grain in the adjacent dark panel. |
| fine texture (rock in snow) | **EQUIVALENT.** Rock edges no better defined. Background hillside slightly more mottled under `E`. |
| foliage | **EQUIVALENT.** |
| dark / low-light | **WORSE under `C` and `D`.** Visible grain in the shadow band that the baseline does not have. |

**No candidate produced a visible improvement in any region I inspected.** The
only visible differences were costs.

Artifact classification: **grain amplification** confirmed visually in `C` and
`D` on dark regions and quantified across all three candidates. No ringing,
halo, plastic skin, block artifact, edge deformation, text deformation or colour
artifact was observed in any arm — the achromatic correction and the ceiling
appear to have done their jobs.

## 8. Metric-versus-eye and metric-versus-metric disagreements

Three, and they point the same way.

**The synthetic noise test disagreed with the photographs.** §4 measured `D` at
noise gain 1.088 and `E` at 1.039 against `B`'s 1.254, predicting both would be
gentler than the incumbent. On photographs they raise flat-region noise forty
times more than `B` does. The cause is the production sharpener's **dead zone**:
its correction is exactly zero below ±2 luma levels. The synthetic test used
σ = 6 noise, comfortably above that threshold, so it never probed the dead zone
at all. Real flat regions in a 4x output fluctuate by well under two levels, and
that is precisely the band `B` discards and the candidates amplify.

**What makes the existing sharpener safe is the dead zone, not the filter
shape.** None of the three candidates has one. That is the single most useful
thing this phase learned and it was invisible to both the synthetic harness and
any no-reference metric taken alone.

**Texture error disagreed with PSNR and SSIM**, as §5 sets out: `C` is best on
texture error on all five photographs and worse than baseline on fidelity on all
five.

**Every detail metric disagreed with the eye.** Sobel rose up to 26%, HF ratio
up to 76%, and no region looked better.

## 9. Runtime and memory

Post-process cost only. The neural pass is 21–31 s per 32 MP photograph at 767
MB peak, identical for every arm because it is shared.

| arm | median post-process, 32 MP | range |
|---|---|---|
| `A-baseline` | 0.01 s | 0.00–0.01 |
| `B-sharpen025` | 1.13 s | 1.03–1.37 |
| `C-guided` | 2.27 s | 2.18–2.36 |
| `D-gated` | 3.66 s | 3.46–3.93 |
| `E-dog` | 1.48 s | 1.42–1.60 |

Cost is not what decides this. Every candidate is cheap next to the neural pass,
and `D`, the most expensive, adds about 15%.

## 10. Is any candidate better than baseline?

**No.**

Against the eight success criteria set for the phase:

| criterion | result |
|---|---|
| 1. Visible improvement in some important region | **Fails.** None, in any region, in any arm. |
| 2. No meaningful artifacts | **Fails.** Visible grain in shadow under `C` and `D`. |
| 3. Does not amplify noise disproportionately | **Fails.** ~40% flat-region noise against the incumbent's 1%. |
| 4. Does not damage skin or identity | Passes. |
| 5. Does not damage text | Passes. |
| 6. Improves or preserves reference reconstruction | **Fails.** Every arm loses PSNR and SSIM on all five photographs. |
| 7. Benefit survives normal viewing | **Fails.** No benefit at any viewing size. |
| 8. Reasonable runtime and memory | Passes. |

Four hard failures, including criterion 6. That is the load-bearing one, because
it is the only criterion decided against ground truth rather than against
judgement — though the reference itself has limits, set out in §12.

## 11. Recommendation

### **REJECT detail recovery.**

No candidate is a candidate. `C`, `D` and `E` should not be integrated, should
not be offered as an option, and should not be revisited in their current form.

This is a negative result and it is a real one. The phase was designed so that a
negative answer would be visible rather than buried, and the reference track
delivered it unambiguously for the arms tested: **none of these three operators
recovered detail; each of them added detail in the wrong places.** With ground
truth available, every one of them moved the reconstruction further from the
original on both fidelity metrics, on all five photographs.

**This is not a finding that detail recovery is impossible**, and it should not
be quoted as one. It is a finding about three operators, at one calibration, on
one corpus, under one degradation pipeline. A plausible reading of the evidence
is that Real-ESRGAN has already committed to an interpretation of its input and
leaves little for a later stage to recover, but that is a hypothesis this phase
suggests rather than a result it establishes — testing it would need candidates
this phase did not build. Nor is it a finding against post-processing in
general: `B`, a post-process, remains the safest active arm measured here.

The relationship to F3 needs stating plainly rather than assumed. F3 found
sharpening at 0.25 gave a blind-validated, visible texture benefit at 1:1 on
textured subjects; **this phase's visual pass did not reproduce that benefit**,
finding `B` equivalent to baseline on all seven regions inspected. That is a
disagreement between phases, not a refutation of either: F3 studied sharpening
strength across six levels with regions chosen for that question, and F5
inspected seven regions chosen for a different one. It is flagged here because a
reader should not take F5's silence on `B` as evidence against F3. The
production sharpening default remains 0 and this phase does not propose changing
it.

### The one direction worth a future phase

The dead-zone finding in §8 is the most promising thread and it is **untested**.
A candidate with a dead zone matched to the sharpener's would plausibly cut the
flat-region noise cost, and it would be cheap to try. It is **not** a reason to
reopen F5 and it is not a production change. A dead zone addresses the noise
cost, which is criterion 3; it does nothing about whether the added detail is in
the right place, which is criterion 6 and the one the reference track settled.
Whether such a variant could pass criterion 6 is untested and this report does
not predict it either way. The narrower question — how much of `B`'s safety
comes from its threshold rather than its filter shape — is worth answering on
its own.

## 12. Confidence and limitations

- **High confidence in the rejection.** On the two fidelity metrics the
  reference track is unanimous: baseline is best on PSNR and on SSIM for all five
  photographs against all three candidates, with no exceptions. It is *not*
  unanimous on the other two — `E` leads edge preservation on 4 of 5 and `C`
  leads texture error on 5 of 5 — and §5 explains why those two do not overturn
  the verdict. The visual inspection found no benefit to weigh against it.
- **Five photographs, one model, one scale, one observer.** `RealESRGAN_x4plus`
  at 4x. Every visual verdict is mine, and F3 documented that I got four of five
  blind regions wrong when I knew the expected direction.
- **The reference is itself a JPEG.** The corpus images are compressed, so the
  ground truth carries its own artifacts. This bounds the absolute PSNR values;
  it does not affect the comparison between arms, which all face the same
  reference.
- **One degradation pipeline, not a family.** A different blur, noise level or
  quality factor might change the numbers. The parameters were fixed in advance
  precisely so they could not be tuned toward a result, and the cost of that
  discipline is that only one point in the space was sampled.
- **Three candidates, not the space of candidates.** "Detail recovery is
  rejected" means these three, calibrated this way, on this corpus. It is not a
  proof that no post-process could work — §11 names the specific untested variant.
- **No dedicated rock region.** The corpus crop table has no rock entry;
  `landscape-detail`'s fine-texture region, which is rock in snow, was used
  instead. Documented rather than worked around by changing the corpus.
- **`E`'s edge-preservation win is not meaningful.** It leads on 4 of 5
  photographs by a median +0.0018 on a correlation, alongside the worst PSNR and
  the worst texture error of any arm.

---

## Reproducibility

```
cd backend
.venv/Scripts/python -m benchmarks.phase5_detail_recovery --operators
.venv/Scripts/python -m benchmarks.phase5_detail_recovery --degradation
.venv/Scripts/python -m benchmarks.phase5_detail_recovery --real
.venv/Scripts/python -m benchmarks.phase5_detail_recovery --strips
```

`--real` and `--degradation` refuse to run until `--operators` has recorded
whether any arm duplicates the incumbent.

Environment: Python 3.11.9, torch 2.7.1+cu118, CUDA on, `NVIDIA GeForce RTX 3050
Laptop GPU`. All five corpus SHA-256 digests verified against the manifest and
**identical to the digests recorded in F4** — the corpus, its manifest and its
attribution were not modified.

Measurements: `phase5_detail_recovery_measurements.json` — 69 rows, 5 operators,
25 degradation, 25 real, 7 crop, 7 strips. Crops: `phase5-detail-crops/`, 42
images including 7 blind strips with panel order recorded in the measurements
file.

Tests: `tests/unit/test_benchmark_detail_recovery.py`, 40 passed.
