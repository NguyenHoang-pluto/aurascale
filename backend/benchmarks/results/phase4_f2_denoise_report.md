# Phase 4 · F2 — denoise strength for `realesr-general-x4v3`

**Research only. `DEFAULT_DENOISE` was NOT changed as part of F2.**
`CREATIVE_DENOISE` was not changed either. No production code, no frontend, no
model registry, no `pyproject.toml`. Nothing committed, nothing pushed.

---

## 1. Research question

Is `DEFAULT_DENOISE = 1.0` too aggressive for real photographs, and is 0.25
actually a better general-purpose value for `realesr-general-x4v3`?

The question has consequences now. Before Phase 4 F1, the client always sent an
explicit denoise, so `CREATIVE_DENOISE = 0.25` was unreachable and every
denoise-capable job ran at 1.0. Since F1 (`bd77494`), Creative genuinely runs at
0.25 — so "is 0.25 right" stopped being hypothetical.

## 2. Hypothesis

1. 1.0 removes substantially more detail than noise on real photographs and is
   not defensible as a general default.
2. 0.25 removes visible noise at a materially smaller cost in texture.
3. 0.25 and 0.50 will be close, and the evidence may not separate them.

All three held. (3) held most strongly, which is why this report does not
declare a single winner between them.

## 3. Exact matrix

**5 photographs × 5 strengths = 25 runs. 25 completed, 0 failed, 0 skipped.**

| | |
|---|---|
| Model | `realesr-general-x4v3` (the only registry entry with a denoise pair) |
| Strengths | **0.00, 0.25, 0.50, 0.75, 1.00** |
| Scale | 4x, single native pass |
| Output | PNG, lossless |
| Sharpening | 0.0 |
| Target resolution | none |
| Chroma prefilter | none (rejected in Phase 3B) |
| Preprocessing | none |
| Tiling | production defaults, tile 256 / pad 16 |
| Precision | fp16, production policy |
| Baseline arm | `dn0.00` — the least-denoised output the model can produce |

**The only experimental variable is denoise strength.**

The **x4plus reference was reused, not re-run**, from
`phase3c_measurements.json`: the same five photographs at 4x, PNG, sharpening 0.
Re-running it would have spent four minutes reproducing recorded numbers. One
caveat travels with it — x4plus OOM-laddered to tile 128 on 4 of 5 images in
Phase 3C, so its *runtime* is not comparable with this sweep's; its **quality**
metrics are measured on output pixels and are.

## 4. Corpus

The established five, manifest-validated on every run, centre-cropped and never
resampled:

| Category | Dimensions |
|---|---|
| landscape-detail | 1732×1154 |
| portrait-skin | 1307×1530 |
| text-signage | 1632×1224 |
| foliage-texture | 1224×1632 |
| low-light-noise | 1892×1057 |

Licences in `corpus/manifest.json`; images not committed.

## 5. Methodology

Every arm runs through the production `RealEsrganUpscaler.upscale`, so tiling,
the OOM ladder and the precision policy are the product's. Metrics reuse
`benchmarks/metrics.py` unchanged; chroma statistics and the shared-flat-mask
correction come from `benchmarks/research_utils.py`. No second benchmark
framework was built.

Image-major, so the flat-block mask is taken once from each image's `dn0.00`
output and reused across its other four arms — the Phase 3B correction, since
denoising changes which blocks are flattest and a per-arm mask would compare
different regions.

**Visual evaluation ran first, blind, before any metric was read.**

### DNI semantics — verified, not assumed

`ModelManager._resolve_blend` was interrogated directly for every value:

| passed | resolved blend | meaning |
|---|---|---|
| `None` | `None` | no blend — the plain x4v3 weights |
| `0.00` | `0.00` | blended **fully to the wdn** counterpart |
| `0.25` | `0.25` | interpolated |
| `0.50` | `0.50` | interpolated |
| `0.75` | `0.75` | interpolated |
| **`1.00`** | **`None`** | **no blend — plain x4v3, identical code path to `None`** |

Two consequences the report depends on and does not gloss over:

- **`1.00` and `None` are the same thing.** Phase 3B confirmed byte-identical
  output. `DEFAULT_DENOISE = 1.0` therefore ships the *un-blended* weights.
- **`0.00` is not "denoise off".** It is the fully-wdn end, and it is the arm
  that keeps the most noise. It is the baseline precisely for that reason.

Both `denoise_passed` and `dni_blend_resolved` are recorded per run.

## 6. Reproducibility

| | |
|---|---|
| Date | 2026-09-09 |
| HEAD | `bd77494` |
| Runner | `benchmarks/phase4_f2.py` |
| Shared helpers | `benchmarks/research_utils.py` — corpus reading, block statistics, chroma metrics and the crop table |
| Measurements | `phase4_f2_measurements.json` (35 rows: 25 sweep + 10 strips) |
| Crops | `phase4-f2-crops/` — 50 per-arm crops + 10 blind strips |
| Model checkpoint | `realesr-general-x4v3.pth` sha256 `8dc7edb9ac80ccdc30c3a5dca6616509…`, 4 885 111 B |
| Denoise pair | `realesr-general-wdn-x4v3.pth` sha256 `1641f8c4464b9f097c9fdda558927371…`, 4 885 111 B |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |
| Python / NumPy / OpenCV / PyTorch | 3.11.9 / 2.4.6 / 5.0.0 / 2.7.1+cu118 |

**Dependency note.** The helpers this runner uses for reading a source, the
chroma statistics and the inspection-crop table were first written inside the
Phase 3 scripts, which are not tracked. They now live in the tracked
`benchmarks/research_utils.py`, so a clean checkout can run F2 without them.
The move was verified numerically first: `research_utils` reproduces the Phase 3
helpers byte-for-byte on all five photographs, so every number in this report is
unchanged by it.

Every run records input and output dimensions, tile size, reduction flag, fp16
state, runtime, peak VRAM and output SHA-256. Rerun with
`python -m benchmarks.phase4_f2 --sweep --strips`.

## 7. Runtime and VRAM

| arm | median runtime | tile | reduced | fp16 | peak VRAM |
|---|---|---|---|---|---|
| dn0.00 | 1.98 s | 256 | 0/5 | yes | 43 MiB |
| dn0.25 | 1.94 s | 256 | 0/5 | yes | 43 MiB |
| dn0.50 | 1.91 s | 256 | 0/5 | yes | 43 MiB |
| dn0.75 | 1.96 s | 256 | 0/5 | yes | 43 MiB |
| dn1.00 | 1.96 s | 256 | 0/5 | yes | 43 MiB |

**Denoise strength is free.** Runtime is flat within noise and VRAM is identical,
because DNI blends the weights once at load — confirming Phase 2.5. No tile
reduction anywhere; the configuration was identical across all 25 cells. So cost
plays no part in the decision.

## 8. Per-image metric results

Delta against `dn0.00`. Outliers are shown, not averaged away.

**high-frequency ratio**

| image | dn0.25 | dn0.50 | dn0.75 | dn1.00 |
|---|---|---|---|---|
| foliage-texture | **+23.3 %** | **+40.0 %** | +18.8 % | −9.5 % |
| landscape-detail | −35.0 % | −43.2 % | −48.0 % | −49.9 % |
| low-light-noise | −7.9 % | −6.1 % | −8.7 % | −10.2 % |
| portrait-skin | −14.1 % | −19.7 % | −39.8 % | −56.7 % |
| text-signage | −17.4 % | −20.8 % | −47.5 % | −63.1 % |
| **median** | **−14.1 %** | **−19.7 %** | **−39.8 %** | **−49.9 %** |

**flat noise** (what denoising is for)

| image | dn0.25 | dn0.50 | dn0.75 | dn1.00 |
|---|---|---|---|---|
| foliage-texture | **+2.8 %** | **+7.0 %** | −8.1 % | −36.3 % |
| landscape-detail | −25.7 % | −33.4 % | −50.2 % | −77.3 % |
| low-light-noise | −17.6 % | −23.0 % | −35.1 % | −46.4 % |
| portrait-skin | −2.1 % | −2.2 % | −14.7 % | −40.4 % |
| text-signage | −12.9 % | −19.1 % | −38.3 % | −83.8 % |
| **median** | **−12.9 %** | **−19.1 %** | **−35.1 %** | **−46.4 %** |

**local contrast** and **sobel p95** (micro-detail and edge strength)

| arm | local contrast | sobel p95 | overshoot | PNG size |
|---|---|---|---|---|
| dn0.25 | **−0.1 %** | **0.0 %** | −3.7 % | −6.1 % |
| dn0.50 | **−0.2 %** | +1.6 % | −0.2 % | −6.8 % |
| dn0.75 | −4.0 % | −0.6 % | −5.7 % | −13.1 % |
| dn1.00 | **−9.0 %** | −4.1 % | −16.7 % | **−25.0 %** |

`text-signage` is the worst case for edges: sobel p95 falls −13.5 % at 0.25 and
**−51.5 % at 1.00**.

## 9. Aggregate results

### Detail cost per unit of noise removed — `|Δhf| / |Δflat_noise|`

| image | dn0.25 | dn0.50 | dn0.75 | dn1.00 |
|---|---|---|---|---|
| foliage-texture | 8.43 | 5.75 | 2.33 | 0.26 |
| landscape-detail | 1.36 | 1.29 | 0.96 | 0.65 |
| low-light-noise | **0.45** | **0.27** | **0.25** | **0.22** |
| portrait-skin | 6.71 | 8.99 | 2.72 | 1.41 |
| text-signage | 1.34 | 1.09 | 1.24 | 0.75 |
| **median** | 1.36 | 1.29 | 1.24 | **0.65** |

**Read this one carefully — it is a trap.** By median ratio `dn1.00` looks
best (0.65). That is the Phase 2.5 trap restated: a favourable ratio at high
strength means it is removing a great deal of everything, not that the trade is
good. Section 10 shows what 1.00 actually looks like.

The row that matters is `low-light-noise`: where there is real noise, denoising
is cheap at every strength (0.22–0.45). Where there is not — portrait 6.71 at
0.25 — it is expensive. That is the same content-dependence Phase 3A measured
and then failed to exploit with a noise score.

### Against the x4plus reference (median across 5)

| metric | dn0.00 | dn0.25 | dn0.50 | dn0.75 | dn1.00 |
|---|---|---|---|---|---|
| hf ratio | +17.6 % | −7.9 % | −8.2 % | −26.2 % | −29.0 % |
| local contrast | +2.3 % | +4.8 % | +7.0 % | +3.0 % | +1.4 % |
| sobel p95 | +0.6 % | +0.5 % | +0.5 % | −8.2 % | −8.9 % |
| flat noise | +75.2 % | +44.3 % | +35.0 % | +14.8 % | −35.8 % |

Context that matters for interpretation: **on portrait-skin every v3 arm loses
heavily to x4plus** (hf −79 % to −91 %, sobel p95 −45 % to −47 %), at every
denoise value including 0.00. v3 is simply not a strong portrait model, and no
denoise setting fixes that. This reproduces Phase 3's finding.

## 10. Visual evaluation

Blind, mapping withheld until each verdict was written, panels shuffled per
region so the ladder never ran 0.00 → 1.00 left to right. Inspected at normal
viewing size, 1:1, and ~12× on the crops in `phase4-f2-crops/`.

Full mapping is in `phase4_f2_measurements.json` under `experiment: "strips"`.
Example: `portrait-skin__skin-cheek` was presented as
`P1..P5 = [dn1.00, dn0.25, dn0.50, dn0.00, dn0.75]`.

### Skin — the decisive region

Blind reading before the mapping was revealed: *"P1 flattest, pore mottling
largely gone, waxy; P2 most texture; P3 slightly less; P4 similar to P2; P5
smoother than P2–P4 but not as flat as P1."*

Mapping: **P1 = dn1.00**, P2 = dn0.25, P3 = dn0.50, P4 = dn0.00, P5 = dn0.75.

- **dn1.00 is visibly plastic.** Pore structure largely erased. `CLEARLY WORSE`.
- **dn0.25 preserves skin texture at the level of dn0.00**, the least-denoised
  arm. `EQUIVALENT` to baseline in texture, while removing noise elsewhere.
- dn0.50 `EQUIVALENT` — indistinguishable from 0.25 on skin.
- dn0.75 `SLIGHTLY WORSE` — smoothing becomes visible.

### Low-light — where denoising earns its place

- dn0.00 `WORSE` — heavy fine grain across the dark band.
- dn0.25 `BETTER` — grain much reduced, structure retained.
- dn0.50 `BETTER` — slightly cleaner still.
- dn0.75 `BETTER` on cleanliness.
- dn1.00 — cleanest in absolute grain, **but the dark band is flattened**; the
  shadow micro-structure is gone. `EQUIVALENT to SLIGHTLY WORSE` overall
  depending on whether shadow detail is valued.

### Text / signage

No ringing, no halos, no colour fringing in any arm. The sky/sign boundary is
crisp at every strength. dn1.00 flattens the dark foliage band above the sign
into near-black mush — `WORSE` for dark detail; `BETTER` for cleanliness.

### Foliage and landscape

Foliage: 0.25 / 0.50 / 0.75 near-indistinguishable; saturated greens preserved
throughout; no colour artefacts. Landscape rock: **dn0.25 retains the most fine
striation**, 0.50 slightly less, 0.75 visibly smoother — `SLIGHTLY WORSE` at
0.75, `WORSE` at 1.00.

### Summary table

| image | dn0.25 | dn0.50 | dn0.75 | dn1.00 | main reason |
|---|---|---|---|---|---|
| portrait-skin | equivalent | equivalent | slightly worse | **clearly worse** | plastic at 1.00 |
| low-light-noise | **better** | **better** | better | equivalent | grain removed; 1.00 flattens shadows |
| text-signage | better | better | slightly worse | worse | dark detail lost at 1.00 |
| foliage-texture | equivalent | equivalent | equivalent | slightly worse | very close throughout |
| landscape-detail | equivalent | slightly worse | slightly worse | **worse** | rock striation lost |

## 11. Metric-vs-eye disagreements

1. **I got low-light wrong, and the metric was right.** My blind reading ranked
   `dn0.25` as the cleanest dark band. The metric says `dn1.00` (flat noise
   −46.4 %), and a labelled re-inspection confirmed the metric. Recorded because
   the standing methodology treats the eye as the primary gate — and here the
   eye erred. It does not change the recommendation (1.00's cleanliness comes
   with flattened shadows), but a report that only published the reads I got
   right would be worthless.
2. **The cost-ratio metric favours `dn1.00` (0.65 median).** Visually 1.00 is
   the worst arm on three of five photographs. A ratio computed over large
   deltas flatters aggressive settings.
3. **`sobel_p95` barely moves for 0.25–0.75** (0.0 %, +1.6 %, −0.6 %) while the
   eye sees clear differences in rock striation and skin pores. Edge-strength
   percentiles do not track micro-texture.
4. **Agreement, for once:** `local_contrast` at −0.1 % / −0.2 % for 0.25 / 0.50
   and −9.0 % at 1.00 matches the visual reading exactly.

## 12. Failure and outlier cases

- **`foliage-texture` is non-monotone and is the phase's outlier.** hf *rises*
  with denoise (+23.3 % at 0.25, +40.0 % at 0.50) before falling, and flat noise
  also rises (+2.8 %, +7.0 %) before falling. A plausible reading is that the
  network resolves leaf structure more confidently once noise is reduced, but
  this phase did not test that and the number is reported as unexplained. It is
  the reason the aggregates use a median.
- **`portrait-skin` flat noise barely moves at 0.25/0.50** (−2.1 %, −2.2 %) —
  there is almost no noise there to remove, which is why the cost ratio is 6.71.
- **v3 loses to x4plus on portraits at every setting.** Not a denoise failure;
  a model property.
- **No inference failures, no OOM, no tile reduction** in any of the 25 cells.

## 13. Final recommendation

### `DEFAULT_DENOISE = 1.0` is not defensible as a default. **Confirmed.**

It costs a median **−49.9 % of high-frequency energy**, −9.0 % local contrast,
−25 % PNG size, and renders skin visibly plastic. Phase 2.5 said this on thinner
evidence; F2 confirms it with a blind visual pass and a corpus-wide sweep.

### Between 0.25 and 0.50, the evidence does not crown a winner.

Reported as a trade-off rather than a verdict, as the decision rule requires:

| | dn0.25 | dn0.50 |
|---|---|---|
| Noise removed | −12.9 % | **−19.1 %** |
| Detail cost (hf) | **−14.1 %** | −19.7 % |
| Local contrast | −0.1 % | −0.2 % |
| Sobel p95 | 0.0 % | +1.6 % |
| Skin | equivalent to baseline | equivalent to baseline |
| Low-light | better | **slightly better** |
| Landscape rock | **retains most striation** | slightly less |

They are indistinguishable on skin and foliage. 0.50 removes about 50 % more
noise for about 40 % more high-frequency cost; 0.25 keeps more rock texture.

**Preference: keep 0.25**, on the same asymmetry Phase 2.5 argued and this phase
did not overturn — under-denoising is recoverable by the user, erased texture is
not. `CREATIVE_DENOISE` is already 0.25, so this is a recommendation to **leave
Creative as it is**, now supported by direct evidence rather than inference.

Against the decision rule's six criteria, 0.25:

1. improves or preserves quality across the corpus — **yes**, better on 2,
   equivalent on 3, worse on none;
2. no plastic/smeared portraits — **yes**, equivalent to the baseline;
3. does not leave excessive low-light noise — **yes**, visibly cleaner than 0.00;
4. no halos, ringing or artefacts — **yes**, none found in any arm;
5. reasonable across categories — **yes**;
6. metrics broadly support it — **yes**: −0.1 % local contrast, 0.0 % sobel p95.

### What this does **not** recommend

Changing `DEFAULT_DENOISE`. That constant is the **slider's starting value**,
which is a UI question about what a user sees before they touch anything, not
about what a mode chooses. F2 measured model behaviour and provides the evidence
for such a change; making it is a separate decision with its own UI
consequences, and it was explicitly out of scope here.

## 14. Confidence

**High** that 1.0 is too aggressive — two phases, blind visual, consistent
across four of five photographs, and the one exception (foliage) is unexplained
rather than contrary.

**Moderate** that 0.25 is the best value. It is clearly *defensible* and clearly
better than 1.0. Whether it beats 0.50 is not resolved, and this report says so
rather than manufacturing a winner.

**Low** on any portrait-specific conclusion — one face, one subject.

## 15. Limitations

1. **Five photographs, one subject for skin.** The plastic-skin finding rests on
   a single face.
2. **`foliage-texture` behaves non-monotonically** and is unexplained.
3. **One region per category** was visually inspected per arm.
4. **My own low-light blind read was wrong** (§11.1). Blind evaluation reduces
   confirmation bias; it does not make a reader infallible.
5. **4x only**, on ~2 MP inputs. No 8x/16x or target-resolution runs; denoise is
   a load-time weight blend so it should be scale-independent, but that is not
   measured here.
6. **The x4plus reference was measured in a different run** with tile reduction
   on 4 of 5 images. Quality metrics are comparable; runtime is not.
7. **Only `realesr-general-x4v3` has a denoise pair**, so this says nothing
   about Standard, which has no denoiser at all.

## 16. Statement

**DEFAULT_DENOISE was NOT changed as part of F2.**

`CREATIVE_DENOISE` was not changed. No file under `backend/app`,
`frontend/src`, `models/` or `pyproject.toml` was modified. F3, F4 and F5 were
not run.
