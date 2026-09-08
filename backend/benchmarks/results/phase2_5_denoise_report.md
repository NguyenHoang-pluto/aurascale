# Phase 2.5 — choosing the denoise default for `realesr-general-x4v3`

**Status: recommendation made, not applied.** `DEFAULT_DENOISE` is still `1.0`
in `frontend/src/stores/useEnhancementStore.ts`. Nothing in the production
pipeline changed.

## Why this exists

Phase 2 measured the denoise sweep on a synthetic image and found `1.0`
removing **98.4 %** of high-frequency energy. That corpus then disqualified
itself: the model correctly treated its gaussian "texture" as noise and erased
it, which is right behaviour on gaussian noise and exactly why synthetic
texture cannot stand in for photographic texture. The recommendation was
withheld pending real photographs. This is that experiment.

## Corpus

Five photographs from Wikimedia Commons, fetched by
`benchmarks/fetch_corpus.py`. Licence and authorship are recorded as the
Commons API reported them, not transcribed by hand, so provenance cannot drift
from the source. The images are **not committed** — they carry attribution
obligations, and the manifest makes the corpus reproducible instead.

| Category | Dimensions | Licence | Author | Source |
|---|---|---|---|---|
| landscape-detail | 1732×1154 | CC BY 4.0 | Vyacheslav Argenberg | [Mountains in snow, Chola Valley, Nepal](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) |
| portrait-skin | 1307×1530 | CC BY 4.0 | (unstated on Commons) | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) |
| text-signage | 1632×1224 | CC BY-SA 4.0 | PortlandAppraisalBlog | [SE Ankeny Street sign, Portland](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) |
| foliage-texture | 1161×1722 | CC BY 4.0 | Sherwin John Carlquist | [Cephalomanes fern foliage, Vanuatu](https://commons.wikimedia.org/wiki/File:(Cephalomanes_fern_foliage_close-up_in_Espiritu_Santo,_Vanuatu)_-_DPLA_-_970b3b811cfb7b136fe34d425d0e9741.jpg) |
| low-light-noise | 1892×1057 | CC0 | Wilfredor | [Széchenyi Chain Bridge at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) |

**Cropped, never scaled.** Each image is a centre crop of the original to about
2 MP. Downscaling averages sensor noise away, and sensor noise is precisely
what this experiment measures — a resize would have answered the question
before the experiment ran. Every pixel is as the camera recorded it. Original
dimensions and the exact crop box are in `corpus/manifest.json`.

## Setup

One variable. Everything else pinned in `benchmarks/denoise.py`:

| | |
|---|---|
| Model | `realesr-general-x4v3` — the only registry entry with a denoise pair, so the only model the setting reaches at all |
| Scale | 4x |
| Denoise | **0.00, 0.25, 0.50, 0.75, 1.00** |
| Sharpening | 0.0 |
| Output | PNG — so the encoder's own quantisation is never measured as denoising |
| Tiling / device | defaults, unchanged (CUDA, tile 256, pad 16) |
| Baseline arm | `denoise-0.00` |

25 runs. All completed; all five arms of an image share identical output
dimensions, so every delta is between comparable images — the harness refuses
cross-resolution deltas and none were needed.

Metrics are measured on **full-resolution PNG outputs** (up to 4644×6888), never
the 4096 px preview, from 16 deterministic seeded tiles per image.

## Semantics, confirmed empirically

Higher value = more denoising. `flat_noise` falls monotonically with the
setting on every usable image, which confirms the reading in
`model_manager.get` and matches upstream's `--denoise_strength`.

## Per-image results (relative to denoise 0.00)

### landscape-detail — snow and rock texture

| arm | hf_ratio | flat_noise | local_contrast | sobel_p95 | overshoot |
|---|---|---|---|---|---|
| 0.25 | −35.0 % | −25.7 % | −5.6 % | −6.5 % | −12.9 % |
| 0.50 | −43.2 % | −33.4 % | −6.5 % | −8.0 % | −14.7 % |
| 0.75 | −48.0 % | −50.2 % | −7.7 % | −8.7 % | −15.6 % |
| 1.00 | −49.9 % | −77.3 % | −9.0 % | −9.4 % | −16.7 % |

### portrait-skin

| arm | hf_ratio | flat_noise | local_contrast | sobel_p95 | overshoot |
|---|---|---|---|---|---|
| 0.25 | −14.1 % | −2.1 % | −0.1 % | 0.0 % | −3.7 % |
| 0.50 | −19.7 % | −2.2 % | −0.2 % | +1.6 % | +2.8 % |
| 0.75 | −39.8 % | −14.7 % | −4.0 % | −0.6 % | −5.7 % |
| 1.00 | −56.7 % | −40.4 % | −11.1 % | −4.1 % | −20.1 % |

### text-signage

| arm | hf_ratio | flat_noise | local_contrast | sobel_p95 | overshoot |
|---|---|---|---|---|---|
| 0.25 | −17.4 % | −12.9 % | −6.6 % | −13.5 % | −21.6 % |
| 0.50 | −20.8 % | −19.1 % | −11.4 % | −26.3 % | −27.5 % |
| 0.75 | −47.5 % | −38.3 % | −18.5 % | −39.8 % | −31.8 % |
| 1.00 | −63.1 % | −83.8 % | −25.2 % | −51.5 % | −37.0 % |

### low-light-noise — the image denoising exists for

| arm | hf_ratio | flat_noise | local_contrast | sobel_p95 | overshoot |
|---|---|---|---|---|---|
| 0.25 | −7.9 % | −17.6 % | **+2.5 %** | +4.2 % | −1.6 % |
| 0.50 | −6.1 % | −23.0 % | **+5.4 %** | +8.0 % | −0.2 % |
| 0.75 | −8.7 % | −35.1 % | **+7.1 %** | +10.1 % | −0.1 % |
| 1.00 | −10.2 % | −46.4 % | **+8.2 %** | +11.4 % | −1.1 % |

### foliage-texture — excluded, see failure cases

| arm | hf_ratio | flat_noise |
|---|---|---|
| 0.25 | +322.9 % | +133.0 % |
| 0.50 | +369.1 % | +204.3 % |
| 0.75 | +292.1 % | +0.4 % |
| 1.00 | −82.1 % | −95.5 % |

## Aggregate (median across the four usable images)

Median, not mean: the mean is dominated by the excluded outlier and would
report `hf_ratio +49.7 %` at 0.25, which is an artefact rather than a finding.

| arm | hf_ratio | flat_noise | local_contrast | sobel_p95 | overshoot |
|---|---|---|---|---|---|
| 0.25 | −15.7 % | −15.3 % | −2.9 % | −3.3 % | −8.3 % |
| 0.50 | −20.3 % | −21.0 % | −3.3 % | −3.2 % | −7.4 % |
| 0.75 | −43.7 % | −36.7 % | −5.8 % | −4.7 % | −10.7 % |
| 1.00 | −53.3 % | −61.8 % | −10.1 % | −6.8 % | −18.4 % |

### Detail cost per unit of noise removed — `|Δhf| / |Δnoise|`, lower is better

| image | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| landscape-detail | 1.36 | 1.29 | 0.96 | 0.65 |
| portrait-skin | 6.71 | 8.99 | 2.72 | 1.41 |
| text-signage | 1.34 | 1.09 | 1.24 | 0.75 |
| low-light-noise | **0.45** | **0.27** | **0.25** | **0.22** |
| **median** | 1.35 | 1.19 | 1.10 | 0.70 |

The single most important row is the last image. Where there is real noise to
remove, denoising is cheap — 0.22–0.45 units of detail per unit of noise. Where
there is not, it is expensive: on the portrait, 0.25 costs **6.7×** more detail
than the noise it removes, because there was almost no noise there to begin
with.

## Cost

| arm | mean processing | mean output |
|---|---|---|
| 0.00 | 9820 ms | 38.2 MB |
| 0.25 | 7595 ms | 35.6 MB |
| 0.50 | 8207 ms | 33.7 MB |
| 0.75 | 9008 ms | 30.2 MB |
| 1.00 | 8576 ms | 24.2 MB |

Processing time is flat within noise — DNI blends the weights once at load, so
the setting costs nothing at inference. Output size falls **37 %** from 0.00 to
1.00, which is itself a measure of how much detail is being removed.

Per-run peak RSS was not captured: the runs went through the job API, and the
worker's peak is not exposed per job. Harness-side measurement cost is bounded
by tile sampling and is independent of the arm.

## Visual observations

Fixed regions per image, same coordinates across all five arms, rendered at
100 %, 200 % and 400 % into `results/crops/`. Regions were chosen
programmatically as the highest-variance area of each image, so the comparison
is not cherry-picked.

- **landscape (`v-landscape-detail-400.png`)** — the clearest evidence. At 0.00
  the snow face is granular; much of that grain is noise. 0.25 and 0.50 clean it
  while the ridge structure stays legible. By 0.75 fine micro-texture is
  visibly going, and 1.00 is smooth — ridges survive, the texture between them
  does not.
- **low-light (`v-low-light-noise-400.png`)** — blotchy chroma mottle at 0.00,
  clearly cleaner by 0.25, cleanest at 1.00. This is the case where the setting
  earns its place.
- **portrait (`v-portrait-skin-400.png`)** — grain on skin at 0.00; smooth by
  1.00. No waxiness at any level on this image, contrary to expectation.
- **text-signage (`v-text-signage-400.png`)** — the chosen region landed in
  dark foliage rather than lettering. It shows texture erased progressively,
  essentially flat at 1.00. **Lettering itself was not visually inspected**,
  which is a gap.

### A suspected artefact that turned out not to exist

At 400 % the mid arms appeared to show a regular cross-hatch absent at both
endpoints, which would have suggested DNI weight blending producing artefacts
at intermediate alphas. Tested by locating the dominant spatial frequency in a
flat region: the peak sits at **the same 72.4 px period at every arm including
both endpoints**, and weakens monotonically with denoise. There is no
blend-specific periodic artefact. The impression was an artefact of viewing
residual noise at 400 % nearest-neighbour.

## Failure cases

**`foliage-texture` is unusable and is excluded from the aggregate.** It is a
dark archival specimen photograph on a near-black background, not the dense
sunlit foliage the category intends. Its baseline `flat_noise` is 0.000396 —
an order of magnitude below every other image (0.0045–0.0066). Relative deltas
against a near-zero baseline are meaningless: the reported +322.9 % `hf_ratio`
at 0.25 is a tiny absolute change in a frame with almost no signal, not a
threefold improvement. Its non-monotonic behaviour (noise rising to 0.50, then
collapsing at 1.00) is the same effect. **The category needs a different
image**; the result says nothing about denoise.

**The `text-signage` crop missed the lettering.** The automatic
highest-variance region selection picked dark foliage in the same frame. Edge
and ringing behaviour on actual text remains visually unverified, though the
metrics for that image do cover it.

## Recommendation

**0.25**, with the reservation below. Not applied.

The reasoning, against the stated rule — *the lowest denoise that reduces
obvious noise without materially reducing high-frequency detail*:

- **1.00 is not defensible as a default.** It costs a median **53 %** of
  high-frequency energy and 25 % of local contrast on text. Phase 2's synthetic
  corpus overstated the magnitude (98.4 % vs 53 %), but the direction and the
  conclusion hold on real photographs.
- **0.75 is not a compromise**, it is most of the way to 1.00: −43.7 % hf for
  −36.7 % noise.
- **0.50 costs 20.3 % of detail for 21.0 % noise** — roughly break-even by the
  median, and it does measurably better than 0.25 on the noisy image.
- **0.25 is the least destructive setting that still denoises.** Median −15.7 %
  hf for −15.3 % noise, and on the low-light image the trade is strongly
  favourable (−7.9 % hf for −17.6 % noise) with local contrast *rising*.

0.25 is chosen over 0.50 because the harm from under-denoising is recoverable —
the user can raise the slider — while erased texture cannot be recovered at
all. Between two defensible values, the reversible error is the better default.

## Confidence

**Moderate, for rejecting 1.00. Low-to-moderate, for 0.25 specifically over
0.50.**

The case against 1.00 is strong: consistent in direction across four images of
different content, large in magnitude, corroborated visually, and now agreeing
with the synthetic result. I would act on it.

The case for 0.25 over 0.50 rests on a preference for reversible error rather
than on a measured separation. The two are within ~5 points of each other on
the median, and 0.50 is genuinely better on the one image where denoising
matters most. A reasonable person could pick either.

## Remaining uncertainty

1. **Four usable images is a small corpus**, one per category, with no
   replication within a category. No confidence intervals are possible.
2. **The foliage category is unrepresented.** Dense natural texture is the
   content most at risk from denoising, and this experiment did not test it.
3. **All sources are JPEGs from Commons**, re-encoded once at quality 97 after
   cropping. Some measured high-frequency energy is compression artefact rather
   than detail, and these metrics cannot separate the two — so the detail
   "cost" of denoising is likely somewhat overstated.
4. **The metrics cannot distinguish real texture from invented texture.** A
   setting that hallucinated plausible detail would score well here.
5. **A single global default is the wrong shape for this problem.** The
   cost/benefit ratio spans 0.22 to 8.99 across four images and tracks how
   noisy the input is. The right answer is adaptive denoise driven by a noise
   estimate — stage 2 of the Auto Enhance design. Any fixed default is a
   compromise between a clean studio portrait and a night photograph.
6. **`x4plus` is unaffected either way.** It has no denoise pair, so
   `_denoise_for` drops the setting; the default only reaches users who
   deliberately select the general model.

## Reproducing

```bash
python -m benchmarks.fetch_corpus     # populates corpus/ and manifest.json
python -m benchmarks.runner --corpus  # confirm what is present
```

The matrix is defined in `benchmarks/denoise.py`; measurements for this run are
in `results/phase2_5_measurements.json`.
