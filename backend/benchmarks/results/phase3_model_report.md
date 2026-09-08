# Phase 3 — which super-resolution model should AuraScale standardise on

**Research only. No production default changed, no model replaced, no frontend
or pipeline code touched.**

## Candidates

Restricted to weights already in the local registry. A model that must first be
downloaded, vendored and licence-checked is an engineering decision rather than
a benchmark arm; HAT and its relatives are assessed separately at the end.

| Arm | Model | Architecture | Params | Weights |
|---|---|---|---|---|
| **x4plus** *(reference)* | `RealESRGAN_x4plus` | RRDBNet, 23 blocks | 16.70 M | 63.9 MB |
| **anime6b** | `RealESRGAN_x4plus_anime_6B` | RRDBNet, 6 blocks | 4.47 M | 17.1 MB |
| **general-v3-dn0** | `realesr-general-x4v3` | SRVGGNetCompact | 1.21 M | 4.7 MB |
| **general-v3-dn1** | `realesr-general-x4v3` | SRVGGNetCompact | 1.21 M | 4.7 MB |

`realesr-general-x4v3` appears twice on purpose. It is the only candidate with
a denoise pair; the other two have no denoising stage at all. Comparing it at
the shipped `DEFAULT_DENOISE` of 1.0 would measure the denoiser rather than the
network, so **dn0** is the like-for-like arm and **dn1** is what a user
selecting it gets today.

## Matrix

5 images × 4 arms = **20 runs, 20 completed, 0 failed.**

| | |
|---|---|
| Scale | 4x — the product default, and native to all three networks, so no arm pays for a second pass |
| Output | PNG, so no codec quantisation is mistaken for a model difference |
| Sharpening | 0.0 |
| Tiling / device | production defaults, unchanged (CUDA, tile 256, pad 16, fp16) |
| Corpus | the Phase 2.5 real-photo corpus, with the foliage image replaced |
| Baseline | `x4plus` |

Metrics come from the existing harness, on **full-resolution PNG outputs** (up
to 5228×6120), never the 4096 px preview, from 16 deterministic seeded tiles.

### Corpus change

`foliage-texture` was a dark archival fern specimen on near-black; Phase 2.5
had to exclude it because its baseline `flat_noise` of 0.000396 made every
relative metric a near-zero denominator. Replaced with **A Bamboo Perspective**
(Wikimedia Commons, CC BY-SA 4.0), dense sunlit bamboo leaves. It now measures
as the highest-texture image in the corpus:

| | old fern | new bamboo |
|---|---|---|
| flat_noise | 0.000396 | **0.009004** (23×) |
| high_frequency_ratio | — | **0.03484** (highest in corpus) |
| local_contrast | — | **0.11672** (highest in corpus) |

Full provenance in `corpus/manifest.json`; the other four images are unchanged.

## Measured results

### Median relative delta vs `x4plus`, across all five images

| arm | sobel_mean | sobel_p95 | local_contrast | hf_ratio | flat_noise | overshoot |
|---|---|---|---|---|---|---|
| anime6b | −17.1 % | −3.4 % | −2.6 % | +9.1 % | **−55.2 %** | **+28.7 %** |
| general-v3-dn0 | **+7.6 %** | +0.6 % | **+3.4 %** | +3.0 % | **+74.0 %** | −3.8 % |
| general-v3-dn1 | −15.6 % | −8.9 % | +1.4 % | −29.0 % | −35.8 % | −19.9 % |

### Per-image, the two that matter most

**portrait-skin** — every arm loses heavily to `x4plus`:

| arm | sobel_mean | local_contrast | hf_ratio | flat_noise |
|---|---|---|---|---|
| anime6b | −46.5 % | −28.6 % | −61.0 % | −82.7 % |
| general-v3-dn0 | −37.8 % | −18.7 % | −73.1 % | −7.0 % |
| general-v3-dn1 | −48.1 % | −27.7 % | −88.4 % | −44.5 % |

**landscape-detail** — `general-v3-dn0` beats the reference on detail, at a cost:

| arm | sobel_mean | local_contrast | hf_ratio | flat_noise |
|---|---|---|---|---|
| anime6b | −17.1 % | −2.6 % | +9.1 % | −55.2 % |
| general-v3-dn0 | **+21.1 %** | **+11.5 %** | **+41.8 %** | **+182.5 %** |
| general-v3-dn1 | −15.6 % | +1.4 % | −29.0 % | −35.8 % |

### Cost

| arm | median time | mean time | median output | vs reference |
|---|---|---|---|---|
| x4plus | 38 849 ms | 38 855 ms | 35.2 MB | 1.0× |
| anime6b | 16 651 ms | 17 253 ms | 31.4 MB | **2.3× faster** |
| general-v3-dn0 | 9 591 ms | 9 467 ms | 42.7 MB | **4.1× faster** |
| general-v3-dn1 | 10 013 ms | 9 667 ms | 30.0 MB | **3.9× faster** |

### VRAM and RAM (measured directly, RTX 3050 Laptop 4096 MiB)

| model | params | VRAM after load | peak VRAM, tile 256 | peak VRAM, tile 512 |
|---|---|---|---|---|
| x4plus | 16.70 M | 32.0 MiB | **584.4 MiB** | not measured — see below |
| anime6b | 4.47 M | 8.6 MiB | 561.0 MiB | 1194.1 MiB |
| general-v3 | 1.21 M | 2.3 MiB | **34.8 MiB** | 131.9 MiB |

`general-v3` uses **17× less VRAM** than `x4plus` at the production tile size.

The `x4plus` tile-512 figure could not be measured: the attempt failed on a
48 MiB *host* numpy allocation with 2.0 GB physically free — Windows commit
exhaustion, the same environmental pressure seen throughout this work. That is
itself a finding: on this machine **host RAM is as tight a constraint as the
4 GB of VRAM**, and it is not a property of the model.

Peak host RSS was ~2.1–2.3 GiB for every arm, dominated by the torch/CUDA
context rather than by the network.

## Visual observations

*Subjective. Fixed regions chosen by eye from the corpus overview — signage
lettering, eye and skin, rock in snow, bamboo leaves, bridge at night — the
same coordinates across all four arms, at 100 %, 200 % and 400 % in
`results/phase3-crops/`. These are impressions, and are kept separate from the
measurements above.*

- **portrait skin (the decisive one)** — `anime6b` does not merely smooth skin,
  it **restyles** it: pores and fine texture are replaced by smooth painterly
  strokes that were not in the source. `x4plus` retains the most genuine pore
  detail. `general-v3-dn0` is close to `x4plus` but softer; `dn1` loses most
  pores.
- **landscape rock** — `general-v3-dn0` renders visibly *more* rock structure
  than `x4plus`, matching its +21 % sobel and +41.8 % hf. `anime6b` flattens
  rocks into blobs with hard cartoon outlines.
- **signage, dark sign face** — `x4plus` keeps fine grain; `anime6b` is
  completely flat; `general-v3-dn0` shows noticeably *more* mottle than the
  reference, consistent with its +94.6 % flat_noise on that image.
- **bamboo foliage** — the four are closest here; `anime6b` gives slightly
  harder, more graphic leaf edges.

## Failure and OOM cases

**No inference failures.** All 20 runs completed; `phase3_failures.json` is
empty. No OOM, no tile reduction, no CPU fallback was triggered at 4x on 2 MP
inputs.

One measurement-side failure, already described: `x4plus` at tile 512 could not
be profiled because of host commit exhaustion, not GPU memory.

## Recommendation

### Best Standard model: **keep `RealESRGAN_x4plus`**

Not because it wins on every metric — it does not — but because it is the only
arm that never fails badly. It has the best skin, no restyling, and the lowest
artefact risk. `general-v3-dn0` beats it on landscape detail while nearly
tripling flat-region noise (+182.5 %) and losing heavily on skin (−73 % hf);
that is a trade a general-purpose default should not make silently.

The honest caveat: **`x4plus` is 4× slower for a quality lead that is
content-dependent rather than uniform.**

### Best Creative candidate: **`realesr-general-x4v3` at low denoise**

It is the only arm that produced *more* detail than the reference anywhere, and
it does so at **4.1× the speed and 1/17th the VRAM**. Its weakness — noise
amplification — is exactly what a Creative mode can expose as a control rather
than hide. Pairing it with the adaptive denoise from the Auto Enhance design
would address the one thing holding it back. It is also the obvious basis for a
"fast preview" path.

### Unsuitable: **`RealESRGAN_x4plus_anime_6B` for photographs**

It restyles rather than restores. The metrics understate this — its −55 %
flat_noise looks like clean output until you see that texture was replaced, not
cleaned — and its **+28.7 % median edge overshoot** is the highest of any arm.
It should remain available for illustration, which is what it was trained for,
and should never become a photographic default.

### Is HAT worth further engineering? **Not yet on this hardware.**

Licences verified from the source repositories: **HAT Apache-2.0**, **SwinIR
Apache-2.0**, **DRCT MIT** — all permissive and compatible. Nothing blocks
adoption legally.

I did **not** integrate HAT, and that is a deliberate judgement rather than an
omission. Integrating it means vendoring a new architecture into
`app/inference/arch/`, adding a manifest entry and a weights download — that is
production code, which this phase was told not to change. Assessed against what
was measured here:

- `x4plus` already takes **39 s** for a 2 MP image at 4x on this GPU. Window
  attention is substantially heavier per pixel than RRDBNet, so a transformer
  would plausibly put a routine job into many minutes.
- VRAM headroom is thin. `x4plus` peaks at 584 MiB on a 4096 MiB card with the
  backend resident, and attention memory grows with tile area, so HAT would
  likely need tiles well below 256 — which raises tile count, and therefore
  time, again.
- Host RAM is already the binding constraint on this machine, and a larger
  model makes that worse before it makes anything better.

**What would change the answer:** a GPU with ≥8 GB, or a measured HAT run
showing a quality lead over `x4plus` large enough to justify a 10×+ time cost.
Neither is available here. The cheaper next experiments are (a) community
RRDBNet fine-tunes, which are drop-in — the loader already builds RRDBNet, so
they cost a manifest entry and a weights file with **no architecture change** —
and (b) adaptive denoise on `general-v3`, which the measurements above suggest
is where the real headroom is.

## Limitations and confidence

**Confidence: moderate for the Standard recommendation, lower for Creative.**

1. **Five images, one per category, no replication.** No confidence intervals
   are possible, and the medians move on single images.
2. **The metrics cannot tell restoration from invention.** `anime6b` scores
   respectably on `hf_ratio` while visibly fabricating texture; only the visual
   pass caught it. Any future model must be judged the same way.
3. **All sources are Commons JPEGs**, re-encoded once at quality 97 after
   cropping. Some measured high-frequency energy is compression artefact.
4. **Visual observations are one person's impressions**, unblinded, and are
   labelled as such. A scored blind comparison with several raters would be
   needed before acting on the Creative recommendation.
5. **Timings come from a memory-pressured machine** with ~2 GB free and a
   shared GPU. The *ratios* between arms are trustworthy; the absolute numbers
   are not a benchmark of the hardware.
6. **Only 4x was tested.** 8x, which runs a cascade, may rank the models
   differently — pass 2 receives a clean model-generated image rather than a
   degraded one, and the networks may differ in how they handle that.
7. **HAT is unmeasured.** Everything said about its speed and memory above is
   inference from architecture, explicitly not measurement.

## Reproducing

```bash
python -m benchmarks.fetch_corpus              # all five images
python -m benchmarks.fetch_corpus foliage-texture   # or just one
```

The matrix is defined in `benchmarks/models.py`; measurements for this run are
in `results/phase3_measurements.json`, crops in `results/phase3-crops/`.
