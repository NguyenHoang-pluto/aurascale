# Image-quality benchmark harness

Phase 0 of the Image Quality Engineering workstream. It measures **full
resolution result files** and reports deltas against a baseline arm.

It does not run inference, and nothing in `app/` imports it. Running a
benchmark cannot perturb the pipeline it is measuring.

## Supplying a corpus

No images are committed and none are downloaded. Photographs carry licences,
and a corpus fetched over the network is not reproducible. Put your own images
under `benchmarks/corpus/<category>/`:

```
benchmarks/corpus/
  landscape-detail/     fine structure across the whole frame
  portrait-skin/        smooth gradients; halos and over-denoising show here first
  text-signage/         hard edges; ringing is obvious to the eye
  foliage-texture/      dense stochastic texture, the first thing denoising erases
  low-light-noise/      real sensor noise, to separate denoising from detail loss
```

Any of `.png`, `.jpg`, `.jpeg`, `.webp`. Keep each input at or below 3 MP so an
8x run stays inside `MAX_OUTPUT_PIXELS` (200 MP).

Check what is present:

```
python -m benchmarks.runner --corpus
```

A missing category is reported, not an error. There is **no ground truth** in
this package, and none is fabricated.

## Measuring

Produce results however you like — the API, a script, by hand — then describe
them in a spec and measure:

```
python -m benchmarks.runner --spec runs.json --json report.json
```

See the module docstring in `runner.py` for the spec format. Every `cost` field
is optional: a run measured after the fact has no timing, and a fabricated one
would be worse than a null.

## What is measured

All on Rec.709 luma, from 16 deterministic 512 px tiles (seeded, so two runs
sample the same regions). Sampling keeps the cost flat whether the result is
12 MP or 192 MP.

| Metric | Reads as |
|---|---|
| `sobel_mean` | overall acutance |
| `sobel_p95` | how hard the strongest edges are driven |
| `local_contrast` | micro-detail |
| `high_frequency_ratio` | fine structure present; falls when an image gets larger without getting sharper |
| `flat_noise` | noise, measured only where there is no structure |
| `edge_overshoot` | halo/ringing **proxy** |

## Reading the results

**Higher is not better.** There is deliberately no aggregate score, because
every interesting failure improves one number and worsens another:

- sharpening raises `sobel_p95` **and** `edge_overshoot` — the pair separates
  real acutance from ringing;
- denoising lowers `flat_noise` **and**, if overdone, `local_contrast` and
  `high_frequency_ratio` — the trio separates cleaning from erasing.

A delta is only reported between arms whose outputs are **the same size**. All
of these metrics are resolution-dependent, so a 4x-versus-8x delta would be a
confident-looking number describing only the resolution change. The harness
refuses it and says why.
