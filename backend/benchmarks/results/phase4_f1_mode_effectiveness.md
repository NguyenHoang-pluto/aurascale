# Phase 4 · F1 — Is Enhancement Mode effective?

**Verdict: CONFIRMED INERT.** Through the real job pipeline, `mode=standard` and
`mode=creative` produce **byte-identical output** from the default UI payload.
Creative changes neither the model nor the denoise. It is recorded in the job
row and displayed in history while having no effect on a single pixel.

**Research only. No production code was modified.** The experiment drives
`backend/app` through its own HTTP API; it does not change it. `backend/app`,
`frontend/src`, `models/` and `pyproject.toml` are untouched. Nothing committed,
nothing pushed.

---

## Method

Not static inspection. The full production path was exercised end to end:

```
multipart POST /api/jobs
  → app.api.routes.jobs.create_job
  → JobService.submit          (parse_options, resolve_model, resolve_denoise)
  → persisted Job.enhance_options in SQLite
  → InProcessJobQueue → JobRunner
  → EnhancementService.enhance (REAL - not the test stub)
  → ModelManager.get           (real weights from models/)
  → RealEsrganUpscaler.upscale (production tiling, fp16, CUDA)
  → encoded PNG on disk
```

The wiring mirrors `tests/api/conftest.py::app_context` with **one deliberate
substitution**: the real `EnhancementService` replaces `StubEnhancement`, so a
real model is loaded and real pixels are produced. Isolated temp storage and
database; the real `models/` directory.

Source image: 96×72 PNG, 13 691 bytes, sha256 `3101ee8ca8174bbe…` — a
deterministic two-tone gradient with a hard vertical edge plus σ=6 noise.
Structured on purpose: a flat or pure-noise source would hide a denoise
difference, which is the opposite of what this test needs. Small so the jobs
finish in seconds.

Everything else pinned: 4x, PNG, `preserveMetadata=true`, no target, no
sharpening.

## A. Exact requests

Payloads are exactly what `buildSettings` + `useEnhanceJob` emit. The UI's
default model is `preferredModel()` = the first downloaded manifest entry =
**`RealESRGAN_x4plus`**, which has no denoise pair, so `buildSettings` emits
`{}` — sharpening is omitted at 0 and `denoiseStrength` is omitted for a model
without the paired weights.

| Case | Payload |
|---|---|
| **1 · standard-ui** | `model=RealESRGAN_x4plus, scale=4, format=png, preserveMetadata=True, mode=standard, settings={}` |
| **2 · creative-ui** | `model=RealESRGAN_x4plus, scale=4, format=png, preserveMetadata=True, mode=creative, settings={}` |
| **3 · creative-bare** *(control)* | `scale=4, format=png, preserveMetadata=True, mode=creative` — **no model, no settings** |
| **4 · creative-v3-ui** *(supplementary)* | `model=realesr-general-x4v3, …, mode=creative, settings={"denoiseStrength":1}` |

Cases 1–3 are the experiment. Case 4 covers the second reachable UI state: the
user manually selects the Creative model, at which point `buildSettings` *does*
send `denoiseStrength`, defaulting to `DEFAULT_DENOISE = 1`.

## B–E. Actual backend resolution, model, denoise, hash

| Case | Resolved model | Resolved denoise | Persisted `enhance_options` | Output | SHA-256 | Bytes |
|---|---|---|---|---|---|---|
| **1 · standard-ui** | `RealESRGAN_x4plus` | `None` | `{"sharpenStrength":0.0,"mode":"standard"}` | 384×288 png | `e87625b45a7e98d4…` | 110 408 |
| **2 · creative-ui** | `RealESRGAN_x4plus` | `None` | `{"sharpenStrength":0.0,"mode":"creative"}` | 384×288 png | **`e87625b45a7e98d4…`** | **110 408** |
| **3 · creative-bare** | `realesr-general-x4v3` | **`0.25`** | `{"sharpenStrength":0.0,"denoiseStrength":0.25,"mode":"creative"}` | 384×288 png | `b367edc81434423a…` | 149 968 |
| **4 · creative-v3-ui** | `realesr-general-x4v3` | **`1.0`** | `{"sharpenStrength":0.0,"denoiseStrength":1.0,"mode":"creative"}` | 384×288 png | `7c044bf5b32622ee…` | 95 934 |

All four jobs completed on CUDA. Model actually loaded is the `model_name`
column above, written by the worker after `ModelManager.get` resolved it.

## F. Output comparison

| Comparison | Bytes | Pixels | Model | Denoise |
|---|---|---|---|---|
| **1 vs 2** — Standard vs Creative, same UI payload | **IDENTICAL** | **IDENTICAL** (max diff 0) | same | same |
| 2 vs 3 — Creative-UI vs what Creative should be | differ | max 84, mean 3.29 | x4plus vs v3 | None vs 0.25 |
| 1 vs 3 | differ | max 84, mean 3.29 | x4plus vs v3 | None vs 0.25 |
| 3 vs 4 — intended Creative vs Creative-with-v3-selected | differ | max 49, mean 2.11 | same (v3) | **0.25 vs 1.0** |

No CASE C ambiguity: cases 1 and 2 match at the **byte** level, so there is
nothing for encoder nondeterminism or metadata to explain.

## G. Runtime

| Case | Processing | Note |
|---|---|---|
| 1 · standard-ui | 12 023 ms | Includes the first `RealESRGAN_x4plus` weight load |
| 2 · creative-ui | 269 ms | Same weights, served from the model cache |
| 3 · creative-bare | 235 ms | `realesr-general-x4v3` (1.21 M params) |
| 4 · creative-v3-ui | 117 ms | v3 with a DNI blend, also cached |

Runtime is not the finding; it is recorded to show that case 2 loaded **no new
model**, which is itself corroboration that Creative selected the same weights
as Standard.

## H. Verdict

### **CONFIRMED INERT**

From the default UI payload, `mode` changes nothing about the produced image.
Cases 1 and 2 differ only in the `mode` string, and their outputs are identical
byte for byte.

The control (case 3) proves this is **not** a broken mode planner: with the
explicit fields removed, Creative correctly resolves to
`realesr-general-x4v3 @ 0.25`, exactly as `mode_planner` intends, and produces
visibly different output.

Case 4 shows the failure has a second, independent half: even when the user
manually selects the Creative model, the UI sends `denoiseStrength = 1.0`, so
`CREATIVE_DENOISE = 0.25` is still overridden. Under **no reachable UI state**
does `CREATIVE_DENOISE` take effect.

## I. Why the explicit fields override the mode

The planners are correct. `mode_planner` documents the rule deliberately:

> An explicit model always wins. That is what keeps every client written before
> modes existed working unchanged.

That is sound. The defect is that the UI **always supplies both fields**, so the
"explicit" branch is the only branch that ever executes.

Three independent contributors, each individually reasonable:

1. **`CreateJobRequest.model` is required.** `useEnhanceJob` always sets
   `model: modelId`. So `resolve_model(mode, requested_model)` returns the
   requested model on every request and the mode's `model_id` is dead.
2. **`setMode` does not touch `modelId`.** In `useEnhancementStore`,
   `setMode: (mode) => { set({ mode }) }`. Choosing Creative changes a label,
   not the model, so the request still carries `RealESRGAN_x4plus`.
3. **`buildSettings` always sends `denoiseStrength` when the model supports it**
   — `if (options.supportsDenoise) settings.denoiseStrength = options.denoiseStrength`
   — and the store's default is `DEFAULT_DENOISE = 1`. There is no "unset"
   state, so `resolve_denoise` never sees `None` for a denoise-capable model.

With the UI default model (`x4plus`, no denoise pair) contributors 1 and 2 alone
make the mode inert. With v3 selected, contributor 3 finishes the job.

**Consequence, combined with earlier phases.** Phase 2.5 measured denoise 1.0
costing a median **−53 % high-frequency energy**; Phase 3 measured it at
**−88.4 % hf** on portrait skin against `x4plus`. Any user who selects the
Creative model today gets 1.0, the most destructive setting, rather than the
0.25 that Phase 2.5 recommended and Phase 2.75 implemented.

## J. Minimal production fix — proposal only, not implemented

The smallest correct change is to **give the client a way to express "I have not
chosen"**, so the mode defaults can apply. Two options, in order of preference:

### Option 1 — omit unset fields on the client *(recommended)*

Frontend only; no backend change; no API change.

- `buildSettings` sends `denoiseStrength` only when the user has actually moved
  the slider. Track a `denoiseTouched` flag in the store (or make
  `denoiseStrength: number | null` with `null` = unset).
- `useEnhanceJob` omits `model` when the user has not explicitly chosen one.
  This needs `CreateJobRequest.model` to become optional — the backend already
  handles a missing model (`resolve_model(mode, None)` → the mode's model, and
  `default_model_id()` as the final fallback), which is exactly what case 3
  demonstrated working.

**Why preferred:** it makes the existing, already-correct backend rule take
effect. Nothing about "explicit wins" changes, so pre-mode clients keep working.

### Option 2 — have `setMode` apply the mode's defaults in the store

`setMode` also sets `modelId` and `denoiseStrength` from a client-side copy of
the mode table.

**Why not preferred:** it duplicates `mode_planner` in the frontend, and the two
copies will drift — the precise failure `mode_planner`'s own docstring says it
exists to prevent.

### Not recommended

Making the mode override explicit fields on the backend. It would break the
compatibility guarantee, and it would silently ignore a user who deliberately
picked a model.

### Related, separate decision

`DEFAULT_DENOISE = 1` remains the slider's default even after a fix, so a user
who *does* touch the slider still starts at the most destructive value. Phase 2.5
recommends 0.25. **That is F2's question, not F1's**, and should not be bundled
into this fix.

## K. Regression tests to add before fixing

Written **before** the fix, so they fail first and prove they test the defect.

**Backend — API level** (`tests/api/test_modes_api.py`)

1. `test_creative_without_explicit_fields_selects_the_creative_model` — mode
   only → `model == CREATIVE_MODEL`, `denoiseStrength == CREATIVE_DENOISE`.
   *(This would pass today; it is the control that locks case 3 in.)*
2. `test_an_explicit_model_still_overrides_the_mode` — locks the compatibility
   rule so a fix cannot silently reverse it.
3. `test_standard_and_creative_differ_when_the_client_omits_the_model` — the
   direct regression: same payload apart from `mode`, and the persisted
   `model_name` must differ.
4. `test_denoise_omitted_lets_the_mode_choose` — `settings={}` with a
   denoise-capable model must resolve to `CREATIVE_DENOISE`, not `None`.

**Backend — service level** (`tests/unit/test_mode_planner.py`)

5. Parametrised truth table over `resolve_model` / `resolve_denoise` for
   (mode ∈ {standard, creative, None}) × (explicit ∈ {set, None}), asserting
   which wins in each of the six cells. Cheap and documents the contract.

**Frontend** (`src/features/enhance/enhance.test.tsx`)

6. `buildSettings` omits `denoiseStrength` when the value is untouched, and
   includes it once touched.
7. The submitted request omits `model` when the user has not chosen one.
8. `setMode` alone changes the submitted `mode` and **not** the submitted
   `model` — locking Option 1's division of responsibility.

**End-to-end** (optional, mirrors this experiment)

9. A byte-comparison test asserting Standard and Creative outputs **differ**
   under the default payload. Needs the real service, so it belongs in a
   slow-marked suite rather than the default run.

## Limitations

1. **One source image**, 96×72, synthetic-but-structured. Adequate for a binary
   wiring question; it says nothing about quality.
2. **The UI payload was reconstructed** from `buildSettings` and `useEnhanceJob`
   rather than captured from a running browser. It matches what that code emits;
   a live capture would be stronger.
3. **Case 4 assumes** the user selecting v3 leaves the slider at its default.
   That is the store's initial state, so it is the common path, not the only one.
4. `preferredModel()` returns the first *downloaded* manifest entry. All five are
   present here, so the UI default is `RealESRGAN_x4plus`. On an installation
   with a different download set the default model could differ — though
   contributors 1 and 2 make the mode inert regardless of which model it picks.

## Reproducibility

| | |
|---|---|
| Date | 2026-09-09 |
| HEAD | `73e4e06` |
| Raw results | embedded in the tables above (§A–G) |
| Method | temporary tracer under the session scratchpad, removed after the run per the experiment brief; §Method plus the §A payloads specify it completely |
| Wiring | `tests/api/conftest.py::app_context`, with `EnhancementService` in place of `StubEnhancement` |
| Source | 96×72 PNG, sha256 `3101ee8ca8174bbe…` |
| Config | 4x, PNG, `preserveMetadata=true`, sharpening 0, production tiling, CUDA fp16 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |

No number here was estimated. Every hash, byte count and pixel delta came from
the four jobs described.
