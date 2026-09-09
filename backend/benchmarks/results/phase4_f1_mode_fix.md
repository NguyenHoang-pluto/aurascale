# Phase 4 · F1 fix — making Enhancement Mode effective

Implementation note for the defect confirmed in
`phase4_f1_mode_effectiveness.md`. **Frontend-only change.** No backend logic,
no model registry, no database schema, no API contract, no inference
architecture, and `DEFAULT_DENOISE` is deliberately untouched.

Not committed, not pushed.

---

## Root cause

`mode_planner` was correct and so was the route. The client was the problem, and
in a way that could not be seen from either side alone.

The backend rule is deliberate and documented:

> An explicit model always wins. That is what keeps every client written before
> modes existed working unchanged.

Sound — but the client **always supplied one**, so the "explicit" branch was the
only branch that ever ran. Three independent contributors:

1. `CreateJobRequest.model` was required, and `useEnhanceJob` always set
   `model: modelId`.
2. `setMode` changed only `mode`; the request still carried whatever model
   `adoptDefaults` had adopted on load.
3. `buildSettings` sent `denoiseStrength` whenever the model supported it, and
   the store's default is `DEFAULT_DENOISE = 1`. There was no "unset" state.

The underlying modelling error: **the client could not distinguish a value the
user chose from a value that merely equals the default.** `RealESRGAN_x4plus`
selected by `adoptDefaults` looked exactly like `RealESRGAN_x4plus` picked from
the dropdown, and a slider resting at 1.0 looked exactly like a slider the user
had dragged to 1.0.

## Before → after

| Scenario | Before | After |
|---|---|---|
| Standard, nothing touched | `model=x4plus` sent → x4plus | **model omitted** → planner picks `STANDARD_MODEL` = x4plus |
| Creative, nothing touched | `model=x4plus` sent → **x4plus, denoise None** — identical to Standard | **model omitted** → planner picks `realesr-general-x4v3` @ **0.25** |
| Creative, user picked v3 | `denoiseStrength=1.0` sent → **1.0**, never `CREATIVE_DENOISE` | denoise omitted until touched → **0.25** |
| Creative, user picked a model | model sent → that model | unchanged — **explicit still wins** |
| Creative, user moved the slider | value sent | unchanged — **explicit still wins**, including an explicit `0` |

Standard's behaviour is **byte-identical before and after** (sha256
`e87625b45a7e98d4…` in both the F1 experiment and this verification). Only
Creative changes, which is the point.

## Why the backend planner stays the source of truth

The fix deliberately does **not** teach the client that Creative means
`realesr-general-x4v3`, or that its denoise is 0.25. The client only learns to
say *"the user did not choose"* — and the backend answers what that means.

`mode_planner`'s own docstring gives the reason:

> Centralising them means the API, the worker and the UI cannot drift apart
> about what "Creative" is.

Copying the mode table into the store would have produced two definitions that
drift — the precise failure the module exists to prevent. So `CREATIVE_MODEL`
and `CREATIVE_DENOISE` appear nowhere in `frontend/src`.

## Explicit-override semantics

Preserved exactly:

```
explicit model   > mode default > default_model
explicit denoise > mode default > none
```

**Absence is by key, never by truthiness.** `denoiseChosen` is a separate
boolean, not `denoiseStrength > 0`, because `0.0` is a real setting — fully the
`wdn` weights — and a truthiness check would silently reclassify it as unset.
Two tests lock this: one in the store, one in `buildSettings`.

## Backward compatibility

Unaffected. A client that sends `model` and `denoiseStrength` keeps getting
exactly what it asked for; the backend was never changed, and the precedence
rule is unchanged. Old jobs with no `mode` still read as `SCALE`/no-mode.

Verified by the pre-existing suite passing unchanged (`test_an_explicit_model_overrides_the_mode`,
`test_an_explicit_denoise_survives_a_mode`, `test_a_request_with_neither_mode_nor_target_is_unchanged`)
plus a new integration test asserting an explicit model still beats the mode
against real weights.

## Files changed

| File | Change |
|---|---|
| `frontend/src/stores/useEnhancementStore.ts` | `modelChosen` / `denoiseChosen` state; set in `setModel` / `setDenoiseStrength`; **not** set by `adoptDefaults` |
| `frontend/src/types/job.ts` | `CreateJobRequest.model` → optional |
| `frontend/src/services/jobsApi.ts` | append `model` only when defined |
| `frontend/src/features/enhance/useEnhanceJob.ts` | `buildSettings` takes `denoiseChosen`; request spreads `model` only when chosen |

Tests added:

| File | Tests |
|---|---|
| `frontend/src/stores/useEnhancementStore.test.ts` | 7 — chosen-tracking, explicit zero, `setMode` stays one-field |
| `frontend/src/services/jobsApi.test.ts` | 5 — new file; the wire payload |
| `frontend/src/features/enhance/enhance.test.tsx` | 4 — `buildSettings` omission rules |
| `backend/tests/integration/test_job_pipeline.py` | 2 — end-to-end, `slow` + `integration` |

**No production file under `backend/app`, `models/` or `pyproject.toml` was
modified.**

## Tests

Written **before** the fix. Six failed against the unfixed client, which is what
made them worth writing:

- 4 store tests — `setModel` / `setDenoiseStrength` did not record a choice;
- `createJob` sent `model` unconditionally;
- `buildSettings` sent an untouched `denoiseStrength`.

After the fix all pass.

**An honest limitation.** The two new *integration* tests exercise the backend,
which was already correct — they would have passed before the fix too. They are
regression **guards** against the wiring bug returning, not detectors that would
have caught it originally. The tests that genuinely fail-before-fix are the six
client-side ones. The defect was client-side, so its detectors have to be.

| Gate | Result |
|---|---|
| backend pytest | **715 passed, 2 skipped, 0 failed** (was 713; +2 integration) |
| ruff check / format | clean / 126 files |
| mypy app / benchmarks | 53 / 15 files, no issues |
| frontend vitest | **442 passed, 25 files** (was 427; +15) |
| typecheck / lint / build | exit 0 / exit 0 / built |

## Real end-to-end verification

Four jobs through the real path — HTTP → `JobService` → SQLite → worker →
real `EnhancementService` → real weights → PNG — with the payloads the fixed
client now emits. Same 96×72 source (sha256 `3101ee8ca8174bbe…`), 4x, PNG,
sharpening 0.

| Job | Payload | Resolved model | Denoise | Output SHA-256 | Bytes |
|---|---|---|---|---|---|
| **A** Standard | mode only | `RealESRGAN_x4plus` | `None` | `e87625b45a7e98d4…` | 110 408 |
| **B** Creative | mode only | **`realesr-general-x4v3`** | **`0.25`** | `b367edc81434423a…` | 149 968 |
| **C** Creative + explicit model | `model=RealESRGAN_x4plus` | `RealESRGAN_x4plus` | `None` | `e87625b45a7e98d4…` | 110 408 |
| **D** Creative + explicit denoise | `denoiseStrength=0.8` | `realesr-general-x4v3` | **`0.8`** | `28770e1441a20482…` | 111 239 |

- **A vs B: outputs DIFFER** — max pixel delta 84, mean 3.29. All jobs reached
  `completed`; 384×288 PNG throughout; no errors.
- **C**: explicit model honoured — the mode did not override it.
- **D**: explicit denoise honoured — `0.8`, not `CREATIVE_DENOISE`.

The strongest single confirmation: **B's hash `b367edc81434423a…` is byte-identical
to F1's control arm** (`3-creative-bare`, the request with no model and no
settings). The fixed client now produces exactly what Creative was always meant
to produce. And **A's hash matches F1's Standard arm exactly**, so Standard is
provably unchanged.

## Known follow-up, not fixed here

**The model dropdown can now disagree with what runs.** In Creative with an
untouched dropdown, the UI displays the adopted default (`RealESRGAN_x4plus`)
while the job correctly runs `realesr-general-x4v3`. Standard is unaffected —
its planner default is the same model the dropdown shows.

Not fixed because every remedy is out of this task's scope: showing "Auto" would
change labels; making the dropdown display the mode's model would either
duplicate the planner in the client or need a new API field. It is a **display**
inconsistency, not a behavioural one — the produced image is now correct in
every case. It should be resolved before Creative is promoted in the UI.

## Out of scope, confirmed untouched

- **`DEFAULT_DENOISE = 1` is unchanged.** Whether the slider should start at
  0.25 is Phase 4 **F2**, and it needs the visual comparison F2 specifies.
  After this fix, Creative's default *reaches* `CREATIVE_DENOISE = 0.25`, which
  is what the mode always intended; the slider's own starting value is a
  separate question.
- F2, F3, F4 and F5 were **not** run.
- No model added, none downloaded, no benchmark run, no dependency added.
