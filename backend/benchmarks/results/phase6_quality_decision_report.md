# Phase 6 — quality decision and production-readiness audit

**Audit only. No production file was modified.** Nothing under `backend/app`,
`frontend/src`, `models/` or any project config was touched. No default was
changed, no candidate integrated, nothing committed, nothing pushed. Phase 7 was
not started.

Audited at `acf97b3`, working tree and index clean, 17 untracked research
artifacts from Phases 3A/3B/3C preserved untouched.

---

## 1. What this phase is for

Nine research phases have produced recommendations. Some were acted on, some
were deliberately not, and the reasons live in nine separate reports. This audit
puts every one of them in one table, states what the production code actually
does today, and marks the places where the two disagree.

It changes nothing. Where evidence supports a change, this report proposes it
and stops.

## 2. Evidence reviewed

Read in full — report, measurements and the implementation each one describes —
rather than from summaries:

| Phase | Question | Outcome |
|---|---|---|
| 2 | image-quality improvements | shipped |
| 2.5 | real-photo denoise sweep | recommended 0.25, not applied |
| 3 | model benchmark | keep `x4plus`; `general-v3` as Creative; anime unsuitable for photos |
| 3A | adaptive denoise | rejected adaptive; recommended chroma prefilter for 3B |
| 3B | chroma prefilter × DNI | **rejected** the chroma prefilter |
| 3C | HAT | **rejected** on this hardware |
| F1 | mode wiring | fixed; modes now reach the worker |
| F2 | denoise benchmark | 1.0 not defensible; 0.25 preferred |
| F3 | sharpening benchmark | 0.25 defensible as a value; default not changed |
| F4 | tile size | keep 256; tile 64 is a visible defect |
| F5 | detail recovery | **rejected** for the three candidates tested |

## 3. Decision matrix

Confidence is graded on evidence, not on enthusiasm. "Proven" means measured
with a blind or reference-based test on the real corpus; "promising" means
measured but with a known gap; "inconclusive" means the evidence does not
separate the options.

| # | Feature / change | Evidence | Quality benefit | Artifact risk | Performance | Memory | Confidence | Decision |
|---|---|---|---|---|---|---|---|---|
| A | `RealESRGAN_x4plus` | Phase 3, F4 | Best skin, no restyling, never fails badly | Lowest of any arm | ~21–31 s / 32 MP | 767 MB peak | Proven | **Production-ready — keep as Standard** |
| B | `realesr-general-x4v3` | Phase 3, F2 | Only arm beating the reference on detail | +182% flat noise at dn0; needs 0.25 | 4.1× faster | ~1/17 VRAM | Proven | **Production-ready — keep as Creative at 0.25** |
| C | `RealESRGAN_x4plus_anime_6B` | Phase 3 | Correct for illustration | Restyles photos; +28.7% edge overshoot, highest measured | n/a | n/a | Proven | **Research/advanced only — never a photographic default** |
| D | DNI weight interpolation | 2.5, 3A, 3B, F2 | The effective luma denoiser; free at inference | None | 0 ms | 0 | Proven | **Production-ready — keep (already shipped)** |
| E | Unsharp-mask sharpening | F3, F5 | Visible texture gain at 1:1 in F3; **not reproduced in F5** | Ringing/halo ≥0.5; grain from 0.15 | ~1.1 s / 32 MP | +96 MB | Inconclusive on default | **Implementation production-ready; default stays 0** |
| F | Adaptive luma denoise | 3A | Estimator works | Mapping r = +0.077 to its premise; destroys skin on noisy portraits | 11.7 s / 16 MP | — | Proven negative | **Rejected** |
| G | Chroma prefilter | 3A, 3B | None found | +0.4% chroma noise, wrong direction; visible degradation | 25 ms | 46 MB | Proven negative | **Rejected** |
| H | HAT | 3, 3C | Slightly better skin only | Visible tile seam at forced tile 128 | 5.35× runtime | 2907 MB peak, fp32 only | Proven negative | **Rejected** |
| I | Detail recovery (guided / gated / DoG) | F5 | None in any region | +40% flat-region noise; visible shadow grain | 1.5–3.7 s | small | Proven negative | **Rejected** |
| J | Tile size 256 | F4 | Reference arm; fastest | None visible | 27.4 s median | 767 MB | Proven | **Production-ready — keep as default** |
| J2 | Tile size 128 | F4 | Visually equivalent in ordinary content | Faint seam, not visible in use; 40% HF loss on one low-contrast field | +75% runtime | 260 MB | Proven | **Acceptable fallback — do not promote to default** |
| J3 | Tile size 64 | F4 | None | **Visible rectangular block artifact**, geometry verified | 3× runtime | 115 MB | Proven negative | **Last resort only — keep, do not remove from ladder** |
| K | 8x cascade (4x + 2x) | audit §6 | Fully neural composition | None found | two passes | guarded per stage | Proven by audit | **Production-ready — no change** |
| L | 16x cascade (4x + 4x) | audit §6 | Same weights twice, consistent character | None found | two passes | guarded per stage | Proven by audit | **Production-ready — no change** |
| M | Target resolution 2K–16K | audit §7 | Exact sizes, aspect preserved | None found | — | intermediate checked | Proven by audit, one gap | **Production-ready — one contract gap noted** |
| N | JPEG 4:4:4 | code audit | Full chroma; subsampling never touches luma | None | — | — | Proven | **Production-ready — keep** |
| O | PNG output | code audit | Lossless path | None | — | — | Proven | **Production-ready — keep** |
| P | Preview pipeline | code audit | Never enlarges; atomic write | None found | — | bounded at 4096 px | Proven by audit | **Production-ready — no change** |
| Q | Memory / OOM ladder | F4, code audit | Job completes instead of failing | Tile 64 rung is a visible defect | — | per-stage limit enforced | Proven | **Production-ready — keep ladder intact** |

## 4. Where metric and eye disagree

Recorded rather than reconciled. Four disagreements are on record and every one
of them changed a decision.

**High-frequency energy is not quality.** F3 established it and F5 confirmed it
from the other side. In F5 every candidate raised Sobel by up to 26% and HF
ratio by up to 76% while no region looked better to the eye.

**Texture statistics are not correct texture.** F5's guided arm had the best
texture error of any arm on all five photographs while losing PSNR and SSIM to
baseline on all five. It added roughly the right amount of texture in the wrong
places. Without the reference track this would have read as recovery.

**An input-side measurement is not an output-side result.** Phase 3A measured a
−7.5% to −37.8% chroma-noise reduction and recommended confirming it. Phase 3B
confirmed the opposite: +0.4% after SR, and a visible degradation. 3A's own
caution is what prevented a bad integration.

**A synthetic probe can miss the mechanism.** F5 predicted from synthetic noise
that two candidates would be gentler than the incumbent sharpener; on
photographs they were forty times worse in flat regions. The cause was the
sharpener's ±2-level dead zone, which the synthetic noise at σ = 6 never
probed.

**And F3 and F5 disagree with each other about sharpening.** F3 found a
blind-validated visible benefit at 0.25 at 1:1 on textured subjects. F5's visual
pass found the same operator equivalent to baseline on all seven regions it
inspected. Neither refutes the other — they inspected different regions for
different questions — but the disagreement is unresolved and it is the reason
item E is graded *inconclusive* rather than promoted.

## 5. Production defaults as they stand today

Read from the code, not from the reports.

| Setting | Value in code | Source |
|---|---|---|
| Standard model | `RealESRGAN_x4plus` | `mode_planner.STANDARD_MODEL` |
| Standard denoise | `None` — model has no pair, so it never applies | `mode_planner.plan_mode` |
| Creative model | `realesr-general-x4v3` | `mode_planner.CREATIVE_MODEL` |
| Creative denoise | `0.25` | `mode_planner.CREATIVE_DENOISE` |
| Schema denoise default | `None` | `schemas/job.py` |
| Sharpening default | `0.0` | `schemas/job.py` |
| Tile size / pad | `256` / `16` | `core/config.py` |
| Default scale | `4` | `core/config.py` |
| Max output pixels | `200_000_000` | `core/config.py` |
| Supported scales | `2, 4, 8, 16` | `enhancement_service.SUPPORTED_SCALES` |
| JPEG chroma | `4:4:4` (`subsampling=0`) | `image_service` |
| Preview | max edge 4096, quality 92, never enlarges | `image_service` |

Standard and Creative are both correctly wired and both match their evidence.
**Creative at 0.25 is exactly what F2 recommended**, and F1 proved by hash that
the mode reaches the worker.

### The one default that does not match its evidence

There is no `DEFAULT_DENOISE` constant in the backend. The shipped default is
the schema's `None`, and F2 established — with Phase 3B confirming byte
identity — that **`None` and DNI `1.00` are the same code path and the same
pixels.**

F2 measured DNI 1.00 as costing a median **−49.9% of high-frequency energy**,
−9.0% local contrast, and rendering skin visibly plastic in a blind pass. Phase
2.5 reached the same conclusion independently. Neither phase has been
contradicted.

That value is unreachable through the two mode paths — Standard's model has no
denoise pair, and Creative supplies 0.25 — but it **is** reachable one way:

1. the client sends an explicit `modelId` of `realesr-general-x4v3` with no
   mode, or with the denoise slider untouched;
2. `buildSettings` omits `denoiseStrength` when the user has not moved the
   slider, by design, so the mode default can win;
3. the backend receives `None`, and `None` is DNI 1.00.

Compounding it, the frontend slider's resting position is
`DEFAULT_DENOISE = 1` in `stores/useEnhancementStore.ts`, so a user who opens
the advanced controls sees denoise displayed at 100% — the setting F2 called not
defensible — and submitting without touching it produces exactly that.

**Proposal, not applied.** Make the no-mode default for a denoise-capable model
resolve to 0.25 rather than `None`, and move the slider's resting position to
match. Evidence: F2 (blind visual pass plus corpus sweep) and Phase 2.5, both
recommending 0.25, and Creative already runs it in production without
complaint. This is a behavioural change to the produced image and belongs in its
own task with its own regression tests — it is **not** part of this audit.

## 6. Cascade audit — 8x and 16x

No bug found. No change proposed.

Routing is not duplicated anywhere: `supported_scales` derives its answer by
calling `plan` and catching the refusal, so the list the API publishes and the
rule a job is validated against cannot drift. `plan_cascade` likewise calls
`plan` rather than restating it.

8x is `x4plus` then `x2plus`; 16x is the same 4x weights twice. The comment
explaining why 16x is not `4x → 2x → 2x` is correct — three passes would put a
forward pass over an image four times larger for no gain.

The memory guard is applied per stage in `_assert_stage_fits`, raising the typed
`OutputTooLargeError` naming which pass failed, so a 16x job that cannot finish
fails loudly instead of returning the 4x intermediate as if it were the answer.
The guard is redundant for anything that passed submission, which the docstring
says, and that redundancy is correct rather than wasteful.

Denoise across a cascade is handled explicitly: `_denoise_for` drops the setting
for a pass whose model has no pair, and applies the same blend to both passes of
a 16x job. The docstring correctly flags that whether a second pass *should*
denoise an already-denoised image is an unanswered quality question. **It
remains unanswered.** No phase has measured it.

## 7. Target resolution audit — 2K to 16K

No bug found. One contract gap.

The planner is sound. The preset fixes the long edge and the short edge follows
the source, so aspect ratio survives. It picks the smallest supported neural
factor that reaches the target, then resamples down — supersampling, which
discards model pixels rather than inventing them, and the report of this is
honest about the distinction. Enlargement is refused rather than answered with a
silent downscale. The pixel limit is checked against the **neural intermediate**,
which is the largest thing the job holds, not the smaller final target.

Frontend and backend agree. `targetPlanning.ts` mirrors the backend and says so,
and `test_the_frontend_preset_table_matches_this_one` reads
`frontend/src/types/job.ts` and asserts the long edges match. They do: 1920,
3840, 5760, 7680, 15360.

**The gap:** `MAX_OUTPUT_PIXELS` is duplicated — `200_000_000` in
`backend/app/core/config.py` and `200_000_000` in `frontend/src/config/limits.ts`
— and **no contract test binds them**. The preset table is protected from drift;
this constant is not. The values agree today. Changing the backend limit alone
would silently leave the UI offering targets the server refuses.

Proposal: extend the existing contract test to read the frontend constant, the
same way it already reads the preset table. It is a test-only change and it is
not made here.

## 8. Model strategy

**Standard — `RealESRGAN_x4plus`, denoise not applicable.** Chosen because it is
the only arm that never fails badly. It does not win every metric; Phase 3 was
explicit that `general-v3-dn0` beats it on landscape detail while nearly
tripling flat-region noise and losing heavily on skin. A general-purpose default
should not make that trade silently. The honest cost is that it is roughly 4×
slower for a lead that is content-dependent.

**Creative — `realesr-general-x4v3` at 0.25.** The only arm that produced more
detail than the reference anywhere, at 4.1× the speed and a fraction of the
VRAM. Its weakness is noise amplification, which Creative exposes as a control
rather than hides, and 0.25 is the F2-confirmed setting that keeps it in check.

**Anime — keep available, never a photographic default.** It restyles rather
than restores. Its −55% flat noise reads as clean output until you see that
texture was replaced, and its +28.7% median edge overshoot is the highest of any
arm measured. Correct for illustration, which is what it was trained for.

**HAT — rejected, and the facts that decided it should not be lost:** fp16
produces NaN so fp32 is required; ~2907 MiB peak; ~5.35× runtime; tile forced to
128; a visible tile seam at that tile size; and no net perceptual win — six of
eight gate questions negative or marginal on a 4 GB RTX 3050. What would change
the answer is a card with ≥8 GB, or an fp16-stable variant. Neither is the
target hardware.

**Model routing by content — not justified.** Phase 3C found the inter-arm
differences too small and inconsistent to support it, and routing would add a
classification step and a second resident model to a card that already
OOM-ladders on `x4plus`.

## 9. Sharpening

Production default is `0.0`. **It should stay there for now**, and the reason is
a disagreement rather than a weakness in F3.

F3's evidence for 0.25 is real: blind-validated texture benefit at 1:1, no
visible ringing or halo on the hardest edges in the corpus, skin and text
equivalent to off, +21.6% dark-region HF that F3 judged measurable but not
objectionable, at ~3% of the neural pass.

| strength | F3 finding |
|---|---|
| 0.15 | visible benefit on texture, smallest dark-region cost (+17.3%) |
| 0.25 | F3's recommended value; no visible ringing or halo |
| 0.50 | visibly worse on low-light; not a defensible default |
| 0.75 | clearly worse on three of five photographs; ringing and halo visible |

What has changed since F3 is that **F5 inspected the same operator at 0.25 on
seven regions and found it equivalent to baseline everywhere.** F5 was not
designed to test sharpening and its regions were chosen for a different
question, so this does not refute F3. But two phases now disagree about whether
the benefit is visible, and enabling a default that changes every user's output
on contested evidence is not warranted.

**0.25 stays a candidate value.** If sharpening is ever enabled by default, F3
supplies the number and 0.15 is the more cautious alternative. Neither should be
switched on without a tie-break: a blind pass on the regions F3 used and the
regions F5 used, judged by someone other than the author of both.

## 10. Denoise

**Is DNI worth keeping?** Yes. Phase 3A tested the alternative directly and no
fixed prefilter beat DNI 1.00's median ratio of 0.65, while NLM cost 11.7 s at
16 MP against DNI's zero. DNI is free at inference because it interpolates
weights, not pixels.

**What default?** 0.25, on the asymmetry Phase 2.5 argued and F2 confirmed:
under-denoising is recoverable by the user, erased texture is not. F2 was
explicit that 0.25 and 0.50 are not separable on the evidence — 0.50 removes
about 50% more noise for about 40% more high-frequency cost — and reported it as
a trade-off rather than crowning a winner. Creative already ships 0.25.

**Adaptive preprocessing?** No. The estimator works; the mapping from estimate
to strength correlates at r = +0.077 with its own premise, and its concrete
failure mode is destroying skin texture on noisy portraits — worse than the
fixed default it would replace.

**Chroma prefilter?** No. Rejected on five independent axes in 3B, and not
parked as research-only because there is no open question: 110 cells plus a
visual pass across five photographs, two models and five DNI rungs all agree.
Cost was never the objection; it does not work.

**Is there evidence to change a production default?** Yes, for the one gap in
§5, and it is the only denoise change this audit supports.

**DNI's two structural gaps remain**, both identified in 3A and neither closed:
it reaches only `realesr-general-x4v3`, so Standard has no denoising available
at all; and it is luma-dominant, leaving chroma mottle largely intact. 3B
established that the obvious fix for the second gap does not work. Nothing
addresses the first.

## 11. Tile size

| | tile 256 | tile 128 | tile 64 |
|---|---|---|---|
| Quality | reference | equivalent in ordinary content | **visible block artifact** |
| Determinism | bit-exact, 2 passes | bit-exact | bit-exact |
| Seams | sub-level; not visible | faint, not visible in use | hard edges on both axes |
| Runtime | 27.4 s median | +75% | ~3× |
| Peak VRAM | 767 MB | 260 MB | 115 MB |

**tile 64's artifact is real and geometrically verified.** On portrait skin it
produces a rectangular block whose corner falls at exactly the tile grid, with
step ratios of 2.43× horizontally and 2.56× vertically against roughly 1.0× and
1.35× for the coarser arms, and a 6.4× texture jump across the joint.

**tile 128's quality loss** is not the seam. It is a 40% high-frequency drop on
a low-contrast dark field, visible at 1:1 on `text-signage`. Elsewhere it is
equivalent.

**tile 256's memory risk is the live caveat.** Free VRAM before the tile-256
passes ranged 2191–3299 MB, and the clamp drops the ceiling to 192 at or below
2048 MB. The narrowest margin observed was **143 MB**, on an otherwise idle GPU.
Another consumer taking half a gigabyte silently clamps it.

**Hardware portability:** everything above is one model, one scale, one 4 GB
RTX 3050. Tile 256 held every time it was requested here; that is not a
guarantee on any 4 GB card, and F4 says so.

**Keep the ladder intact.** Its lower rungs are not graceful degradation, but a
job that finishes with a visible artifact is better than a job that fails, and
that is the trade the ladder exists to make.

## 12. Detail recovery

Rejected in F5 and **not reopened here.**

Why: with ground truth available, all three candidates moved the reconstruction
further from the original on both fidelity metrics on all five photographs, and
none produced a visible improvement in any inspected region. The strongest
single piece of evidence is the split — the guided arm had the best texture
error of any arm while losing PSNR and SSIM to baseline everywhere, which is
added texture in the wrong places.

Limitations, restated so the rejection is not over-read: three operators, one
calibration, one corpus, one degradation pipeline, one observer, and a ground
truth that is itself a JPEG. F5 explicitly does not claim detail recovery is
impossible.

**The dead-zone hypothesis is untested.** A candidate with a threshold matched
to the sharpener's would plausibly cut the flat-region noise cost. That is a
statement about criterion 3 only; it says nothing about whether the added detail
lands correctly, which is criterion 6 and the one the reference track settled.

**What would justify reopening:** a candidate that passes the reference track —
that is, improves or preserves PSNR and SSIM against ground truth on the
degradation corpus. Nothing less. A no-reference metric moving in a pleasing
direction is specifically what F5 was built to disbelieve.

## 13. Biggest remaining quality risks

1. **The denoise default gap in §5.** The only place where shipped behaviour
   contradicts settled evidence. A user selecting the Creative model outside
   Creative mode gets the setting two phases called not defensible.
2. **Standard has no denoising available at all.** `x4plus` has no denoise pair,
   so a noisy photograph in Standard mode gets no noise handling whatever. 3A
   identified this; nothing has addressed it.
3. **Sharpening is unresolved, not settled.** F3 and F5 disagree and the default
   is off, which is the safe side, but the question is open.
4. **The tile-256 VRAM margin is 143 MB.** Any other GPU consumer silently
   moves users to tile 128 and its texture cost.
5. **`MAX_OUTPUT_PIXELS` is duplicated without a contract test.**
6. **The Creative model dropdown displays the wrong model.** F1 documented it:
   in Creative with an untouched dropdown the UI shows `RealESRGAN_x4plus` while
   the job correctly runs `realesr-general-x4v3`. Display-only, behaviour is
   right, and F1 said it should be resolved before Creative is promoted.
7. **Second-pass denoise in a 16x cascade is unmeasured.** The code applies the
   same blend twice and says openly that whether it should is unanswered.

## 14. The three experiments worth running next

Chosen because each one closes a decision that is currently blocked, not because
they are interesting.

### N1 — Sharpening tie-break

| | |
|---|---|
| Hypothesis | Sharpening at 0.15–0.25 produces a visible benefit on textured subjects at 1:1 that survives an inspection designed not to favour it |
| Why it matters | The only way to close the F3/F5 disagreement, which currently blocks the one enhancement the codebase already owns |
| Upside | A defensible on-by-default setting, at ~3% of the neural pass |
| Downside | Confirms the benefit is not reliable; sharpening stays off, which is the status quo |
| Corpus | Existing five, no change |
| Metrics | F3's set, reported but not decisive |
| Blind test | **Required and central** — both F3's regions and F5's, shuffled together so region choice cannot drive the answer |
| Success | Benefit visible at 1:1 on a majority of textured regions, with no region worse |
| Failure | Equivalent on a majority, or any region worse |
| Runtime | ~15 min, 5 photographs × 3 arms, neural pass cached |
| Touches production | No |
| New dependency | No |

### N2 — Denoise for Standard

| | |
|---|---|
| Hypothesis | A noisy photograph in Standard mode can be improved without a denoise pair, by a pre-SR luma step or by routing to Creative at 0.25 |
| Why it matters | Standard is the default mode and has no noise handling at all; risk 2 above |
| Upside | Closes the largest structural gap in the pipeline |
| Downside | 3A already refuted fixed luma preprocessing; this must not repeat it — the arm is routing, not filtering |
| Corpus | Existing five; `low-light-noise` is the decisive image, and the corpus has only one |
| Metrics | Flat-region noise, HF ratio, skin texture, plus reference-based scoring on F5's degradation harness |
| Blind test | Required |
| Success | Visibly cleaner low-light with no skin or texture loss, against Standard baseline |
| Failure | Any skin degradation, or no visible low-light gain |
| Runtime | ~25 min |
| Touches production | No — routing is simulated in the benchmark |
| New dependency | No |
| **Caveat** | One noisy photograph is thin evidence for a default. This experiment likely reports "needs a broader corpus" rather than a verdict, and that is a legitimate outcome |

### N3 — Tile 256 headroom on a loaded GPU

| | |
|---|---|
| Hypothesis | Under realistic desktop GPU load, free VRAM falls below 2048 MB often enough that tile 128 is the common path rather than the exception |
| Why it matters | Every F4 quality result assumes tile 256 was granted. If it usually is not, the shipped quality is tile 128's, including its 40% HF loss on low-contrast fields |
| Upside | Either confirms the default is what users get, or reveals the clamp threshold needs revisiting |
| Downside | Cheap either way |
| Corpus | Existing five |
| Metrics | Free VRAM before each pass, effective tile, honoured rate — all already recorded by the F4 harness |
| Blind test | Not applicable; this is an instrumentation question |
| Success | Either a measured honoured-rate under load, or a documented reason the threshold should move |
| Failure | Cannot construct a realistic load; report the limitation |
| Runtime | ~20 min |
| Touches production | No |
| New dependency | No |

**Deliberately not chosen.** Community RRDBNet fine-tunes: no way to evaluate
licence and provenance within a research phase, and Phase 3 already showed model
swaps trade one failure mode for another. Adaptive DNI: 3A refuted the mapping
and nothing has changed. Quality-aware postprocessing: F5 just rejected the
closest thing to it, and reopening without a reference-track pass would repeat
the mistake F5 was built to catch.

## 15. Validation

| check | result |
|---|---|
| backend pytest | **805 passed, 2 skipped** (full suite: unit, api, integration) |
| backend ruff check | **1 error** — pre-existing, research file |
| backend ruff format | **1 file** would be reformatted — same file |
| backend mypy, `app/` | **clean**, 53 source files |
| backend mypy, `benchmarks/` | **clean**, 20 source files |
| backend mypy, `tests/` | **29 errors** in 10 files, all pre-existing |
| frontend vitest | 442 passed, 25 files |
| frontend typecheck | clean |
| frontend lint | clean |
| frontend build | succeeds, chunk-size advisory only |

### Failures, classified rather than hidden

**Research-file lint failure, introduced by me in F4.**
`backend/tests/unit/test_benchmark_tile_quality.py:13` fails `ruff check` with
I001 (unsorted import block) and `ruff format --check`. It was committed in
`74f3fdb`. The cause is mine: during F4 I ran neither tool, and during F5 I ran
both but only against the two F5 files. It is a research test file, it does not
affect production, and the 52 tests in it pass. **Not fixed here** — this phase
is audit-only and the file is already committed. The fix is `ruff check --fix`
plus `ruff format` on that one path, in its own change.

**Pre-existing test typing errors.** 29 mypy errors across 10 test files, the
largest group being 10 in `test_cleanup_service.py`. None are in production
code, none are in the F4 or F5 test files, and none were introduced by any
research phase. They predate this work. Not addressed here.

**No production failure of any kind.** `app/` is mypy-clean and every frontend
check passes.

## 16. Scope and limitations of this audit

- **It is a reading, not a re-measurement.** No experiment was rerun. Every
  number quoted is from a committed or preserved measurements file.
- **The disagreements are recorded, not resolved.** F3 versus F5 on sharpening,
  and 3A versus 3B on chroma, are both left standing with the decision explained.
- **One observer stands behind most visual verdicts**, across every phase.
- **One GPU, one scale, one corpus of five photographs** underlies nearly all of
  it. Every hardware claim is an RTX 3050 4 GB claim.
- **Nothing here authorises a production change.** The two proposals — the
  denoise default in §5 and the contract test in §7 — are proposals.
