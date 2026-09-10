# Phase 10 — product readiness audit

**Audit only. No production file was modified.** No code was changed, no test
was changed, nothing was committed or pushed. One temporary reproduction test
was written to confirm a finding by rendering rather than by reading, and
deleted immediately after; the working tree is byte-identical to where Phase 9
left it apart from this report.

Audited at `acf97b3`, with the Phase 7/8/9 changes present in the working tree.

---

## 1. Executive summary

AuraScale is in better shape than a product at this stage usually is. The
protections that matter for a local tool are present and mostly correct: path
traversal is closed by a hex whitelist, uploads are checked by magic bytes
before decode, the queue is bounded, cancellation is honoured at every stage
since Phase 8, a restart marks interrupted jobs failed instead of leaving them
spinning, and retention sweeps orphans. Phases 6, 8 and 9 each looked for
problems and mostly found the work already done.

**One real bug was found, and it is a user-facing one.** In Creative mode with
an untouched model dropdown, the interface shows `Real-ESRGAN x4 Plus`, disables
the noise-reduction slider, and states that the model has no denoise weights.
The job then runs `realesr-general-x4v3` at denoise 0.25. Every one of those
three statements is wrong for the job that actually runs, and the user cannot
reach the denoise control in the mode where it matters most. Phase 4 F1
documented the display half of this and said it should be resolved before
Creative is promoted; the control half appears not to have been noticed.

Everything else found is either a known limitation that is correct for a local
tool, or architectural work that only becomes necessary at deployment. Nothing
else qualifies as a bug.

**Verdict.** Safe for temporary public testing with a trusted audience once the
Creative mismatch is fixed or Creative is hidden. Not ready for unattended
public deployment, and the gap there is not defects but absent
infrastructure: no authentication, no rate limiting, an in-memory queue and
event broker that cannot survive a second API instance, and SQLite.

## 2. Current architecture

```
Browser
  → Vite dev server (5173, /api proxied)
      → FastAPI (127.0.0.1:8001)
          → asyncio.Queue → ThreadPoolExecutor(max_workers=1)
              → Real-ESRGAN on RTX 3050 4 GB, CPU fallback
          → SQLite via aiosqlite
          → local filesystem: inputs, outputs, previews, thumbs
```

Single process, single GPU, single writer. Progress reaches the browser over
SSE from an in-memory broker. Public testing is done by pointing a Cloudflare
Quick Tunnel at the Vite dev server, which proxies only `/api`, so the backend
is never directly addressable.

## 3. Product and UX findings

### P1 — Creative mode shows the wrong model and disables a control that works

**Severity: HIGH. MUST FIX BEFORE PUBLIC.**

**Where.** [`EnhancePanels.tsx:34-38`](frontend/src/features/enhance/EnhancePanels.tsx#L34-L38),
[`useEnhanceJob.ts:89`](frontend/src/features/enhance/useEnhanceJob.ts#L89),
[`EnhancementControls.tsx:156-186`](frontend/src/features/enhance/EnhancementControls.tsx#L156-L186).

**Reproduction.** Confirmed by rendering `EnhancePanels` with mode `creative`,
two models stubbed on `GET /api/models`, and the dropdown untouched:

| observed | value |
|---|---|
| `modelId` in the store | `RealESRGAN_x4plus` |
| `modelChosen` | `false` |
| denoise slider | carries `data-disabled` |
| denoise description | "Real-ESRGAN x4 Plus has no denoise weights to blend." |
| model the job actually runs | `realesr-general-x4v3` at denoise 0.25 |

**Mechanism.** `adoptDefaults` picks the first downloaded model so the panel has
something selected, and deliberately leaves `modelChosen` false. `buildSettings`
omits `model` while `modelChosen` is false, which is what lets the backend's
mode planner choose — that part is correct and was the Phase 4 F1 fix. What was
not carried through is that the panel derives everything else from
`selected = models.find(m => m.id === modelId)`, and `modelId` is the adopted
default, not the mode's model. So `supportsDenoise` is read off the wrong model.

**Impact.** Three separate wrongs from one cause. The model name displayed is
not the model that runs. The noise-reduction slider is disabled even though the
running model supports it, so Creative users cannot change the one setting
Creative exists to expose. And the explanatory sentence names a model and makes
a claim that is false for this job.

**Recommended fix.** Resolve the displayed model from the mode when the user has
not chosen one, so the panel and the planner agree. The planner already owns
this decision in `mode_planner.plan_mode`; the client needs the same answer,
either by mirroring it or by having the API report the mode's model. Showing
"Auto" is a weaker alternative that fixes the lie but not the disabled slider.

**Regression test: yes.** Assert that in Creative with an untouched dropdown the
denoise slider is enabled and the displayed model matches what a submission
would run.

### P2 — No completion feedback in the log for a user-reported slow job

**Severity: LOW. CAN DEFER.** Covered under observability as O1.

### Verified correct, not findings

Settings survive a model change: `setModel` snaps the factor to the nearest
supported one rather than resetting it. Scale and target are mutually exclusive
and the disabled one stays visible rather than vanishing, which reads as
alternatives rather than a control disappearing. Locale switching updates every
Enhance string live, proven by Phase 9's rendered tests. Error panels translate
from the stable `code`, never from the server's English sentence.

## 4. API findings

No bugs found. The contract behaves as documented.

Validation is typed throughout: 415 for a file whose bytes are not an image,
422 for a malformed settings blob or an out-of-range value, 404 for an unknown
job, 409 for a result that does not exist yet, 413 for an oversized upload.
Phase 8 added 54 tests covering these including path-shaped job ids, truncated
PNGs, renamed text files and floods.

**Repeated DELETE** returns 204 the first time and 404 afterwards, because the
job is gone. Not idempotent in the strict sense; acceptable and arguably more
informative than a silent 204.

**Repeated download and preview** are plain `FileResponse` reads and are safe to
repeat.

**`quality` is ignored for PNG** rather than refused. This looked like a missing
check during the Phase 8 audit and is not: the field controls nothing for a
lossless format, and refusing a request over an ignored field would reject good
submissions. Validated on the lossy path, which is where it means something.

**Legacy requests** — a submission with no mode, no model and no settings — are
the oldest shape and still resolve correctly, now to `DEFAULT_DENOISE` for a
denoise-capable model after Phase 7.

## 5. Resource and performance findings

### R1 — Storage cap is checked before the job, not against what the job will produce

**Severity: MEDIUM. MUST FIX BEFORE PRODUCTION.**

**Where.** [`storage_service.py:135-146`](backend/app/services/storage_service.py#L135-L146),
called from `job_service.create` with no `incoming_bytes`.

**Reasoning.** `assert_capacity()` accepts an `incoming_bytes` argument and the
submission path does not pass one. At 9.9 GB used against a 10 GB cap the check
passes, and a 16x PNG result can then be hundreds of megabytes. The cap is a
soft floor, not a quota.

**Impact.** On a local machine, bounded by retention and by how fast one GPU
produces results. On a public deployment with several users it is the wrong
shape of limit.

**Recommended fix.** Pass the upload size, and project the output from the
planned scale and format. Both numbers are known at submission.

**Regression test: yes**, once fixed.

### Worst-case paths traced, no leak found

A 16 MP input at 16x reaches roughly 4 GP, which `MAX_OUTPUT_PIXELS` refuses at
200 MP before any work starts, and the check is applied to the **neural
intermediate** rather than the smaller final target — the largest thing the job
holds. Per-stage limits are re-applied inside the cascade by
`_assert_stage_fits`, so a two-pass job that cannot finish fails with a typed
error naming the stage rather than returning the 4x intermediate as the answer.

Encoding runs on a thread, so the event loop keeps serving. The model stays
resident by design and `release_cuda_memory()` runs in a `finally` between jobs,
which is what stops allocator fragmentation from failing the next job at a tile
size this one managed.

## 6. Concurrency findings

No state-machine violations remain. Every combination the brief lists was traced
against the code:

| state | result |
|---|---|
| completed + cancel_requested | **fixed in Phase 8**; a cancel during encoding now routes to `_finish_cancelled` |
| failed + output file | impossible: `_finish_failed` deletes the output and sets `output_path=None` |
| cancelled + result file | impossible: `_finish_cancelled` deletes it |
| running without DB row | impossible: the row is committed before the id reaches the queue |
| queued without upload | impossible: a queue refusal deletes both the row and the upload |
| DB row without files | expected and handled; the row carries the error |
| files without DB row | swept by `sweep_orphans` |

**Cancellation while queued** stops before any decode or model load. **Duplicate
cancellation** is a set insert, harmless. **Worker failure** is caught and the
job reaches a terminal state, with the consumer surviving. **Shutdown during a
job** asks for cancellation and waits, so the current tile finishes and files
close before exit. **Restart** marks queued and processing jobs failed with a
message telling the user to resubmit.

**Multiple SSE subscribers** each get their own bounded queue; a slow one has its
oldest event dropped rather than blocking the publisher, which is what stops a
stalled browser from stalling inference.

## 7. Storage findings

No orphan class found that the sweep does not cover. Paths are derived only from
generated hex ids, `delete()` refuses anything resolving outside the storage
tree and logs only the basename, and the uploaded filename never reaches the
filesystem.

**Crash at each stage** leaves at most one input file and possibly one partial
output, both of which `sweep_orphans` removes because no row references them.
Inputs survive a failed or cancelled job on purpose — the comparison view needs
them — and expire on the 24-hour retention window.

## 8. Database findings

| aspect | classification |
|---|---|
| Indexes on `created_at`, `status`, `expires_at` | **B — appropriate**; they match the history sort, the recovery query and the sweep |
| Pagination with limit and offset | **B — appropriate** at this size |
| Row committed before the id is queued | **B — correct**; prevents a job that is accepted and never runs |
| Single writer under `aiosqlite` | **C — production scaling limitation** |
| Offset pagination | **D — future concern**; degrades on deep pages, irrelevant at current volume |

No transaction bug found. The one ordering hazard — announcing a job before it
is durable — is explicitly handled and commented.

## 9. Security findings

### S1 — Tracebacks are returned to the client outside production

**Severity: MEDIUM. MUST FIX BEFORE PUBLIC** (as configuration, not code).

**Where.** [`errors.py:68`](backend/app/api/errors.py#L68). `include_trace` is
`environment != "production"`, and the shipped default is `development`.

**Impact.** A 500 returns a full traceback with filesystem paths. Unreachable
through the intended tunnel, because Vite proxies only `/api` — verified in
Phase 7 by requesting `/docs` and `.env` through the public URL and getting the
SPA shell. It becomes exposure the moment anyone tunnels port 8001 directly.

**Recommended fix.** Set `ENVIRONMENT=production` for any exposed run. No code
change; adding production behaviour to the dev server would be worse.

### S2 — `/docs`, `/redoc` and `/openapi.json` are always mounted

**Severity: LOW. MUST FIX BEFORE PRODUCTION.**

**Where.** [`main.py:36-38`](backend/app/main.py#L36-L38). Not gated by
environment. Same reachability caveat as S1.

### S3 — No authentication and no rate limiting

**Severity: HIGH for production, ACCEPTABLE CURRENTLY.**

Anyone with the URL can use the GPU and read every job in the shared history.
The queue bounds GPU monopolisation to one job plus eight waiting, but nothing
bounds the rate of rejected submissions. Correct for a local tool; a blocker for
unattended deployment.

### S4 — SSE connections are not counted

**Severity: LOW. CAN DEFER.**

Each subscription is a bounded queue on a topic, cleaned up in a `finally`.
Phase 8 proved twenty open-and-abandon cycles leave no subscribers behind. The
bound is whatever the ASGI server allows, not anything the application enforces.

### Verified closed

Path traversal, magic-byte validation, decompression bombs (16 MP checked from
the header before decode, twelve times stricter than Pillow's own guard and
applied earlier), filename sanitisation, storage-escape on delete, and error
bodies that carry no filesystem paths.

## 10. Observability findings

### O1 — A successful job logs no completion line

**Severity: LOW. CAN DEFER.**

The lifecycle is well covered for failure — queued, cancelled, deleted, handler
raised, failed unexpectedly, progress write failed — and each carries `job_id`.
Per-pass detail records model, tile, device and output size. What is missing is
a single line when a job finishes successfully, with total elapsed time.

**Impact.** Diagnosing "my upscale was slow" means querying SQLite rather than
reading the log. `processing_ms` is stored, so nothing is lost, but the log
alone cannot answer it.

**What would be needed to diagnose a real user's failed upscale today:** the
`job_id`, which the UI does show. From that the log gives the failure and the
stage, and the row gives the settings. That is adequate. Absent are queue wait
time, VRAM at failure, and any aggregation.

## 11. Deployment findings

Two components do not survive the architecture the brief describes.

**The job queue is in-process.** `InProcessJobQueue` holds an `asyncio.Queue`
and a thread pool inside the API process. A second API instance would have its
own queue and its own GPU assumption. **Must become an external queue** before
more than one instance runs.

**The progress broker is in-memory.** `ProgressBroker._topics` is a dict. A
browser connected to instance A receives nothing for a job running on instance
B. **SSE does not survive horizontal scaling** without a shared bus, or without
pinning a job's stream to the instance running it.

**What can stay as it is:** the inference path, the tiling and OOM ladder, the
resolution planner, the cascade, image validation and encoding. None of them
assume co-location.

**What must change:** SQLite to PostgreSQL; local filesystem to object storage,
which also removes the assumption that the API process can read the file a
worker wrote; queue and broker as above.

**Where GPU inference should live:** a separate worker. The current split is
already close — the runner talks to the queue through an interface, and
`JobQueue` is documented as an interface "so that swapping to Celery later means
writing one implementation rather than touching the services above it". That
foresight holds up.

**When the GPU worker dies:** today the recovery service marks interrupted jobs
failed at startup. With an external queue that becomes a visibility-timeout
redelivery, which is a different and better behaviour, but it is not what the
code does now.

## 12. Capacity and scaling findings

Qualitative, from measured single-job numbers: roughly 21–31 s for a 2 MP image
at 4x, 767 MB peak VRAM at tile 256, one job at a time.

| load | behaviour | first bottleneck |
|---|---|---|
| 1 user | comfortable | none |
| 5 concurrent | one runs, four queue; last waits ~2 min | **GPU**, by design |
| 20 concurrent | 8 queue, the rest are refused with `queue_full` | **GPU**, then the queue bound |
| 100 concurrent | ~91 refused immediately | **GPU**; SQLite and SSE fan-out follow |

**The first bottleneck is the GPU and it is not close.** One 4 GB card serving
one job at a time is the constraint everything else sits behind. SQLite's single
writer, the SSE fan-out and disk throughput are all comfortably ahead of it at
any load this hardware can accept. That ordering changes the moment inference
moves to a scalable worker pool, at which point the database becomes next.

## 13. Test coverage assessment

**Behaviour actually tested:** cancellation at every stage including the
encoding window, queue bounds, upload rejection and the absence of leftover
files, path-shaped ids, typed error codes, denoise resolution end to end
including persistence, i18n across both languages at component level, model
description mapping against the real manifest, seam geometry against the
production tiler, and a full real-weights job over HTTP with SSE.

**Statically inspected only:** the deployment claims in §11, worker death under
an external queue, disk exhaustion behaviour, and multi-instance SSE. All are
about architecture that does not exist yet.

**Needing real end-to-end validation:** the browser paths. Every frontend test
runs in jsdom, which has no `EventSource` — the suite exercises the polling
fallback, and the SSE path is covered only from the backend side. Mobile and
narrow-viewport behaviour is not covered by any automated test. Keyboard
navigation is partially covered through accessible-name queries.

## 14. Findings severity table

| # | Finding | Area | Severity | Classification | Test? |
|---|---|---|---|---|---|
| P1 | Creative shows wrong model, disables working denoise control | UX / bug | **HIGH** | MUST FIX BEFORE PUBLIC | yes |
| S1 | Tracebacks returned outside production | Security | MEDIUM | MUST FIX BEFORE PUBLIC (config) | no |
| R1 | Storage cap ignores the output it is about to create | Resource | MEDIUM | MUST FIX BEFORE PRODUCTION | yes |
| S3 | No authentication, no rate limiting | Security | HIGH (prod) | MUST FIX BEFORE PRODUCTION | no |
| D1 | In-process queue cannot scale past one instance | Architecture | HIGH (prod) | MUST FIX BEFORE PRODUCTION | no |
| D2 | In-memory SSE broker cannot scale past one instance | Architecture | HIGH (prod) | MUST FIX BEFORE PRODUCTION | no |
| D3 | SQLite single writer | Database | MEDIUM (prod) | MUST FIX BEFORE PRODUCTION | no |
| S2 | `/docs` and `/openapi.json` always mounted | Security | LOW | MUST FIX BEFORE PRODUCTION | no |
| O1 | No completion log line | Observability | LOW | CAN DEFER | no |
| S4 | SSE connection count unbounded | Security | LOW | CAN DEFER | no |
| T1 | SSE untested from the browser side | Testing | LOW | CAN DEFER | yes |
| T2 | No mobile or narrow-viewport coverage | Testing | LOW | CAN DEFER | yes |
| I1 | `enhance:mode.description` defined but unused | Hygiene | INFO | ACCEPTABLE | no |
| I2 | Offset pagination degrades on deep pages | Database | INFO | ACCEPTABLE | no |
| I3 | Repeated DELETE returns 404 rather than 204 | API | INFO | ACCEPTABLE | no |

**Counts:** 2 HIGH now (P1, and S3/D1/D2 which are HIGH only for production),
3 MEDIUM, 4 LOW, 3 INFO. **One actual bug: P1.**

## 15. MUST FIX BEFORE PUBLIC

1. **P1 — the Creative model and denoise mismatch.** The only code fix on this
   list. Alternatively hide Creative until it is fixed, which F1 already
   recommended.
2. **S1 — run with `ENVIRONMENT=production`.** Configuration, not code.
3. Keep the tunnel pointed at Vite, never at port 8001. Already the case.
4. Accept and state plainly that the URL is unauthenticated and every visitor
   shares one history.

## 16. MUST FIX BEFORE PRODUCTION

1. Authentication and per-IP rate limiting (S3).
2. External queue (D1) and shared or pinned progress transport (D2).
3. PostgreSQL (D3) and object storage.
4. Real storage quota (R1).
5. Disable or protect `/docs` (S2).
6. Observability: completion and queue-wait logging, error aggregation, alerts.

## 17. CAN DEFER

O1, S4, T1, T2, I1, I2, I3. None affects correctness, and none becomes urgent
before the production items above are done.

## 18. Recommended Phase 11 onward

**Phase 11 — fix P1.** Small, self-contained, and the only thing standing
between the current build and honest public testing. Make the panel resolve the
displayed model from the mode when the user has not chosen one, and add the
regression test that would have caught it.

**Phase 12 — pre-deployment hardening.** R1's real quota, S2's docs gating,
O1's completion logging, and a production configuration checklist. All small,
all independent, none architectural.

**Phase 13 — decide the deployment shape before building it.** The queue and
broker decisions in §11 constrain everything after them and are cheap to make on
paper and expensive to reverse in code. A design document, not an
implementation.

**Phase 14 — extract the GPU worker.** Only after 13. The existing `JobQueue`
interface is the seam.

**Phase 15 — PostgreSQL and object storage.** Last, because they are mechanical
once the worker is separate, and pointless before it.

Authentication is deliberately not in this sequence. It is a product decision
about who the users are, and the answer changes what to build.

---

## Reproduction of P1

Rendered rather than reasoned. `EnhancePanels` with mode `creative`, two models
on `GET /api/models`, dropdown untouched: the store settles on
`RealESRGAN_x4plus` with `modelChosen` false, the denoise slider carries
`data-disabled`, and the panel renders "Real-ESRGAN x4 Plus has no denoise
weights to blend." A submission in that state omits `model`, so
`mode_planner.plan_mode(CREATIVE)` selects `realesr-general-x4v3` and
`resolve_denoise` supplies 0.25.

The test used to confirm this was temporary and has been deleted; §18 recommends
adding a permanent one as part of the fix.
