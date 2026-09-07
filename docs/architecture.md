# Architecture

This document records *why* the system is shaped the way it is. For the API
surface see [`api.md`](api.md); for the pixel pipeline see
[`image-processing.md`](image-processing.md).

## 1. Shape of the system

PixelForge AI is a two-process application: a static React bundle and a Python
service that owns the GPU. There is no third tier — no broker, no cache server,
no separate worker process — and that absence is deliberate (see §3).

```
                      ┌──────────────────────────────┐
                      │ Browser                      │
                      │  React 19 · Zustand · Query  │
                      └──────┬───────────────┬───────┘
              REST (JSON,    │               │  SSE
              multipart)     │               │  (progress)
                             v               v
                      ┌──────────────────────────────┐
                      │ FastAPI (single uvicorn proc)│
                      │                              │
                      │  api/        transport       │
                      │  services/   business logic  │
                      │  repositories/ persistence   │
                      │  workers/    queue + pool    │
                      └──────┬───────────────┬───────┘
                             │               │
             asyncio.Queue   │               │  SQLAlchemy (async)
                             v               v
                  ┌────────────────────┐  ┌──────────────┐
                  │ Worker thread      │  │ SQLite       │
                  │  ModelManager      │  │ job records  │
                  │  TiledRunner       │  └──────────────┘
                  │  RRDBNet (PyTorch) │
                  └─────────┬──────────┘
                            v
                    storage/{inputs,outputs,thumbs}
```

## 2. Layering rules

Dependencies point in one direction only:

```
api  ──►  services  ──►  repositories  ──►  models (ORM)
             │
             └────────►  inference  ──►  arch (vendored)
```

- `api/` knows about HTTP. It validates, delegates, and translates exceptions.
  It contains no business rules.
- `services/` knows about the product. It does not import FastAPI, and does not
  know what a request is.
- `repositories/` knows about persistence. `JobRepository` is an abstract base
  class; `SqlAlchemyJobRepository` is one implementation. Services depend on the
  ABC, which is what keeps the PostgreSQL migration a configuration change.
- `inference/` knows about tensors. It has no idea jobs or databases exist; it
  takes an array and returns an array, reporting progress through a callback.

Nothing substantial lives in `main.py` — it is an application factory only.

## 3. Decision: in-process queue, not Celery

**Options considered**

| Option | Durable jobs | Multi-node | Ops cost |
| --- | --- | --- | --- |
| Celery + Redis | yes | yes | Redis container, broker config, second process |
| RQ + Redis | yes | yes | Redis container |
| `asyncio.Queue` + thread pool | via SQLite | no | none |

**Chosen: `asyncio.Queue` + `ThreadPoolExecutor`.**

The deciding argument is that a single GPU serialises inference regardless of
the broker. `MAX_CONCURRENT_JOBS` is 1 on a one-GPU machine, so a distributed
queue would add a container, a serialisation hop for every progress event, and
a second deployment unit while delivering exactly the same throughput.

Durability is handled where it actually matters: job state lives in SQLite, so a
crash leaves a durable record. On startup the service marks any job still in
`processing` as `failed` with a clear message rather than leaving it stuck.

The escape hatch is the `JobQueue` interface. Swapping to Celery later means
writing one new implementation; services and API layers are untouched.

## 4. Decision: worker thread, not asyncio task

PyTorch inference is blocking native code that does not yield to the event loop.
Running it in a coroutine would freeze SSE streams, health checks and every
other request for the duration of a job — potentially minutes on CPU.

`loop.run_in_executor(pool, ...)` moves it to a dedicated thread. The GIL is
released inside PyTorch's native kernels, so the event loop genuinely continues
to run. A pool sized at `MAX_CONCURRENT_JOBS` also gives us GPU serialisation
for free, with no extra locking.

Progress crosses the thread boundary via
`asyncio.run_coroutine_threadsafe`, which is the supported way to publish from a
worker thread into the loop that owns the SSE subscribers.

## 5. Decision: SSE, not WebSocket

Progress is strictly one-directional: server to client. SSE gives automatic
reconnection with `Last-Event-ID`, survives proxies that mishandle WebSocket
upgrades, and needs no client library — `EventSource` is built into the browser.

Cancellation, the only client-to-server action during a job, is a plain
`DELETE /api/jobs/{id}`; it does not justify a duplex channel.

`GET /api/jobs/{id}` remains available as a polling fallback so the UI still
functions where SSE is blocked entirely.

## 6. Decision: vendored architecture, not the `realesrgan` package

The upstream `realesrgan` package depends on `basicsr`, which:

- imports `torchvision.transforms.functional_tensor`, deleted in torchvision
  0.17, so it raises `ModuleNotFoundError` on import with any current PyTorch;
- assumes NumPy < 2;
- has had no release since 2022.

The workarounds in circulation involve monkey-patching a third-party package at
import time. That is not something to build a product on.

Separately, `RealESRGANer.enhance()` is a single blocking call with no progress
hook. The requirement for honest, stage-and-tile-accurate progress means writing
our own tiling loop regardless — at which point the package provides nothing we
still need.

**What we vendor:** `RRDBNet` and `SRVGGNetCompact`, both pure `nn.Module`
definitions, from the Apache-2.0 upstream, unmodified except for formatting, in
`backend/app/inference/arch/` with an `ATTRIBUTION.md`.

**What we do not change:** the pretrained weights, which are the official
release artifacts loaded with `strict=True`. Output is bit-comparable to
upstream for the same input and tile configuration.

## 7. Model management

`ModelManager` owns every loaded model.

- **Lazy**: nothing is loaded at startup; the first job that needs a model loads
  it. Startup stays fast and a CPU-only machine never pays for CUDA init.
- **Cached**: models stay resident between jobs. Loading `RealESRGAN_x4plus`
  costs seconds; doing it per request would dominate runtime.
- **Bounded**: on a 4 GB GPU only one model stays resident. An LRU eviction
  frees the previous model and calls `torch.cuda.empty_cache()` before loading a
  new one.
- **Thread-safe**: a `threading.Lock` guards loading so two concurrent jobs
  cannot both start loading the same weights and double the VRAM spike.

`ModelManager.get(model_id)` returns an `Upscaler`:

```python
class Upscaler(Protocol):
    id: str
    scale: int
    arch: str

    def upscale(
        self,
        image: np.ndarray,
        *,
        tile: int,
        tile_pad: int,
        on_progress: Callable[[float], None],
        should_cancel: Callable[[], bool],
    ) -> np.ndarray: ...
```

Any future model — SwinIR, HAT, a diffusion upscaler — becomes a new
implementation of this protocol plus a manifest entry. No other layer changes.

## 8. Device strategy

`DEVICE=auto` resolves to CUDA when `torch.cuda.is_available()`, else CPU.
An explicit `DEVICE=cuda` on a machine without CUDA is an error at startup, not
a silent downgrade — silent downgrades are how people end up wondering why a job
takes forty minutes.

On CUDA: fp16, because it roughly halves activation memory and is faster on
every card that supports it. On CPU: fp32, because fp16 on CPU is emulated and
therefore slower.

Out-of-memory is treated as a recoverable condition, not a crash. The ladder is:
free the cache, halve the tile size, retry (up to three times), then fall back to
CPU. The job record stores what actually happened so the UI can say "reduced
tile size to fit available GPU memory" instead of showing a CUDA traceback.

## 9. Error model

Every user-triggerable failure has a class in `app/core/exceptions.py` carrying a
stable `code`, an HTTP status, a human-readable `message`, and optional
`technical` detail.

The API serialises these as RFC 9457 problem documents. The frontend maps `code`
to its own copy and renders `technical` inside a collapsed disclosure. Unhandled
exceptions produce a generic message in production; the traceback is logged, and
included in the response body only outside production.

This is the mechanism behind requirement §10: users see
"Your image is too large to process with the current GPU memory", developers
expand one row and see the `CUDA out of memory` detail.

## 10. Security posture

| Threat | Control |
| --- | --- |
| Oversized upload | Size checked while streaming, before the body is buffered |
| Decompression bomb | `MAX_INPUT_PIXELS` enforced after header parse, before decode |
| Spoofed content type | Magic-byte sniff plus `PIL.Image.verify()`; the declared MIME type and file extension are both ignored |
| Path traversal | Filenames are server-generated UUIDs; the client never supplies a path component |
| Information disclosure | Storage paths live only in the database, never in a response body |
| Metadata leakage | Metadata is preserved only on explicit request; GPS tags are dropped when stripping |
| Disk exhaustion | Retention sweeper plus `MAX_TOTAL_STORAGE_GB` |

## 11. Frontend structure

State is split by ownership:

- **Server state** — jobs, system info, history — belongs to TanStack Query,
  which handles caching, retries and invalidation.
- **Client state** — the selected file, current settings, viewer zoom and
  comparison mode, theme — belongs to Zustand stores.

Keeping these separate avoids the common failure of manually mirroring server
responses into a global store and then fighting staleness.

Feature folders (`features/upload`, `features/viewer`, ...) own their components,
hooks and local types. `components/ui` holds only primitives with no product
knowledge. A component in one feature never imports from another feature's
internals; shared behaviour moves to `hooks/` or `lib/`.

## 12. Testing strategy

| Layer | What is tested | How it stays fast |
| --- | --- | --- |
| `core` | config validation, error serialisation, log formatting | pure unit tests |
| `api` | status codes, problem-document shape, contract | httpx ASGI transport, no network |
| `services` | job lifecycle, validation rules | repository faked via the ABC |
| `inference` | tiling geometry, seam handling, OOM backoff | synthetic tensors, tiny model |
| end-to-end | a real image through a real model | marked `slow`, opt-in with `-m slow` |

The real-inference test is genuinely real — it downloads weights and produces an
upscaled file — but it is excluded from the default run so the everyday suite
stays in the sub-second range.
