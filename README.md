# PixelForge AI

Real image super-resolution in the browser, backed by Real-ESRGAN running on
your own hardware. Upload an image, upscale it 2x/4x/8x, compare the result
against the original, and download it.

> **Build status — Phase 10 of 14 complete.**
> The whole loop works in the browser: drop an image in, choose a model, scale
> and output settings, watch real per-tile progress, compare before and after
> with a slider, side by side or a fixed split, download the result, and find
> recent jobs again on the History screen. Real-ESRGAN runs on your GPU (or
> CPU) with a denoise control backed by DNI weight interpolation. **History is
> not an archive** — jobs and their images are removed after
> `TEMP_RETENTION_HOURS`, 24 by default. Every unbuilt region states which
> phase delivers it. See [Roadmap](#roadmap).

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [Requirements](#requirements)
6. [Installation — CPU](#installation--cpu)
7. [Installation — NVIDIA GPU](#installation--nvidia-gpu)
8. [Environment variables](#environment-variables)
9. [Running locally](#running-locally)
10. [Docker](#docker)
11. [API documentation](#api-documentation)
12. [Model information](#model-information)
13. [Performance considerations](#performance-considerations)
14. [Troubleshooting](#troubleshooting)
15. [Project structure](#project-structure)
16. [Roadmap](#roadmap)

---

## Overview

PixelForge AI is a desktop-first web application for upscaling and restoring
images. It is not a wrapper around a hosted API: inference runs locally in a
PyTorch process you control, using the official Real-ESRGAN weights.

The design goal is a focused enhancement workspace — upload, compare, enhance,
download — rather than a dashboard. The image is the interface.

## Features

Implemented today (Phases 1-10):

- Monorepo with strict TypeScript and strict mypy on both sides
- Dark-first design token system (Tailwind v4, OKLCH palette)
- Accessible component library: buttons, panels, segmented controls, select,
  slider, switch, tooltips, status indicators, progress, error disclosure
- Application shell with responsive navigation and the workspace layout
- Drag-and-drop, click and paste upload, with format detected from file
  contents rather than the extension, and a real decode to reject corrupt files
- Image viewer: 25/50/100/200/400% zoom, fit to screen, drag pan,
  ctrl+wheel zoom, full keyboard control, alpha checkerboard
- Image information panel showing dimensions, resolution, size and format
- Typed API client that turns every failure — HTTP, network, timeout, abort —
  into one `ApiError` shape
- Live backend status indicator polling `/api/health`
- Structured logging with per-job context fields
- Full error taxonomy mapped to RFC 9457 `application/problem+json`
- Configuration via `.env`, validated by Pydantic at startup
- `GET /api/health`, `GET /api/system` and `GET /api/models`, with OpenAPI
  docs at `/docs`
- Device resolution that reports *why* it chose CUDA or CPU, and a Settings
  panel plus top-bar indicator showing the measured GPU, VRAM, CUDA, PyTorch
  and CPU rather than a guess
- Model registry read from `models/manifest.json`, annotated with which weights
  are on disk
- SQLAlchemy 2 job records in SQLite, versioned by Alembic and migrated on
  startup, with interrupted jobs recovered when the process restarts
- Real Real-ESRGAN inference against the official weights: `RealESRGAN_x4plus`,
  `RealESRGAN_x2plus`, `RealESRGAN_x4plus_anime_6B` and `realesr-general-x4v3`,
  loaded strictly into vendored `RRDBNet` / `SRVGGNetCompact` architectures
- VRAM-safe tiled inference with real per-tile progress, cancellation between
  tiles, and an out-of-memory ladder that halves the tile before falling back
  to the CPU rather than failing the job
- Denoise strength by DNI weight interpolation between `realesr-general-x4v3`
  and its `wdn` counterpart — a real blended network, not a blend of outputs
- 2x, 4x and 8x, where 8x is two neural passes (4x then 2x) so no part of the
  result is resampled rather than generated
- Model manager that loads lazily, caches between jobs, keeps one model resident
  on a GPU, and frees VRAM before loading the next
- Weight downloader that verifies SHA-256 before a file becomes visible
- Job API: submit an image, follow it over Server-Sent Events, download the
  result with resumable `Range` support, and cancel or delete it
- An in-process queue and worker thread, so inference never blocks the event
  loop and a job survives as a durable record rather than as memory
- Server-side repeat of every upload check — streamed size cap, magic bytes,
  structural verify, dimension and output-pixel guards — because the browser's
  checks are a convenience, not a control
- Cooperative cancellation that stops between tiles and keeps no partial result
- Retention sweeper that removes expired jobs, their files, and any orphans
- Enhancement controls in the browser: model, upscale factor, noise reduction
  and sharpening, with output format, quality and metadata handling
- Impossible combinations are unselectable rather than rejected: `/api/models`
  publishes the factors each model can actually produce, derived from the same
  planner that validates a job, and the UI greys out the rest
- Live job progress from the event stream, with polling as the fallback, and a
  download that streams from the server rather than through JavaScript
- Before/after comparison in three modes — a draggable slider, side by side,
  and a fixed split — all driven by one zoom/pan transform so the two images
  can never drift out of alignment
- A resolution-capped preview keeps large results out of the DOM, and above
  100% zoom the viewer fetches a full-resolution crop of just the visible
  region
- History of recent jobs with thumbnails, the model, scale, sizes, processing
  time and outcome, filterable by status and paged, with view, download and
  delete — and the retention window stated on the page rather than left to be
  discovered when an entry disappears
- Environment diagnostic script that explains CUDA problems in plain language

Planned, with the phase that delivers each:

| Feature | Phase |
| --- | --- |
| Docker images, CPU and GPU profiles | 12 |

## Architecture

```
Browser (React + Zustand)
   |  REST + Server-Sent Events
   v
FastAPI  -->  JobService  -->  in-process queue  -->  worker thread
   |                                                       |
   |                                                       v
   |                                             ModelManager (cached)
   |                                                       |
   v                                                       v
SQLite (job records)                              Real-ESRGAN (PyTorch)
                                                           |
                                                           v
                                                   storage/outputs
```

Inference runs on a dedicated worker thread, never on the event loop, so
progress streaming and health checks stay responsive during a long job.
Full detail: [`docs/architecture.md`](docs/architecture.md).

## Tech stack

**Frontend** — React 19, TypeScript (strict), Vite 8, Tailwind CSS v4,
Zustand, TanStack Query, lucide-react, Vitest + Testing Library, oxlint.

**Backend** — Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic,
PyTorch, Pillow, OpenCV (headless), pytest, ruff, mypy.

**Why no `basicsr` / `realesrgan` package** — both are unmaintained, import a
`torchvision` module removed in 0.17, and assume NumPy < 2. We vendor the model
*architecture* (`RRDBNet`, `SRVGGNetCompact`) from the Apache-2.0 upstream and
load the official pretrained weights unmodified, so output quality is identical
while the dependency tree stays healthy. It also lets the tiling loop report
genuine per-tile progress, which `RealESRGANer.enhance()` cannot.

## Requirements

- **Python** 3.11 or newer
- **Node.js** 20 or newer (developed against 24)
- **NVIDIA GPU** optional. CUDA-capable card with 4 GB VRAM or more is
  recommended; the app runs on CPU otherwise, roughly 20-60x slower.
- **Disk** ~4 GB for PyTorch plus ~200 MB for model weights

## Installation — CPU

```bash
git clone <repository-url> aurascale
cd aurascale
```

Windows:

```powershell
.\scripts\setup.ps1 -Cpu
```

Linux / macOS:

```bash
./scripts/setup.sh --cpu
```

## Installation — NVIDIA GPU

Windows:

```powershell
.\scripts\setup.ps1
```

Linux / macOS:

```bash
./scripts/setup.sh
```

### CUDA requirements — read this if the GPU is not detected

The project pins **`torch==2.7.1+cu118`**, not the current default CUDA 12
build. This is deliberate:

| torch build | Minimum NVIDIA driver |
| --- | --- |
| `cu118` (what we pin) | 452.39 (Windows) / 450.80 (Linux) |
| `cu124` / `cu126` | 527.41 (Windows) / 525.60 (Linux) |

CUDA 11 minor-version compatibility means the cu118 wheels work on a much wider
range of drivers, including the older drivers still shipping on many RTX
30-series laptops. If your driver is 527 or newer and you want the newer
runtime, change `backend/requirements-cuda.txt` to a `cu124` build — nothing
else needs to change.

Verify your setup at any time:

```bash
backend/.venv/Scripts/python scripts/check_env.py   # Windows
backend/.venv/bin/python scripts/check_env.py       # Linux / macOS
```

It prints your GPU, driver, torch build, CUDA availability and free VRAM, and
explains the cause when CUDA is unavailable.

## Environment variables

Copy `.env.example` to `.env`. Every value has a working default; the file
documents each one inline. The most important:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `DEFAULT_MODEL` | `RealESRGAN_x4plus` | Model selected on first load |
| `DEFAULT_SCALE` | `4` | Upscale factor |
| `TILE_SIZE` | `256` | Tile edge in pixels; `0` disables tiling |
| `TILE_PAD` | `16` | Overlap that hides tile seams |
| `USE_FP16` | `true` | Half precision on CUDA; ignored on CPU |
| `MAX_UPLOAD_SIZE_MB` | `32` | Rejected before the body is buffered |
| `MAX_INPUT_PIXELS` | `16000000` | Decompression-bomb guard |
| `MAX_CONCURRENT_JOBS` | `1` | Keep at 1 for a single GPU |
| `DATABASE_URL` | SQLite in `STORAGE_DIR` | Change to `postgresql+asyncpg://...` to migrate |
| `AUTO_MIGRATE` | `true` | Run Alembic on startup; set `false` and migrate as a deploy step in production |

Secrets are never committed: `.env` is git-ignored and `.env.example` holds no
credentials.

## Running locally

Windows — opens backend and frontend in separate windows:

```powershell
.\scripts\dev.ps1
```

Linux / macOS — both in the foreground, Ctrl-C stops both:

```bash
./scripts/dev.sh
```

| Service | URL |
| --- | --- |
| Frontend | http://localhost:5173 |
| Component gallery (dev only) | http://localhost:5173/design |
| Backend | http://127.0.0.1:8000 |
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

The component gallery renders every design-system primitive for visual review.
The route is registered behind `import.meta.env.DEV`, so it folds away and the
page is tree-shaken out of a production build.

### Checks

```bash
cd frontend
npm run typecheck
npm run lint
npm test
npm run build
```

```bash
cd backend
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy app
.venv/Scripts/python -m pytest
```

### Database migrations

The schema is versioned with Alembic and applied on startup while
`AUTO_MIGRATE` is true, so a fresh clone needs no migration step. To apply or
author migrations by hand:

```bash
cd backend
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m alembic revision --autogenerate -m "describe the change"
```

Autogenerate compares `app/models/db.py` against the live database, so review
the generated script before committing it — SQLite reports a narrower set of
changes than PostgreSQL does.

## Docker

Container support arrives in Phase 12, with a CPU `docker-compose.yml` and a
`docker-compose.gpu.yml` overlay using the NVIDIA Container Toolkit. On Windows,
GPU passthrough requires running Docker under WSL2.

## API documentation

Live, generated from the code, at `/docs`. The full designed surface —
including the endpoints not yet built — is specified in
[`docs/api.md`](docs/api.md).

Errors use RFC 9457 problem documents:

```json
{
  "type": "https://pixelforge.ai/errors/image_too_large",
  "title": "Image too large",
  "status": 413,
  "code": "image_too_large",
  "detail": "Your image is 81 MP, larger than the 16 MP limit.",
  "technical": "input=9000x9000 limit=16000000"
}
```

`detail` is written to be shown to a user verbatim; `technical` populates the
collapsible developer panel in the UI.

## Model information

| Model | Scale | Architecture | Use |
| --- | --- | --- | --- |
| `RealESRGAN_x4plus` | 4x | RRDBNet (23 blocks) | Default; photographs |
| `RealESRGAN_x4plus_anime_6B` | 4x | RRDBNet (6 blocks) | Illustration, anime |
| `RealESRGAN_x2plus` | 2x | RRDBNet (23 blocks) | 2x preset; second pass of 8x |
| `realesr-general-x4v3` | 4x | SRVGGNetCompact | Compact; supports denoise control |
| `realesr-general-wdn-x4v3` | 4x | SRVGGNetCompact | Denoise weights, blended via DNI |

**There is no official 8x Real-ESRGAN weight.** The 8x preset is a genuine
two-pass pipeline (4x then 2x, both neural) rather than a 4x pass followed by
bicubic resampling. The UI labels it as such.

Weights are downloaded into `models/`, verified against the SHA-256 digests in
`models/manifest.json`, and never committed:

```bash
backend/.venv/Scripts/python scripts/download_models.py --all   # Windows
backend/.venv/bin/python scripts/download_models.py --all       # Linux / macOS
```

Naming one model fetches what that model needs, including the `wdn` half of the
denoise pair. A checkpoint is moved into place only after its digest matches, so
a failed download never leaves a file that looks usable.

## Performance considerations

- **Tiling is mandatory below roughly 8 GB VRAM.** A 4x pass on 1280x720
  produces 5120x2880; the untiled activations exceed 4 GB. `TILE_SIZE=256` is
  the tested default for a 4 GB card.
- **Models are loaded once** and cached in the `ModelManager`, not per request.
- **fp16 on CUDA, fp32 on CPU** — half precision is slower on CPU, not faster.
- **Out-of-memory is recovered, not surfaced as a crash**: the tile size is
  halved and retried, then falls back to CPU, and the job records what happened.
- **Large results are never sent to the browser whole.** The comparison viewer
  loads a capped-resolution preview and fetches full-resolution crops only for
  the visible region above 100% zoom; the full file is fetched on download.

## Troubleshooting

**`CUDA available: False` but I have an NVIDIA GPU.**
Run `scripts/check_env.py`. The usual cause is a CPU-only torch build (reinstall
with `requirements-cuda.txt`) or a driver older than the torch build requires —
see [CUDA requirements](#cuda-requirements--read-this-if-the-gpu-is-not-detected).

**Backend starts but the frontend shows network errors.**
Confirm the backend is on port 8000 and that `CORS_ORIGINS` includes
`http://localhost:5173`. The Vite dev server proxies `/api`, so a same-origin
setup needs no CORS entry at all.

**`npm run build` fails on a fresh clone.**
Delete `node_modules` and `package-lock.json`, then run `npm install`. The
project requires Node 20 or newer.

**Port 5173 or 8000 already in use.**
The dev server uses `strictPort`, so it fails loudly rather than silently moving
to another port. Stop the other process, or change `PORT` in `.env`.

## Project structure

```
aurascale/
├── frontend/          React + TypeScript workspace UI
│   └── src/
│       ├── components/ui/       design-system primitives
│       ├── components/layout/   app shell, top nav, workspace layout
│       ├── components/feedback/ error, empty and phase states
│       ├── features/      upload, viewer, settings, processing, history, system
│       ├── pages/         routed screens
│       ├── hooks/         shared behaviour (media queries, ...)
│       ├── stores/        Zustand client state
│       ├── services/      typed API client
│       ├── test/          shared test harness
│       └── styles/        design tokens
├── backend/           FastAPI service
│   ├── app/
│   │   ├── api/           routers and error translation
│   │   ├── core/          config, logging, exceptions, lifespan
│   │   ├── inference/     model manager, tiling, vendored architectures
│   │   ├── repositories/  data access behind an abstract interface
│   │   ├── services/      business logic
│   │   └── workers/       job queue and worker pool
│   └── tests/
├── models/            weight manifest; downloaded weights (git-ignored)
├── docker/            container definitions (Phase 12)
├── docs/              architecture, API, image processing, deployment
├── scripts/           setup, dev servers, environment diagnostics
└── storage/           runtime inputs/outputs/thumbnails (git-ignored)
```

## Roadmap

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | Architecture and analysis | Done |
| 2 | Repository scaffold and tooling | Done |
| 3 | Design system and app shell | Done |
| 4 | Upload and image viewer | Done |
| 5 | FastAPI service: system, models, persistence | Done |
| 6 | Real Real-ESRGAN inference with tiling | Done |
| 6b | Denoise strength via DNI | Done |
| 7 | Async jobs, SSE progress, cancellation | Done |
| 8 | Frontend/backend integration | Done |
| 9 | Comparison viewer | Done |
| 10 | History | Done |
| 11 | Test suites | Pending |
| 12 | Docker | Pending |
| 13-14 | Performance and UX polish | Pending |

## License

Model weights are distributed by the Real-ESRGAN project under BSD-3-Clause and
are downloaded at runtime, not redistributed here. Vendored architecture code
retains its upstream Apache-2.0 notice in
`backend/app/inference/arch/ATTRIBUTION.md`.
