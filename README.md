# PixelForge AI

Real image super-resolution in the browser, backed by Real-ESRGAN running on
your own hardware. Upload an image, upscale it 2x/4x/8x, compare the result
against the original, and download it.

> **Build status — Phase 2 of 14 complete.**
> This repository currently contains the project scaffold: tooling, design
> tokens, configuration, error taxonomy, logging, and a running FastAPI service
> with a health endpoint. **Image processing is not implemented yet** — it lands
> in Phase 6. Every screen in the UI states which phase implements it rather
> than showing placeholder behaviour. See [Roadmap](#roadmap).

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

Implemented today (Phase 2):

- Monorepo with strict TypeScript and strict mypy on both sides
- Dark-first design token system (Tailwind v4, OKLCH palette)
- Structured logging with per-job context fields
- Full error taxonomy mapped to RFC 9457 `application/problem+json`
- Configuration via `.env`, validated by Pydantic at startup
- `GET /api/health` with OpenAPI docs at `/docs`
- Environment diagnostic script that explains CUDA problems in plain language

Planned, with the phase that delivers each:

| Feature | Phase |
| --- | --- |
| Drag-and-drop upload, validation, image viewer | 4 |
| System/GPU status endpoint and indicator | 5 |
| Real Real-ESRGAN inference with tiling | 6 |
| Denoise strength via DNI weight interpolation | 6b |
| Async jobs, SSE progress, cancellation | 7 |
| Before/after comparison (slider, side-by-side, split) | 9 |
| Enhancement history (SQLite) | 10 |
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
| `DATABASE_URL` | SQLite file | Change to `postgresql+asyncpg://...` to migrate |

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
| Backend | http://127.0.0.1:8000 |
| Swagger UI | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

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

Weights are downloaded on demand into `models/`, verified against the SHA-256
digests in `models/manifest.json`, and never committed.

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
│       ├── components/    layout and shared UI primitives
│       ├── features/      upload, viewer, settings, processing, history, system
│       ├── pages/         routed screens
│       ├── stores/        Zustand state
│       ├── services/      typed API client
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
| 3 | Design system and app shell | Pending |
| 4 | Upload and image viewer | Pending |
| 5 | FastAPI service: system, models, persistence | Pending |
| 6 | Real Real-ESRGAN inference with tiling | Pending |
| 6b | Denoise strength via DNI | Pending |
| 7 | Async jobs, SSE progress, cancellation | Pending |
| 8 | Frontend/backend integration | Pending |
| 9 | Comparison viewer | Pending |
| 10 | History | Pending |
| 11 | Test suites | Pending |
| 12 | Docker | Pending |
| 13-14 | Performance and UX polish | Pending |

## License

Model weights are distributed by the Real-ESRGAN project under BSD-3-Clause and
are downloaded at runtime, not redistributed here. Vendored architecture code
retains its upstream Apache-2.0 notice in
`backend/app/inference/arch/ATTRIBUTION.md`.
