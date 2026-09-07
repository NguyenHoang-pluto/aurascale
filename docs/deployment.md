# Deployment

> **Status.** Container images and compose files land in Phase 12. This document
> is the deployment design they will be built to, plus the local-run procedure
> that works today.

## 1. Local development (works now)

Two processes, started together by the dev scripts:

```powershell
.\scripts\setup.ps1     # once
.\scripts\dev.ps1
```

```bash
./scripts/setup.sh      # once
./scripts/dev.sh
```

Vite serves the frontend on 5173 and proxies `/api` to uvicorn on 8000, so the
browser sees a single origin and CORS is not involved. `strictPort` is on
deliberately: a port conflict fails loudly rather than silently relocating the
server and leaving the proxy pointed at nothing.

## 2. Container topology (Phase 12)

```
                    ┌────────────────────────┐
   :5173 / :80 ───► │ frontend               │
                    │  nginx + static bundle │
                    │  /api → backend:8000   │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ backend                │
                    │  uvicorn + PyTorch     │
                    │  volumes: storage,     │
                    │           models       │
                    └────────────────────────┘
```

nginx serves the built bundle and proxies `/api`, which keeps the same-origin
model from development and means the production image needs no CORS
configuration either.

Two named volumes:

| Volume | Contents | Why it is a volume |
| --- | --- | --- |
| `models` | downloaded weights | ~200 MB; re-downloading on every deploy is wasteful |
| `storage` | inputs, outputs, thumbnails, SQLite | must survive container replacement |

### CPU profile

```bash
docker compose up --build
```

Uses `requirements-cpu.txt`. Works anywhere Docker runs, needs no GPU runtime,
and is the profile CI builds. Inference is roughly 20–60× slower than GPU.

### GPU profile

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires the NVIDIA Container Toolkit on the host and a CUDA base image
matching the pinned torch build.

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**On Windows**, GPU passthrough requires Docker Desktop with the WSL2 backend;
the Hyper-V backend cannot pass a GPU through. If GPU containers are more
friction than they are worth on a development laptop, running the backend
natively with `scripts/dev.ps1` and containerising only the frontend is a
perfectly reasonable split.

### Image build notes

- Frontend uses a multi-stage build: Node builds the bundle, nginx serves it.
  The Node layer is discarded, so the runtime image carries no build toolchain.
- Backend installs torch in its own layer, before application code is copied, so
  editing a source file does not invalidate a multi-gigabyte layer.
- `opencv-python-headless` rather than `opencv-python`: the headless build has no
  GUI dependency, which removes a large set of X11 libraries from the image.
- The container runs as a non-root user; `storage` and `models` are chowned to it.

## 3. Configuration

Containers read the same variables as local development, supplied through the
environment rather than a mounted `.env`.

Production differences from the defaults:

| Variable | Development | Production |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | `production` |
| `LOG_FORMAT` | `text` | `json` |
| `HOST` | `127.0.0.1` | `0.0.0.0` |
| `CORS_ORIGINS` | localhost:5173 | the real origin, or unset behind nginx |

`ENVIRONMENT=production` also stops tracebacks being included in error
responses; they continue to be logged in full.

## 4. Health and readiness

`GET /api/health` has no database or GPU dependency, so it reports liveness
rather than the health of everything downstream — a GPU problem should not cause
an orchestrator to restart-loop a container that is serving requests correctly.

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 20s
```

`GET /api/system` is the deeper probe: device, CUDA availability and loaded
models. Use it for dashboards, not for restart decisions.

## 5. Scaling

The backend is stateful in one respect that matters: jobs and their files live
on a local volume. Running two replicas against the same SQLite file is not
supported.

To scale beyond one machine:

1. Move `DATABASE_URL` to PostgreSQL. Only the repository binding changes —
   services depend on the `JobRepository` ABC, not on SQLAlchemy.
2. Move storage to object storage behind the same `StorageService` interface.
3. Replace the in-process `JobQueue` with a distributed implementation.

Until then, vertical scaling is the honest answer: one process, one GPU,
`MAX_CONCURRENT_JOBS=1`. A single 4 GB GPU cannot usefully run two inferences at
once, so the in-process design costs nothing at this size.

## 6. Operational limits

| Concern | Control |
| --- | --- |
| Disk growth | `TEMP_RETENTION_HOURS`, `MAX_TOTAL_STORAGE_GB`, periodic sweeper |
| Runaway upload | `MAX_UPLOAD_SIZE_MB`, enforced while streaming |
| Memory exhaustion | `MAX_INPUT_PIXELS`, `MAX_OUTPUT_PIXELS`, tiling, OOM ladder |
| Queue overload | Bounded queue; returns `queue_full` (503) rather than accepting work it cannot start |
| Restart with in-flight jobs | Startup marks stale `processing` jobs as `failed` with an explanatory message |

## 7. Backups

Back up the `storage` volume: it holds the SQLite database and every result. The
`models` volume is reproducible from `models/manifest.json` and does not need
backing up.

SQLite should be copied with `sqlite3 .backup` or with the container stopped —
copying the file while it is being written can capture a torn page.
