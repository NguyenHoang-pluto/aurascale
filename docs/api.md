# API reference

Base path: `/api`. Interactive docs at `/docs`, schema at `/openapi.json`.

Responses use camelCase; request bodies accept camelCase. All errors are RFC
9457 problem documents with media type `application/problem+json`.

> **Implementation status.** Only `GET /api/health` exists today (Phase 2).
> Every other endpoint below is the agreed contract, implemented in the phase
> noted against it. This document is the specification the implementation is
> written against, not a description of what already runs.

---

## Conventions

### Error response

Every non-2xx response has this shape:

```json
{
  "type": "https://pixelforge.ai/errors/image_too_large",
  "title": "Image too large",
  "status": 413,
  "code": "image_too_large",
  "detail": "Your image is 81 MP, which is larger than the 16 MP limit.",
  "technical": "input=9000x9000 pixels=81000000 limit=16000000",
  "context": { "limitPixels": 16000000, "actualPixels": 81000000 }
}
```

| Field | Purpose |
| --- | --- |
| `code` | Stable enum; the frontend switches on this, never on `detail` |
| `detail` | Written for a human; safe to display verbatim |
| `technical` | Developer detail for the collapsible panel; may be absent |
| `context` | Structured values for building richer messages; may be absent |

### Error codes

| Code | Status | Meaning |
| --- | --- | --- |
| `unsupported_format` | 415 | Not a JPEG, PNG or WEBP by content inspection |
| `corrupted_image` | 422 | Declared an image but could not be decoded |
| `file_too_large` | 413 | Exceeds `MAX_UPLOAD_SIZE_MB` |
| `image_too_large` | 413 | Exceeds `MAX_INPUT_PIXELS` |
| `image_too_small` | 422 | Below `MIN_INPUT_DIMENSION` on either axis |
| `output_too_large` | 413 | Requested scale would exceed `MAX_OUTPUT_PIXELS` |
| `invalid_parameters` | 422 | Unknown model, unsupported scale, bad quality value |
| `out_of_memory` | 507 | Exhausted GPU and CPU memory after the retry ladder |
| `gpu_unavailable` | 503 | `DEVICE=cuda` was requested but CUDA is not usable |
| `storage_full` | 507 | `MAX_TOTAL_STORAGE_GB` reached |
| `model_not_found` | 404 | Model id not in the manifest |
| `model_load_failed` | 500 | Weights present but failed to load |
| `model_download_failed` | 502 | Download failed or digest mismatch |
| `inference_failed` | 500 | Model ran but raised |
| `job_not_found` | 404 | Unknown job id |
| `job_not_completed` | 409 | Result requested before the job finished |
| `queue_full` | 503 | Queue at capacity |
| `timeout` | 504 | Exceeded the processing time limit |
| `internal_error` | 500 | Unclassified |

---

## System

### `GET /api/health`

Liveness. Deliberately free of database and GPU dependencies, so it answers even
when those are degraded — this is what a container healthcheck probes.

**Status: implemented (Phase 2).**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "uptimeSeconds": 128.4
}
```

### `GET /api/system`

Capability and hardware report. Drives the GPU indicator and the System section
of Settings. *Phase 5.*

```json
{
  "device": "cuda",
  "cudaAvailable": true,
  "cudaVersion": "11.8",
  "torchVersion": "2.7.1+cu118",
  "gpuName": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "vramTotalMb": 4096,
  "vramFreeMb": 3338,
  "driverVersion": "512.74",
  "cpu": "AMD Ryzen 7 5800H",
  "cpuCores": 8,
  "ramTotalMb": 16384,
  "loadedModels": ["RealESRGAN_x4plus"],
  "fp16": true,
  "tileSize": 256
}
```

`vramFreeMb` is sampled at request time. When CUDA is unavailable the GPU fields
are `null` and `device` is `"cpu"` — this is a normal, supported state, not an
error.

### `GET /api/models`

Available models from `models/manifest.json`, annotated with whether the weights
are present on disk. *Phase 5.*

```json
[
  {
    "id": "RealESRGAN_x4plus",
    "name": "Real-ESRGAN x4 Plus",
    "description": "General-purpose 4x upscaler. Best default for photographs.",
    "arch": "RRDBNet",
    "scale": 4,
    "supportsDenoise": false,
    "downloaded": true,
    "sizeMb": 63.9,
    "loaded": true
  }
]
```

---

## Jobs

### `POST /api/jobs`

Create an enhancement job. Returns immediately; inference happens on a worker.
*Phase 7.*

**Request** — `multipart/form-data`

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `image` | file | yes | JPEG, PNG or WEBP, validated by content |
| `model` | string | no | Defaults to `DEFAULT_MODEL` |
| `scale` | int | no | `2`, `4` or `8`; defaults to `DEFAULT_SCALE` |
| `format` | string | no | `png`, `jpeg`, `webp`; defaults to the input format |
| `quality` | int | no | 50–100 for JPEG/WEBP; ignored for PNG |
| `preserveMetadata` | bool | no | Defaults to `true` |
| `settings` | JSON string | no | See below |

`settings` object:

```json
{
  "sharpenStrength": 0.0,
  "denoiseStrength": 0.5,
  "tileSize": null,
  "tilePad": null
}
```

`sharpenStrength` (0–1) drives an unsharp-mask post-process and is labelled as
a post-process in the UI, not as an AI feature. `denoiseStrength` (0–1) applies
only to models with `supportsDenoise`; it is the DNI interpolation coefficient
between the standard and denoise weight sets. `tileSize` and `tilePad` override
the configured defaults; `null` means "decide automatically from free VRAM".

**Response** — `202 Accepted`

```json
{
  "jobId": "9f1c0f2a4b7d4c1e8a3b6d5e2f0a1c9b",
  "status": "queued",
  "queuePosition": 0,
  "createdAt": "2026-01-15T10:32:04Z"
}
```

**Errors** — `unsupported_format`, `corrupted_image`, `file_too_large`,
`image_too_large`, `image_too_small`, `output_too_large`, `invalid_parameters`,
`queue_full`, `storage_full`.

### `GET /api/jobs/{jobId}`

Full job record. Also the polling fallback when SSE is unavailable. *Phase 7.*

```json
{
  "jobId": "9f1c0f2a4b7d4c1e8a3b6d5e2f0a1c9b",
  "status": "processing",
  "stage": "running_inference",
  "progress": 50,
  "model": "RealESRGAN_x4plus",
  "scale": 4,
  "device": "cuda",
  "input": { "width": 1280, "height": 720, "sizeBytes": 1887437, "format": "JPEG" },
  "output": null,
  "processingMs": null,
  "error": null,
  "createdAt": "2026-01-15T10:32:04Z",
  "startedAt": "2026-01-15T10:32:04Z",
  "finishedAt": null
}
```

`status` is one of `queued`, `processing`, `completed`, `failed`, `cancelled`.

`stage` is one of `validating`, `loading_model`, `preprocessing`,
`running_inference`, `postprocessing`, `encoding`, and is `null` outside
`processing`.

On completion, `output` is populated and `processingMs` is the measured
inference wall time, excluding queue wait:

```json
{
  "output": { "width": 5120, "height": 2880, "sizeBytes": 9122611, "format": "PNG" },
  "processingMs": 8420
}
```

On failure, `error` carries the same fields as a problem document:

```json
{
  "error": {
    "code": "out_of_memory",
    "detail": "Your image is too large to process with the available GPU memory.",
    "technical": "CUDA out of memory. Tried to allocate 1.24 GiB..."
  }
}
```

### `GET /api/jobs/{jobId}/events`

Server-Sent Events progress stream. *Phase 7.*

```
Content-Type: text/event-stream
Cache-Control: no-cache, no-transform
```

Event types:

```
event: progress
data: {"progress":50,"stage":"running_inference","tilesDone":30,"tilesTotal":60}

event: stage
data: {"stage":"postprocessing","progress":75}

event: completed
data: {"jobId":"9f1c...","processingMs":8420}

event: failed
data: {"code":"out_of_memory","detail":"..."}

event: cancelled
data: {"jobId":"9f1c..."}
```

The stream closes after a terminal event. A comment heartbeat (`: ping`) is sent
every 15 seconds so idle connections are not reaped by intermediaries.

Progress is measured, not simulated. During `running_inference` it is
`tilesDone / tilesTotal` scaled into that stage's share of the bar. Stages that
cannot be subdivided report their own boundaries only, rather than inventing
intermediate percentages.

### `GET /api/jobs/{jobId}/result`

The processed image.

```
Content-Type: image/png
Content-Disposition: attachment; filename="pixelforge-9f1c0f2a-5120x2880.png"
```

Returns `job_not_completed` (409) if the job has not finished, `job_not_found`
(404) if the id is unknown or the file has been swept. Supports `Range` requests
so a large download can resume. *Phase 7.*

### `GET /api/jobs/{jobId}/preview`

Resolution-capped JPEG of the result for the comparison viewer, so a 200 MP
output is never loaded into the DOM. Long edge capped at 4096 px, quality 92.
*Phase 9.*

Optional query parameters for high-zoom inspection: `x`, `y`, `w`, `h` request a
full-resolution crop of the source region instead of a downscaled whole.

### `GET /api/jobs/{jobId}/thumbnail`

256 px thumbnail for the history grid. *Phase 10.*

### `DELETE /api/jobs/{jobId}`

Cancel if running, delete otherwise. *Phase 7.*

Cancellation is cooperative: a flag is set and the tiling loop checks it between
tiles, so cancellation takes effect within one tile rather than immediately.
A job that has already completed is deleted along with its files.

Returns `204 No Content`.

### `GET /api/jobs`

Job history, newest first. *Phase 10.*

| Parameter | Default | Notes |
| --- | --- | --- |
| `limit` | 20 | Max 100 |
| `offset` | 0 | |
| `status` | — | Filter by job status |

```json
{
  "items": [ { "jobId": "...", "status": "completed" } ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```
