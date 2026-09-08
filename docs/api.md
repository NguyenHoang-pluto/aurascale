# API reference

Base path: `/api`. Interactive docs at `/docs`, schema at `/openapi.json`.

Responses use camelCase; request bodies accept camelCase. All errors are RFC
9457 problem documents with media type `application/problem+json`.

> **Implementation status.** The system endpoints (Phases 2 and 5) and the job
> endpoints — submit, record, events, result, cancel (Phase 7) — exist today.
> Still to come: `/preview` (Phase 9), `/thumbnail` and `GET /api/jobs`
> (Phase 10). Each of those is the agreed contract, implemented in the phase
> noted against it.

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
of Settings.

**Status: implemented (Phase 5).**

```json
{
  "device": "cuda",
  "deviceReason": "CUDA device detected",
  "torch": {
    "available": true,
    "version": "2.7.1+cu118",
    "cudaVersion": "11.8",
    "cudaAvailable": true,
    "importError": null
  },
  "gpu": {
    "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
    "vramTotalMb": 4095,
    "vramFreeMb": 3333,
    "capability": "8.6"
  },
  "cpuName": "AMD Ryzen 5 5625U with Radeon Graphics",
  "cpuCoresPhysical": 6,
  "cpuCoresLogical": 12,
  "ramTotalMb": 7532,
  "ramAvailableMb": 441,
  "pythonVersion": "3.11.9",
  "platform": "Windows 10",
  "fp16": true,
  "tileSize": 256,
  "tilePad": 16
}
```

`vramFreeMb` is sampled at request time — it is the number that decides whether
the next job needs a smaller tile size. `gpu` is `null` and `device` is `"cpu"`
when CUDA is unavailable: a normal, supported state, not an error, and
`deviceReason` says which of the possible causes applied.

`torch.available` is `false` on an install where PyTorch is missing or broken;
`torch.importError` then carries the import failure verbatim, so a
`DEVICE=cuda` machine that silently fell back to CPU can be diagnosed from this
one response.

The endpoint never raises for missing hardware: a machine with no GPU, no CUDA
build of torch, or no torch at all still gets a complete report.

### `GET /api/models`

Available models from `models/manifest.json`, annotated with whether the weights
are present on disk and which upscale factors each one can reach.

**Status: implemented (Phase 5).**

```json
[
  {
    "id": "RealESRGAN_x4plus",
    "name": "Real-ESRGAN x4 Plus",
    "description": "General-purpose 4x upscaler. Best default for photographs.",
    "arch": "RRDBNet",
    "scale": 4,
    "supportsDenoise": false,
    "supportedScales": [4, 8],
    "downloaded": true,
    "sizeMb": 63.9
  }
]
```

`downloaded` reports whether the weight file named in the manifest is present
under `MODELS_DIR`; `sizeMb` is its on-disk size, and `null` until it is there.

`supportedScales` is derived from the same pass planner that validates a job,
so it and `POST /api/jobs` can never disagree: a factor absent from the list is
refused with `invalid_parameters`, and a client should not offer it. A 4x model
lists `[4, 8]` because 8x is reached by a second 2x pass; a 2x model lists `[2]`.

Weight sets that are not selectable on their own are omitted — the `wdn`
counterpart used for DNI denoise interpolation is part of
`realesr-general-x4v3`, not a model a user picks.

---

## Jobs

### `POST /api/jobs`

Create an enhancement job. Returns immediately; inference happens on a worker.

**Status: implemented (Phase 7).**

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
a post-process in the UI, not as an AI feature. It is omitted when zero rather
than sent as 0. `denoiseStrength` (0–1) applies
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

Full job record. Also the polling fallback when SSE is unavailable.

**Status: implemented (Phase 7).**

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

Server-Sent Events progress stream.

**Status: implemented (Phase 7).**

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

The bands are: validation and preprocessing 0-10 %, loading the model 10-15 %,
inference 15-90 %, post-processing and encoding 90-100 %. `tilesDone` and
`tilesTotal` appear only inside the inference band, because that is the only
stage with a real count behind them.

A client that subscribes late — or reconnects — is sent the most recent event
immediately, and a job that has already finished yields its terminal event and
closes, so a stream never hangs waiting for something that has already
happened.

### `GET /api/jobs/{jobId}/result`

The processed image.

```
Content-Type: image/png
Content-Disposition: attachment; filename="pixelforge-9f1c0f2a-5120x2880.png"
```

Returns `job_not_completed` (409) if the job has not finished, `job_not_found`
(404) if the id is unknown or the file has been swept. Supports `Range` requests
so a large download can resume.

**Status: implemented (Phase 7).**

### `GET /api/jobs/{jobId}/preview`

Resolution-capped JPEG of the result for the comparison viewer, so a 200 MP
output is never loaded into the DOM. Long edge capped at 4096 px, quality 92.
*Phase 9.*

Optional query parameters for high-zoom inspection: `x`, `y`, `w`, `h` request a
full-resolution crop of the source region instead of a downscaled whole.

### `GET /api/jobs/{jobId}/thumbnail`

256 px thumbnail for the history grid. *Phase 10.*

### `DELETE /api/jobs/{jobId}`

Cancel if running, delete otherwise.

**Status: implemented (Phase 7).**

Cancellation is cooperative: a flag is set and the tiling loop checks it between
tiles, so cancellation takes effect within one tile rather than immediately.
A job that has already completed is deleted along with its files.

A cancelled job keeps no output: the partial image is deleted and
`/result` continues to answer `job_not_completed`. The flag is both held in
memory, where the worker thread can read it between tiles, and persisted, so a
cancellation requested before the job starts is honoured rather than raced.

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
