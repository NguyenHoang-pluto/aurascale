# Container definitions

Docker support is delivered in **Phase 12**. This directory will hold:

| File | Purpose |
| --- | --- |
| `Dockerfile.frontend` | Multi-stage: Node builds the bundle, nginx serves it |
| `Dockerfile.backend` | CPU image, installs `requirements-cpu.txt` |
| `Dockerfile.backend.gpu` | CUDA base image, installs `requirements-cuda.txt` |
| `nginx.conf` | Static serving plus `/api` proxy, with SSE buffering disabled |

The compose files live at the repository root: `docker-compose.yml` for CPU and
`docker-compose.gpu.yml` as the GPU overlay.

The design these will be built to is documented in
[`../docs/deployment.md`](../docs/deployment.md).

One constraint worth knowing before Phase 12: nginx buffers proxied responses by
default, which would break the Server-Sent Events progress stream. The location
block for `/api/jobs/*/events` needs `proxy_buffering off;` and
`proxy_read_timeout` raised well above the longest expected job.
