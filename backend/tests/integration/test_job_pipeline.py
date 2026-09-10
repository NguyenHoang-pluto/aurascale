"""A real image through the real job flow, over HTTP, on the real model.

Nothing is stubbed here. An upload goes through the API, a worker thread runs
the actual Real-ESRGAN weights on the GPU (or CPU), progress arrives over SSE
with measured tile counts, and the finished file is downloaded and opened.

This is the test that would catch a job pipeline that reports success without
having enhanced anything, so its assertions are about the pixels as much as the
status codes.

Marked `slow`, and skipped when the weights are not downloaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import shutil
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api import deps
from app.core import migrations
from app.core import runtime as runtime_module
from app.core.config import REPO_ROOT, Settings
from app.core.database import dispose_engine, init_engine
from app.core.runtime import Runtime, repository_scope
from app.models.enums import JobStatus
from app.services.enhancement_service import EnhancementService
from app.services.image_service import ImageService
from app.services.mode_planner import CREATIVE_MODEL, STANDARD_MODEL
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.workers.progress import ProgressBroker
from app.workers.queue import InProcessJobQueue
from app.workers.runner import JobRunner

pytestmark = [pytest.mark.slow, pytest.mark.integration, pytest.mark.anyio]

# The smallest real model, so the suite stays usable.
MODEL = "realesr-general-x4v3"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def live_app(settings: Settings) -> AsyncIterator[dict[str, Any]]:
    """The whole application, wired exactly as `lifespan` wires it."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    real_models = REPO_ROOT / "models"
    shutil.copy(real_models / "manifest.json", settings.models_dir / "manifest.json")

    models = ModelService(settings)

    # Both halves of the pair, because `MODEL` now runs a DNI blend by default.
    #
    # This fixture used to copy the primary checkpoint alone, and that was
    # enough while an unspecified denoise resolved to `None` - no blend, so the
    # `wdn` counterpart was never read. `DEFAULT_DENOISE` is 0.25 now, which is
    # a blend, and a blend needs both files.
    #
    # Copying both is not a workaround. It is what production already
    # guarantees: `ModelDownloadService` treats the pair as one unit, on the
    # grounds that "the denoise-capable model is useless on its own", so anyone
    # who has `MODEL` has its counterpart too. The fixture was the only place
    # where half a pair could exist.
    required = [f"{MODEL}.pth"]
    pair = models.get(MODEL).entry.denoise_pair
    if pair is not None:
        required.append(f"{pair}.pth")

    for name in required:
        if (settings.models_dir / name).is_file():
            continue
        # Symlinking would be neater, but a copy works without privileges.
        source = real_models / name
        if not source.is_file():
            pytest.skip(f"{name} is not downloaded; run scripts/download_models.py")
        shutil.copy(source, settings.models_dir / name)

    from app.main import create_app

    await migrations.upgrade_to_head(settings)
    init_engine(settings)

    storage = StorageService(settings)
    storage.ensure_ready()

    broker = ProgressBroker()
    broker.bind_loop(asyncio.get_running_loop())
    enhancement = EnhancementService(settings, models=models)

    holder: dict[str, JobRunner] = {}

    async def handle(job_id: str) -> None:
        await holder["runner"].run(job_id)

    queue = InProcessJobQueue(settings, handle)
    runner = JobRunner(
        settings,
        scope=repository_scope,
        queue=queue,
        broker=broker,
        enhancement=enhancement,
        images=ImageService(settings),
        storage=storage,
        models=models,
    )
    holder["runner"] = runner
    await queue.start()

    class NoSweeper:
        async def start(self) -> None: ...
        async def stop(self) -> None: ...

    runtime_module.set_runtime(
        Runtime(
            queue=queue,
            broker=broker,
            enhancement=enhancement,
            runner=runner,
            cleanup=NoSweeper(),  # type: ignore[arg-type]
        )
    )

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield {"client": client, "settings": settings, "enhancement": enhancement}

    await queue.stop()
    enhancement.release()
    runtime_module.set_runtime(None)
    await dispose_engine()


def photo_bytes(width: int = 160, height: int = 120) -> tuple[bytes, np.ndarray[Any, Any]]:
    """A synthetic photograph: gradients, hard edges and grain."""
    generator = np.random.default_rng(12)
    rows, columns = np.mgrid[0:height, 0:width]

    picture = np.dstack(
        [columns / width * 255, rows / height * 255, np.full((height, width), 150.0)]
    )
    picture[30:90, 40:120] = 235
    picture[45:70, 60:100] = 20
    picture = np.clip(picture + generator.normal(0, 6, picture.shape), 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(picture).save(buffer, format="PNG")
    return buffer.getvalue(), picture


async def wait_for_terminal(client: AsyncClient, job_id: str, timeout: float = 300.0) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        body = (await client.get(f"/api/jobs/{job_id}")).json()
        if JobStatus(body["status"]).is_terminal:
            return body
        await asyncio.sleep(0.1)

    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


async def test_a_real_image_is_enhanced_through_the_whole_job_flow(
    live_app: dict[str, Any],
) -> None:
    """Upload, run the real model, stream progress, download the result."""
    client = live_app["client"]
    content, original = photo_bytes()

    created = await client.post(
        "/api/jobs",
        files={"image": ("photo.png", content, "image/png")},
        data={"model": MODEL, "scale": "4"},
    )
    assert created.status_code == 202
    job_id = created.json()["jobId"]

    # Follow the stream to completion, collecting what it actually reported.
    events: list[tuple[str, dict[str, Any]]] = []
    async with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        name = ""
        async for line in stream.aiter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
                if name in {"completed", "failed", "cancelled"}:
                    break

    names = [name for name, _ in events]
    assert names[-1] == "completed", f"job did not complete: {events[-1]}"

    # Progress was measured, monotonic, and inside the inference band.
    reports = [data for name, data in events if name == "progress"]
    assert reports, "no measured progress was reported"
    assert [r["progress"] for r in reports] == sorted(r["progress"] for r in reports)
    assert all(15 <= r["progress"] <= 90 for r in reports)
    assert all(r["tilesDone"] <= r["tilesTotal"] for r in reports)
    assert reports[-1]["tilesDone"] == reports[-1]["tilesTotal"]

    # The record says what actually ran.
    body = (await client.get(f"/api/jobs/{job_id}")).json()
    assert body["status"] == "completed"
    assert body["progress"] == 100
    assert body["output"] == {
        "width": 640,
        "height": 480,
        "sizeBytes": body["output"]["sizeBytes"],
        "format": "PNG",
    }
    assert body["device"] in {"cuda", "cpu"}
    assert body["processingMs"] > 0

    # The downloaded file is a real 4x enlargement of the input, not a resize
    # of nothing: shrinking it back should land near the original.
    result = await client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "image/png"

    with Image.open(io.BytesIO(result.content)) as enhanced:
        assert enhanced.size == (640, 480)
        pixels = np.asarray(enhanced.convert("RGB"))

    import cv2

    shrunk = cv2.resize(pixels, (160, 120), interpolation=cv2.INTER_AREA)
    difference = np.abs(shrunk.astype(int) - original.astype(int)).mean()
    assert difference < 12, f"result does not resemble the input (mean |diff| {difference:.1f})"


async def test_a_job_cancelled_during_inference_keeps_no_result(
    live_app: dict[str, Any],
) -> None:
    """Cancellation lands between tiles, and never leaves a partial image."""
    client = live_app["client"]
    settings = live_app["settings"]
    # Large enough, with small tiles, that there are many tile boundaries to
    # stop at rather than one long forward pass.
    content, _ = photo_bytes(320, 240)

    created = await client.post(
        "/api/jobs",
        files={"image": ("photo.png", content, "image/png")},
        data={"model": MODEL, "scale": "4", "settings": json.dumps({"tileSize": 64})},
    )
    job_id = created.json()["jobId"]

    for _ in range(600):
        body = (await client.get(f"/api/jobs/{job_id}")).json()
        if body["status"] == "processing" and body["stage"] == "running_inference":
            break
        await asyncio.sleep(0.05)

    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 204

    final = await wait_for_terminal(client, job_id)
    assert final["status"] == "cancelled"
    assert final["output"] is None

    assert (await client.get(f"/api/jobs/{job_id}/result")).status_code == 409
    assert list(settings.outputs_dir.iterdir()) == [], "a partial result was kept"


async def test_a_mode_alone_changes_what_the_real_pipeline_produces(
    live_app: dict[str, Any],
) -> None:
    """The Phase 4 F1 regression, end to end on real weights.

    F1 found Standard and Creative producing byte-identical output because the
    client always sent an explicit model and denoise, so `mode_planner`'s
    defaults could never apply. This submits the payload the corrected client
    emits - **no model, no denoise** - and asserts the mode alone decides.

    If this fails byte-identical again, the wiring bug is back.
    """
    client = live_app["client"]
    settings = live_app["settings"]

    # The fixture copies only MODEL. Standard needs x4plus, and Creative's 0.25
    # is a DNI blend, so it also needs the wdn pair.
    for name in (f"{STANDARD_MODEL}.pth", "realesr-general-wdn-x4v3.pth"):
        source = REPO_ROOT / "models" / name
        if not source.is_file():
            pytest.skip(f"{name} is not downloaded; run scripts/download_models.py")
        target = settings.models_dir / name
        if not target.is_file():
            shutil.copy(source, target)

    content, _ = photo_bytes()
    results: dict[str, dict[str, Any]] = {}

    for mode in ("standard", "creative"):
        created = await client.post(
            "/api/jobs",
            files={"image": ("photo.png", content, "image/png")},
            # Exactly what the fixed client sends: mode only.
            data={"scale": "4", "format": "png", "mode": mode, "settings": "{}"},
        )
        assert created.status_code == 202, created.text
        body = await wait_for_terminal(client, created.json()["jobId"])
        assert body["status"] == "completed", body

        result = await client.get(f"/api/jobs/{created.json()['jobId']}/result")
        assert result.status_code == 200
        results[mode] = {
            "model": body["model"],
            "sha256": hashlib.sha256(result.content).hexdigest(),
            "bytes": len(result.content),
        }

    # The mode chose the model, which is what the explicit field used to prevent.
    assert results["standard"]["model"] == STANDARD_MODEL
    assert results["creative"]["model"] == CREATIVE_MODEL

    # And the pixels actually differ. Byte equality here is the bug returning.
    assert results["standard"]["sha256"] != results["creative"]["sha256"], (
        "Standard and Creative produced identical output - Enhancement Mode is inert again"
    )


async def test_an_explicit_model_still_beats_the_mode_in_the_real_pipeline(
    live_app: dict[str, Any],
) -> None:
    """The other half of the F1 fix: the precedence rule must not have flipped.

    A client that names a model keeps getting it, whatever the mode says. That
    is what keeps every client written before modes existed working, and the fix
    must not have bought mode support by breaking it.
    """
    client = live_app["client"]

    content, _ = photo_bytes()
    created = await client.post(
        "/api/jobs",
        files={"image": ("photo.png", content, "image/png")},
        data={"model": MODEL, "scale": "4", "mode": "standard", "settings": "{}"},
    )
    assert created.status_code == 202, created.text
    body = await wait_for_terminal(client, created.json()["jobId"])

    assert body["status"] == "completed"
    # MODEL is the Creative model; Standard's default would be x4plus.
    assert body["model"] == MODEL != STANDARD_MODEL


async def test_denoise_settings_reach_the_real_model(live_app: dict[str, Any]) -> None:
    """The DNI blend is applied by a job, not just by a direct engine call."""
    client = live_app["client"]
    settings = live_app["settings"]

    wdn = REPO_ROOT / "models" / "realesr-general-wdn-x4v3.pth"
    if not wdn.is_file():
        pytest.skip("the denoise pair is not downloaded")
    shutil.copy(wdn, settings.models_dir / wdn.name)

    content, _ = photo_bytes(96, 96)
    created = await client.post(
        "/api/jobs",
        files={"image": ("photo.png", content, "image/png")},
        data={
            "model": MODEL,
            "scale": "4",
            "settings": json.dumps({"denoiseStrength": 0.5}),
        },
    )

    final = await wait_for_terminal(client, created.json()["jobId"])

    assert final["status"] == "completed", final.get("error")
    assert final["output"]["width"] == 384


async def test_an_upload_that_is_not_an_image_never_reaches_the_gpu(
    live_app: dict[str, Any],
) -> None:
    client = live_app["client"]

    response = await client.post(
        "/api/jobs",
        files={"image": ("payload.png", b"\x7fELF\x02\x01\x01" + b"\x00" * 128, "image/png")},
        data={"model": MODEL, "scale": "4"},
    )

    assert response.status_code == 415
    assert list(live_app["settings"].inputs_dir.iterdir()) == []


async def test_a_job_cancelled_while_encoding_does_not_finish_as_completed(
    live_app: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must be honoured after inference, not only during it.

    The runner checks for cancellation between tiles and once more before
    post-processing, and then encodes. Encoding a 4x result is not instant -
    a 16x PNG is hundreds of megapixels - so there is a real window between the
    last checkpoint and the write of a terminal status. A `DELETE` landing in
    that window used to be answered "cancelled" while the job went on to record
    `completed`, leaving the output file on disk and the row carrying both
    `status=completed` and `cancel_requested=True`.

    The window is opened deliberately here rather than raced for: `encode` is
    wrapped to sleep before doing its real work, which makes the ordering
    deterministic instead of timing-dependent.
    """
    client = live_app["client"]
    settings = live_app["settings"]
    content, _ = photo_bytes(160, 120)

    real_encode = ImageService.encode

    def slow_encode(self: ImageService, *args: Any, **kwargs: Any) -> Any:
        # Long enough for the DELETE below to be handled while this thread
        # sits here. The loop keeps running because encode is off-thread.
        time.sleep(2.0)
        return real_encode(self, *args, **kwargs)

    monkeypatch.setattr(ImageService, "encode", slow_encode)

    created = await client.post(
        "/api/jobs",
        files={"image": ("photo.png", content, "image/png")},
        data={"model": MODEL, "scale": "4"},
    )
    job_id = created.json()["jobId"]

    for _ in range(900):
        body = (await client.get(f"/api/jobs/{job_id}")).json()
        if body["stage"] == "encoding":
            break
        if JobStatus(body["status"]).is_terminal:
            pytest.fail(f"job reached {body['status']} before encoding could be observed")
        await asyncio.sleep(0.05)
    else:  # pragma: no cover - the slow encode makes this unreachable
        pytest.fail("job never reached the encoding stage")

    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 204

    final = await wait_for_terminal(client, job_id)

    assert final["status"] == "cancelled", (
        "a job cancelled during encoding recorded itself as completed; "
        "the DELETE was answered 'cancelled' and the user was told so"
    )
    assert final["output"] is None
    assert (await client.get(f"/api/jobs/{job_id}/result")).status_code == 409
    assert list(settings.outputs_dir.iterdir()) == [], "a cancelled job kept its output"
