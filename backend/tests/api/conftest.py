"""Shared harness for the job and history API tests.

The inference engine is replaced with a stub that enlarges by pixel repetition:
these tests are about the HTTP contract, the job lifecycle and the storage
rules, and a real 4x pass would add seconds to each of them without testing
anything the integration suite does not already cover with real weights.

Everything else is real - real multipart uploads, real validation, real SQLite,
the real queue and worker thread, real files on disk.

Lives in a conftest rather than being imported between test modules: pytest
collects fixtures from here automatically, and importing a fixture by name
gives every consumer a redefinition warning.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api import deps
from app.core import runtime as runtime_module
from app.core.config import Settings
from app.core.database import init_engine
from app.core.runtime import Runtime, repository_scope
from app.models.enums import JobStatus
from app.services.enhancement_service import EnhancementResult
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.storage_service import StorageService
from app.workers.progress import ProgressBroker
from app.workers.queue import InProcessJobQueue
from app.workers.runner import JobRunner

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class StubEnhancement:
    """Enlarges by repeating pixels, with the same interface as the real one.

    Progress is reported per "tile" so the runner's arithmetic and the event
    stream are exercised exactly as they are with the real engine.
    """

    def __init__(self, settings: Settings, models: ModelService) -> None:
        self._settings = settings
        self._models = models
        self.calls: list[Any] = []
        self.delay = 0.0
        self.fail_with: Exception | None = None

    def plan(self, model_id: str, scale: int) -> list[str]:
        from app.services.enhancement_service import EnhancementService

        return EnhancementService(self._settings, models=self._models).plan(model_id, scale)

    def enhance(
        self,
        image: np.ndarray[Any, Any],
        request: Any,
        *,
        on_progress: Any = None,
        should_cancel: Any = None,
    ) -> EnhancementResult:
        import time

        from app.inference.upscaler import JobCancelledError, UpscaleReport

        self.calls.append(request)
        if self.fail_with is not None:
            raise self.fail_with

        steps = 4
        for index in range(steps):
            if should_cancel is not None and should_cancel():
                raise JobCancelledError
            if self.delay:
                time.sleep(self.delay)
            if on_progress is not None:
                on_progress((index + 1) / steps)

        scale = request.scale
        enlarged = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

        return EnhancementResult(
            image=enlarged,
            scale=scale,
            passes=["stub"],
            reports=[UpscaleReport(device="cpu", tile_size=256, tiles=steps, fp16=False)],
        )

    def release(self) -> int:
        return 0


@pytest.fixture
async def app_context(settings: Settings, tmp_path: Any) -> AsyncIterator[dict[str, Any]]:
    """The real application wiring, with the engine stubbed out.

    The real manifest is copied in: the stub needs no weights, but model and
    scale validation is real, and validating against a fake registry would test
    the fixture rather than the rules.
    """
    import shutil

    from app.core import migrations
    from app.core.config import REPO_ROOT
    from app.main import create_app

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "models" / "manifest.json", settings.models_dir / "manifest.json")

    await migrations.upgrade_to_head(settings)
    init_engine(settings)

    storage = StorageService(settings)
    storage.ensure_ready()

    models = ModelService(settings)
    broker = ProgressBroker()
    broker.bind_loop(asyncio.get_running_loop())
    enhancement = StubEnhancement(settings, models)

    holder: dict[str, JobRunner] = {}

    async def handle(job_id: str) -> None:
        await holder["runner"].run(job_id)

    queue = InProcessJobQueue(settings, handle)
    runner = JobRunner(
        settings,
        scope=repository_scope,
        queue=queue,
        broker=broker,
        enhancement=enhancement,  # type: ignore[arg-type]
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
            enhancement=enhancement,  # type: ignore[arg-type]
            runner=runner,
            cleanup=NoSweeper(),  # type: ignore[arg-type]
        )
    )

    app = create_app()
    app.dependency_overrides[deps.get_settings] = lambda: settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "queue": queue,
            "broker": broker,
            "enhancement": enhancement,
            "storage": storage,
            "settings": settings,
        }

    await queue.stop()
    runtime_module.set_runtime(None)

    from app.core.database import dispose_engine

    await dispose_engine()


def png_bytes(width: int = 64, height: int = 48, mode: str = "RGB") -> bytes:
    generator = np.random.default_rng(3)
    array = generator.integers(0, 256, (height, width, len(mode)), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array.squeeze() if len(mode) == 1 else array, mode=mode).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


async def submit(client: AsyncClient, **fields: Any) -> Any:
    """POST a real multipart upload."""
    content = fields.pop("content", None) or png_bytes()
    data = {key: str(value) for key, value in fields.items() if value is not None}

    return await client.post(
        "/api/jobs",
        files={"image": ("input.png", content, "image/png")},
        data=data,
    )


async def wait_for_status(client: AsyncClient, job_id: str, *, timeout: float = 20.0) -> Any:
    """Poll until the job reaches a terminal state."""
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/jobs/{job_id}")
        body = response.json()
        if JobStatus(body["status"]).is_terminal:
            return body
        await asyncio.sleep(0.02)

    raise AssertionError(f"job {job_id} did not finish within {timeout}s")
