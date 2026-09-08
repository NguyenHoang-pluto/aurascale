"""API contract tests for the system and model endpoints."""

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport

from app.core.config import Settings
from app.services.model_service import _read_manifest


@pytest.fixture
async def client(app) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestSystemEndpoint:
    async def test_returns_the_documented_shape(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/system")

        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {
            "device",
            "deviceReason",
            "torch",
            "gpu",
            "cpuName",
            "ramTotalMb",
            "ramAvailableMb",
            "pythonVersion",
            "platform",
            "fp16",
            "tileSize",
            "tilePad",
        }

    async def test_reports_a_real_device(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/api/system")).json()

        assert body["device"] in {"cuda", "cpu"}
        assert body["deviceReason"]

    async def test_serialises_camel_case_for_the_typescript_client(
        self, client: httpx.AsyncClient
    ) -> None:
        body = (await client.get("/api/system")).json()

        assert "cudaAvailable" in body["torch"]
        assert "cpuCoresLogical" in body
        # snake_case must not leak through.
        assert "cuda_available" not in body["torch"]
        assert "ram_total_mb" not in body

    async def test_reports_real_host_memory(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/api/system")).json()

        assert body["ramTotalMb"] > 0
        assert 0 <= body["ramAvailableMb"] <= body["ramTotalMb"]

    async def test_gpu_block_matches_cuda_availability(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/api/system")).json()

        if body["torch"]["cudaAvailable"]:
            assert body["gpu"] is not None
            assert set(body["gpu"]) == {
                "name",
                "vramTotalMb",
                "vramFreeMb",
                "capability",
            }
            assert body["gpu"]["vramFreeMb"] <= body["gpu"]["vramTotalMb"]
        else:
            assert body["gpu"] is None

    async def test_never_discloses_filesystem_paths(self, client: httpx.AsyncClient) -> None:
        """§ 16: internal paths must not appear in a response body."""
        raw = (await client.get("/api/system")).text

        assert "storage" not in raw.lower() or "/storage/" not in raw
        assert "\\Tool\\" not in raw
        assert "C:\\" not in raw


class TestModelsEndpoint:
    async def test_lists_the_shipped_models(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/models")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 3

    async def test_each_entry_has_the_documented_fields(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/api/models")).json()

        for entry in body:
            assert set(entry) == {
                "id",
                "name",
                "description",
                "arch",
                "scale",
                "supportsDenoise",
                "supportedScales",
                "downloaded",
                "sizeMb",
            }

    async def test_each_entry_publishes_the_scales_it_can_reach(
        self, client: httpx.AsyncClient
    ) -> None:
        """The client offers exactly these, rather than restating the routing
        rule and drifting from what a job would actually accept."""
        body = (await client.get("/api/models")).json()
        by_id = {entry["id"]: entry for entry in body}

        # A 4x model reaches 8x by a second 2x pass; a 2x model reaches only 2x.
        assert by_id["RealESRGAN_x4plus"]["supportedScales"] == [4, 8]
        assert by_id["RealESRGAN_x2plus"]["supportedScales"] == [2]
        assert by_id["realesr-general-x4v3"]["supportedScales"] == [4, 8]

    async def test_the_native_scale_is_always_supported(self, client: httpx.AsyncClient) -> None:
        for entry in (await client.get("/api/models")).json():
            assert entry["scale"] in entry["supportedScales"]

    async def test_omits_non_selectable_weight_sets(self, client: httpx.AsyncClient) -> None:
        """The DNI denoise counterpart is blended, never chosen directly."""
        ids = {entry["id"] for entry in (await client.get("/api/models")).json()}

        assert "realesr-general-wdn-x4v3" not in ids
        assert "RealESRGAN_x4plus" in ids

    async def test_reports_weights_as_not_downloaded_before_they_exist(
        self, client: httpx.AsyncClient
    ) -> None:
        body = (await client.get("/api/models")).json()

        # The test settings point at an empty models directory.
        for entry in body:
            assert entry["downloaded"] is False
            assert entry["sizeMb"] is None

    async def test_reports_weights_once_present(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> None:
        _read_manifest.cache_clear()
        (settings.models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"x" * (2 * 1024 * 1024))

        body = (await client.get("/api/models")).json()

        entry = next(item for item in body if item["id"] == "RealESRGAN_x4plus")
        assert entry["downloaded"] is True
        assert entry["sizeMb"] == 2.0

    async def test_missing_manifest_returns_a_problem_document(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> None:
        _read_manifest.cache_clear()
        (settings.models_dir / "manifest.json").unlink()
        _read_manifest.cache_clear()

        response = await client.get("/api/models")

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.json()["code"] == "model_not_found"


class TestOpenApi:
    async def test_new_endpoints_are_documented(self, client: httpx.AsyncClient) -> None:
        paths = (await client.get("/openapi.json")).json()["paths"]

        assert "/api/system" in paths
        assert "/api/models" in paths
        assert "/api/health" in paths


@pytest.fixture(autouse=True)
def _seed_manifest(settings: Settings) -> None:
    """Copy the real manifest into the test models directory.

    The shipped manifest is used rather than a fixture so these tests fail if
    it is edited into an invalid state.
    """
    _read_manifest.cache_clear()
    from app.core.config import BACKEND_ROOT

    source = BACKEND_ROOT.parent / "models" / "manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    (settings.models_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    _read_manifest.cache_clear()
