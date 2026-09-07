from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from app import __version__


@pytest.fixture
async def client(app) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_reports_ok(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["uptimeSeconds"] >= 0


async def test_unknown_route_returns_problem_json(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert set(body) >= {"type", "title", "status", "code", "detail"}


async def test_openapi_schema_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/health" in response.json()["paths"]
