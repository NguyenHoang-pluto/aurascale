"""History listing and thumbnails.

These use the shared API harness from the package conftest: a real app, a real
database, a real queue and a stubbed engine. What is under test is the listing contract - order, paging,
filtering, and what a list item is allowed to contain - and the thumbnail
endpoint's lazy cache.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image

from tests.api.conftest import submit, wait_for_status

pytestmark = pytest.mark.anyio


async def complete_jobs(app_context: dict[str, Any], count: int) -> list[str]:
    """Submit `count` jobs and wait for each, oldest first."""
    ids: list[str] = []

    for _ in range(count):
        created = (
            await submit(app_context["client"], model="realesr-general-x4v3", scale=4)
        ).json()
        await wait_for_status(app_context["client"], created["jobId"])
        ids.append(str(created["jobId"]))

    return ids


# ------------------------------------------------------------------ listing


async def test_an_empty_history_is_an_empty_page_not_an_error(
    app_context: dict[str, Any],
) -> None:
    response = await app_context["client"].get("/api/jobs")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_history_lists_finished_jobs_newest_first(
    app_context: dict[str, Any],
) -> None:
    ids = await complete_jobs(app_context, 3)

    body = (await app_context["client"].get("/api/jobs")).json()

    assert [item["jobId"] for item in body["items"]] == list(reversed(ids))
    assert body["total"] == 3


async def test_a_list_item_carries_what_the_grid_shows(
    app_context: dict[str, Any],
) -> None:
    await complete_jobs(app_context, 1)

    item = (await app_context["client"].get("/api/jobs")).json()["items"][0]

    assert item["status"] == "completed"
    assert item["model"] == "realesr-general-x4v3"
    assert item["scale"] == 4
    assert item["input"] == {
        "width": 64,
        "height": 48,
        "sizeBytes": item["input"]["sizeBytes"],
        "format": "PNG",
    }
    assert item["output"]["width"] == 256
    assert item["processingMs"] >= 0
    assert item["createdAt"] is not None


async def test_a_list_item_never_carries_a_storage_path(
    app_context: dict[str, Any],
) -> None:
    """History is the widest surface over the job table, so this is where a
    path leak would be easiest to miss."""
    settings = app_context["settings"]
    await complete_jobs(app_context, 1)

    body = (await app_context["client"].get("/api/jobs")).json()
    item = body["items"][0]

    assert str(settings.storage_dir) not in str(body)
    assert not [key for key in item if "path" in key.lower()]


# ------------------------------------------------------------------- paging


async def test_paging_returns_a_slice_and_the_full_total(
    app_context: dict[str, Any],
) -> None:
    ids = await complete_jobs(app_context, 5)

    first = (await app_context["client"].get("/api/jobs", params={"limit": 2})).json()
    second = (await app_context["client"].get("/api/jobs", params={"limit": 2, "offset": 2})).json()

    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    # total counts everything, not just the page, so the UI can size its pager.
    assert first["total"] == 5
    assert second["total"] == 5
    assert first["items"][0]["jobId"] == ids[-1]
    assert second["items"][0]["jobId"] == ids[-3]


async def test_paging_past_the_end_is_empty_rather_than_an_error(
    app_context: dict[str, Any],
) -> None:
    await complete_jobs(app_context, 2)

    body = (await app_context["client"].get("/api/jobs", params={"offset": 50})).json()

    assert body["items"] == []
    assert body["total"] == 2


async def test_the_page_size_is_capped(app_context: dict[str, Any]) -> None:
    """Without the cap a client could ask for the whole table in one request."""
    response = await app_context["client"].get("/api/jobs", params={"limit": 500})

    assert response.status_code == 422


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": -1}, {"offset": -1}])
async def test_nonsense_paging_is_refused(
    app_context: dict[str, Any], params: dict[str, int]
) -> None:
    response = await app_context["client"].get("/api/jobs", params=params)

    assert response.status_code == 422


async def test_the_maximum_page_size_is_accepted(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].get("/api/jobs", params={"limit": 100})

    assert response.status_code == 200
    assert response.json()["limit"] == 100


# ---------------------------------------------------------------- filtering


async def test_history_can_be_filtered_by_status(app_context: dict[str, Any]) -> None:
    await complete_jobs(app_context, 2)

    completed = (
        await app_context["client"].get("/api/jobs", params={"status": "completed"})
    ).json()
    failed = (await app_context["client"].get("/api/jobs", params={"status": "failed"})).json()

    assert completed["total"] == 2
    assert failed == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_a_failed_job_appears_in_history_with_its_error(
    app_context: dict[str, Any],
) -> None:
    from app.core.exceptions import InferenceError

    app_context["enhancement"].fail_with = InferenceError("The model failed.", technical="stub")
    created = (await submit(app_context["client"], model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(app_context["client"], created["jobId"])

    item = (await app_context["client"].get("/api/jobs", params={"status": "failed"})).json()[
        "items"
    ][0]

    assert item["status"] == "failed"
    assert item["error"]["code"] == "inference_failed"
    assert item["output"] is None


async def test_an_unknown_status_is_refused(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].get("/api/jobs", params={"status": "wobbly"})

    assert response.status_code == 422


# --------------------------------------------------------------- thumbnails


async def test_a_thumbnail_is_a_small_webp(app_context: dict[str, Any]) -> None:
    job_id = (await complete_jobs(app_context, 1))[0]

    response = await app_context["client"].get(f"/api/jobs/{job_id}/thumbnail")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "WEBP"
        # This result is 256x192, already inside the tile, so it is not scaled.
        assert max(image.size) <= 256


async def test_a_thumbnail_is_cached_and_reused(app_context: dict[str, Any]) -> None:
    """A grid asks for twenty at once; building them per view would be the
    page's dominant cost."""
    settings = app_context["settings"]
    job_id = (await complete_jobs(app_context, 1))[0]

    await app_context["client"].get(f"/api/jobs/{job_id}/thumbnail")
    cached = settings.thumbs_dir / f"{job_id}.webp"
    assert cached.is_file()

    stamped = cached.stat().st_mtime_ns
    await app_context["client"].get(f"/api/jobs/{job_id}/thumbnail")

    assert cached.stat().st_mtime_ns == stamped


async def test_a_thumbnail_before_completion_is_a_conflict(
    app_context: dict[str, Any],
) -> None:
    app_context["enhancement"].delay = 0.2
    created = (await submit(app_context["client"], model="realesr-general-x4v3", scale=4)).json()

    response = await app_context["client"].get(f"/api/jobs/{created['jobId']}/thumbnail")

    assert response.status_code == 409
    await wait_for_status(app_context["client"], created["jobId"])


async def test_a_thumbnail_for_an_unknown_job_is_a_404(
    app_context: dict[str, Any],
) -> None:
    response = await app_context["client"].get(f"/api/jobs/{'0' * 32}/thumbnail")

    assert response.status_code == 404


async def test_a_thumbnail_for_a_swept_result_is_a_404(
    app_context: dict[str, Any],
) -> None:
    """The row can outlive the file, and the grid has to cope with that."""
    settings = app_context["settings"]
    job_id = (await complete_jobs(app_context, 1))[0]

    for path in settings.outputs_dir.iterdir():
        path.unlink()

    response = await app_context["client"].get(f"/api/jobs/{job_id}/thumbnail")

    assert response.status_code == 404


# ----------------------------------------------------------------- deletion


async def test_a_deleted_job_leaves_history(app_context: dict[str, Any]) -> None:
    ids = await complete_jobs(app_context, 2)

    await app_context["client"].delete(f"/api/jobs/{ids[0]}")

    body = (await app_context["client"].get("/api/jobs")).json()
    assert [item["jobId"] for item in body["items"]] == [ids[1]]
    assert body["total"] == 1


async def test_deleting_a_job_removes_its_thumbnail(app_context: dict[str, Any]) -> None:
    settings = app_context["settings"]
    job_id = (await complete_jobs(app_context, 1))[0]
    await app_context["client"].get(f"/api/jobs/{job_id}/thumbnail")
    assert (settings.thumbs_dir / f"{job_id}.webp").is_file()

    await app_context["client"].delete(f"/api/jobs/{job_id}")

    assert list(settings.thumbs_dir.iterdir()) == []
