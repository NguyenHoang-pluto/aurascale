"""The job endpoints, end to end through the ASGI app.

The app, the stubbed engine and the submission helpers come from the package
conftest, which the history tests share.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest
from PIL import Image

from tests.api.conftest import png_bytes, submit, wait_for_status

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- submit


async def test_a_submission_is_accepted_and_queued(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], model="realesr-general-x4v3", scale=4)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["queuePosition"] == 0
    assert len(body["jobId"]) == 32


async def test_the_job_runs_and_completes(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    final = await wait_for_status(client, created["jobId"])

    assert final["status"] == "completed"
    assert final["progress"] == 100
    assert final["stage"] is None
    assert final["input"] == {
        "width": 64,
        "height": 48,
        "sizeBytes": final["input"]["sizeBytes"],
        "format": "PNG",
    }
    assert final["output"]["width"] == 256
    assert final["output"]["height"] == 192
    assert final["processingMs"] >= 0
    assert final["error"] is None


async def test_defaults_are_applied_when_nothing_is_specified(app_context: dict[str, Any]) -> None:
    created = (await submit(app_context["client"])).json()

    body = (await app_context["client"].get(f"/api/jobs/{created['jobId']}")).json()

    assert body["model"] == app_context["settings"].default_model
    assert body["scale"] == app_context["settings"].default_scale


async def test_a_response_never_contains_a_storage_path(app_context: dict[str, Any]) -> None:
    """Paths are server-side detail (architecture § 10); a response must not
    carry the storage root, a path-shaped field, or the uploaded filename."""
    client = app_context["client"]
    settings = app_context["settings"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    final = await wait_for_status(client, created["jobId"])

    serialised = json.dumps({"created": created, "final": final})

    assert str(settings.storage_dir) not in serialised
    assert settings.storage_dir.name not in serialised
    assert "input.png" not in serialised
    assert not [key for key in final if "path" in key.lower()]
    assert not [key for key in final.get("input", {}) if "path" in key.lower()]


# --------------------------------------------------------------- validation


async def test_a_non_image_upload_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], content=b"MZ\x90\x00 not an image")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_format"


async def test_a_truncated_image_is_refused(app_context: dict[str, Any]) -> None:
    whole = png_bytes(256, 256)
    response = await submit(app_context["client"], content=whole[: len(whole) // 2])

    assert response.status_code == 422
    assert response.json()["code"] == "corrupted_image"


async def test_an_image_below_the_minimum_size_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], content=png_bytes(16, 16))

    assert response.status_code == 422
    assert response.json()["code"] == "image_too_small"


async def test_an_unknown_model_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], model="NoSuchModel")

    assert response.status_code == 404
    assert response.json()["code"] == "model_not_found"


async def test_a_scale_the_model_cannot_produce_is_refused(app_context: dict[str, Any]) -> None:
    """A 4x model asked for 2x: refused with a suggestion, never swapped silently."""
    response = await submit(app_context["client"], model="RealESRGAN_x4plus", scale=2)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_parameters"
    assert body["context"]["suggestion"] == "RealESRGAN_x2plus"


async def test_an_unsupported_scale_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], scale=3)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_parameters"


async def test_denoise_on_a_model_without_it_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(
        app_context["client"],
        model="RealESRGAN_x4plus",
        scale=4,
        settings='{"denoiseStrength": 0.5}',
    )

    assert response.status_code == 422
    assert "denoise" in response.json()["detail"].lower()


async def test_malformed_settings_json_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], settings="{not json")

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_parameters"


async def test_an_out_of_range_strength_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], settings='{"sharpenStrength": 4}')

    assert response.status_code == 422


async def test_a_refused_submission_leaves_no_file_behind(app_context: dict[str, Any]) -> None:
    settings = app_context["settings"]
    await submit(app_context["client"], content=b"not an image at all")

    assert list(settings.inputs_dir.iterdir()) == []


async def test_settings_reach_the_engine(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    created = (
        await submit(
            client,
            model="realesr-general-x4v3",
            scale=4,
            settings='{"sharpenStrength": 0.3, "denoiseStrength": 0.8, "tileSize": 128}',
        )
    ).json()
    await wait_for_status(client, created["jobId"])

    request = app_context["enhancement"].calls[-1]
    assert request.sharpen_strength == 0.3
    assert request.denoise_strength == 0.8
    assert request.tile_size == 128


# ------------------------------------------------------------------- record


async def test_an_unknown_job_is_a_problem_document(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].get("/api/jobs/" + "0" * 32)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "job_not_found"


# -------------------------------------------------------------------- events


async def test_the_event_stream_reports_measured_progress(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    app_context["enhancement"].delay = 0.05
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    events: list[tuple[str, dict[str, Any]]] = []
    async with client.stream("GET", f"/api/jobs/{created['jobId']}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        name = ""
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
                if name in {"completed", "failed", "cancelled"}:
                    break

    names = [name for name, _ in events]
    assert names[-1] == "completed"
    assert "progress" in names

    tile_reports = [data for name, data in events if name == "progress"]
    assert tile_reports[0]["tilesTotal"] == tile_reports[0]["tilesTotal"]
    assert [report["progress"] for report in tile_reports] == sorted(
        report["progress"] for report in tile_reports
    )
    # Inference progress stays inside its band.
    assert all(15 <= report["progress"] <= 90 for report in tile_reports)


async def test_streaming_a_finished_job_still_closes(app_context: dict[str, Any]) -> None:
    """A late subscriber must be told the outcome, not left hanging."""
    client = app_context["client"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(client, created["jobId"])

    names: list[str] = []
    async with client.stream("GET", f"/api/jobs/{created['jobId']}/events") as response:
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                names.append(line.split(":", 1)[1].strip())
                if names[-1] in {"completed", "failed", "cancelled"}:
                    break

    assert names == ["completed"]


async def test_streaming_an_unknown_job_is_a_problem_document(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].get(f"/api/jobs/{'0' * 32}/events")

    assert response.status_code == 404


# -------------------------------------------------------------------- result


async def test_the_result_is_downloadable(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(client, created["jobId"])

    response = await client.get(f"/api/jobs/{created['jobId']}/result")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "attachment" in response.headers["content-disposition"]
    assert "256x192" in response.headers["content-disposition"]

    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (256, 192)


async def test_the_download_name_never_echoes_the_uploaded_filename(
    app_context: dict[str, Any],
) -> None:
    """The uploaded name is client-supplied text; it does not go in a header."""
    client = app_context["client"]
    response = await client.post(
        "/api/jobs",
        files={"image": ("../../etc/passwd.png", png_bytes(), "image/png")},
        data={"model": "realesr-general-x4v3", "scale": "4"},
    )
    created = response.json()
    await wait_for_status(client, created["jobId"])

    result = await client.get(f"/api/jobs/{created['jobId']}/result")

    disposition = result.headers["content-disposition"]
    assert "passwd" not in disposition
    assert ".." not in disposition


async def test_the_result_of_an_unfinished_job_is_a_conflict(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    app_context["enhancement"].delay = 0.2
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    response = await client.get(f"/api/jobs/{created['jobId']}/result")

    assert response.status_code == 409
    assert response.json()["code"] == "job_not_completed"
    await wait_for_status(client, created["jobId"])


async def test_a_range_request_returns_part_of_the_file(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(client, created["jobId"])

    response = await client.get(
        f"/api/jobs/{created['jobId']}/result", headers={"Range": "bytes=0-99"}
    )

    assert response.status_code == 206
    assert len(response.content) == 100


# ------------------------------------------------------------------- preview


async def completed_job(app_context: dict[str, Any]) -> str:
    """Submit and wait, for the tests that need a finished result."""
    client = app_context["client"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(client, created["jobId"])
    return str(created["jobId"])


async def test_the_preview_is_a_capped_jpeg(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    job_id = await completed_job(app_context)

    response = await client.get(f"/api/jobs/{job_id}/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "JPEG"
        # This result is well inside the cap, so it is served at its own size.
        assert image.size == (256, 192)


async def test_the_preview_is_cached_and_reused(app_context: dict[str, Any]) -> None:
    """Re-encoding a large result on every request is the cost this avoids."""
    client = app_context["client"]
    settings = app_context["settings"]
    job_id = await completed_job(app_context)

    await client.get(f"/api/jobs/{job_id}/preview")
    cached = settings.previews_dir / f"{job_id}.jpg"
    assert cached.is_file()

    stamped = cached.stat().st_mtime_ns
    cached.write_bytes(cached.read_bytes())  # touch without changing content
    await client.get(f"/api/jobs/{job_id}/preview")

    # A second request does not rebuild the file it already has.
    assert cached.stat().st_mtime_ns >= stamped


async def test_a_crop_returns_the_requested_region(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    job_id = await completed_job(app_context)

    response = await client.get(
        f"/api/jobs/{job_id}/preview", params={"x": 10, "y": 20, "w": 64, "h": 48}
    )

    assert response.status_code == 200
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (64, 48)


@pytest.mark.parametrize(
    "params",
    [
        {"x": 0, "y": 0, "w": 0, "h": 10},
        {"x": 0, "y": 0, "w": 10, "h": 0},
        {"x": -5, "y": 0, "w": 10, "h": 10},
        {"x": 9000, "y": 0, "w": 10, "h": 10},
        {"x": 0, "y": 0, "w": 9000, "h": 10},
    ],
)
async def test_an_invalid_crop_is_refused(
    app_context: dict[str, Any], params: dict[str, int]
) -> None:
    client = app_context["client"]
    job_id = await completed_job(app_context)

    response = await client.get(f"/api/jobs/{job_id}/preview", params=params)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_parameters"


async def test_a_partial_crop_is_refused(app_context: dict[str, Any]) -> None:
    """Three of four parameters is a mistake, not a request to guess."""
    client = app_context["client"]
    job_id = await completed_job(app_context)

    response = await client.get(f"/api/jobs/{job_id}/preview", params={"x": 0, "y": 0, "w": 10})

    assert response.status_code == 422
    assert "all four" in response.json()["detail"]


async def test_a_malformed_crop_value_is_refused(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    job_id = await completed_job(app_context)

    response = await client.get(
        f"/api/jobs/{job_id}/preview", params={"x": "left", "y": 0, "w": 10, "h": 10}
    )

    assert response.status_code == 422


async def test_a_preview_before_completion_is_a_conflict(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    app_context["enhancement"].delay = 0.2
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    response = await client.get(f"/api/jobs/{created['jobId']}/preview")

    assert response.status_code == 409
    assert response.json()["code"] == "job_not_completed"
    await wait_for_status(client, created["jobId"])


async def test_a_preview_for_an_unknown_job_is_a_404(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].get(f"/api/jobs/{'0' * 32}/preview")

    assert response.status_code == 404


async def test_a_preview_for_a_swept_result_is_a_404(app_context: dict[str, Any]) -> None:
    """The record can outlive the file; the endpoint must say so plainly."""
    client = app_context["client"]
    settings = app_context["settings"]
    job_id = await completed_job(app_context)

    for path in settings.outputs_dir.iterdir():
        path.unlink()

    response = await client.get(f"/api/jobs/{job_id}/preview")

    assert response.status_code == 404
    assert response.json()["code"] == "job_not_found"


async def test_deleting_a_job_removes_its_cached_preview(app_context: dict[str, Any]) -> None:
    client = app_context["client"]
    settings = app_context["settings"]
    job_id = await completed_job(app_context)
    await client.get(f"/api/jobs/{job_id}/preview")
    assert (settings.previews_dir / f"{job_id}.jpg").is_file()

    await client.delete(f"/api/jobs/{job_id}")

    assert list(settings.previews_dir.iterdir()) == []


# ------------------------------------------------------------ cancel/delete


async def test_cancelling_a_running_job_stops_it_without_a_result(
    app_context: dict[str, Any],
) -> None:
    client = app_context["client"]
    app_context["enhancement"].delay = 0.3
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    job_id = created["jobId"]

    # Wait until it is genuinely running before asking it to stop.
    for _ in range(100):
        body = (await client.get(f"/api/jobs/{job_id}")).json()
        if body["status"] == "processing":
            break
        await asyncio.sleep(0.02)

    response = await client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 204

    final = await wait_for_status(client, job_id)
    assert final["status"] == "cancelled"
    assert final["output"] is None

    # No partial result is kept, and none can be downloaded.
    assert (await client.get(f"/api/jobs/{job_id}/result")).status_code == 409
    assert list(app_context["settings"].outputs_dir.iterdir()) == []


async def test_deleting_a_finished_job_removes_it_and_its_files(
    app_context: dict[str, Any],
) -> None:
    client = app_context["client"]
    settings = app_context["settings"]
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()
    await wait_for_status(client, created["jobId"])

    response = await client.delete(f"/api/jobs/{created['jobId']}")

    assert response.status_code == 204
    assert (await client.get(f"/api/jobs/{created['jobId']}")).status_code == 404
    assert list(settings.inputs_dir.iterdir()) == []
    assert list(settings.outputs_dir.iterdir()) == []


async def test_deleting_an_unknown_job_is_a_404(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].delete(f"/api/jobs/{'0' * 32}")

    assert response.status_code == 404


# ------------------------------------------------------------------ failure


async def test_a_failing_job_records_the_error(app_context: dict[str, Any]) -> None:
    from app.core.exceptions import InferenceError

    client = app_context["client"]
    app_context["enhancement"].fail_with = InferenceError(
        "The model failed while processing this image.", technical="stub failure"
    )
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    final = await wait_for_status(client, created["jobId"])

    assert final["status"] == "failed"
    assert final["error"]["code"] == "inference_failed"
    assert final["error"]["technical"] == "stub failure"
    assert final["output"] is None


async def test_an_unexpected_failure_still_reaches_a_terminal_state(
    app_context: dict[str, Any],
) -> None:
    """A bug in the engine must not leave a job stuck at "processing" forever."""
    client = app_context["client"]
    app_context["enhancement"].fail_with = RuntimeError("something unexpected")
    created = (await submit(client, model="realesr-general-x4v3", scale=4)).json()

    final = await wait_for_status(client, created["jobId"])

    assert final["status"] == "failed"
    assert final["error"]["code"] == "internal_error"
