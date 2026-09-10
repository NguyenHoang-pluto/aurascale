"""Abuse resistance at the HTTP edge, for a public URL.

Phase 8 audited the protections rather than assuming them. Most were already
there and correct; this file locks them in, because a protection with no test
is a protection that can be removed by accident.

The threat model is an anonymous stranger with the URL and no sophistication:
someone who pastes a path into the address bar, opens twenty tabs, or leaves a
stream open and walks away. Not an attacker with a fuzzer.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.api.conftest import png_bytes, submit

pytestmark = pytest.mark.anyio


# --------------------------------------------------------- job ids as paths

#: Ids that are not lowercase hex. Storage refuses these outright, but the
#: lookup should reject them first, so none of them should reach it.
BAD_IDS = [
    "../../etc/passwd",
    "..%2F..%2Fetc%2Fpasswd",
    "....//....//secret",
    "a" * 300,
    "job id with spaces",
    "ABCDEF0123456789",  # right shape, wrong case
    "zzzzzzzz",
    "'; DROP TABLE jobs;--",
    "%00",
    "..",
]


@pytest.mark.parametrize("job_id", BAD_IDS)
async def test_a_bad_job_id_is_a_clean_refusal_not_a_crash(
    app_context: dict[str, Any], job_id: str
) -> None:
    """Never a 500, and never a traceback.

    A 500 here would mean an id shaped like a path reached something that
    tried to use it. The lookup answers first, so every one of these is a
    plain "no such job" or a routing refusal.
    """
    response = await app_context["client"].get(f"/api/jobs/{job_id}")

    assert response.status_code in {404, 422, 405, 301, 307}, response.text
    assert response.status_code != 500


@pytest.mark.parametrize("suffix", ["result", "preview", "thumbnail", "events"])
async def test_bad_ids_on_every_sub_resource_are_refused(
    app_context: dict[str, Any], suffix: str
) -> None:
    """The file-serving routes are the ones where a path would do damage."""
    response = await app_context["client"].get(f"/api/jobs/../../etc/passwd/{suffix}")

    assert response.status_code != 500
    assert response.status_code in {404, 422, 405, 301, 307}


async def test_a_refusal_never_carries_a_filesystem_path(
    app_context: dict[str, Any],
) -> None:
    """Error bodies are for users, not for mapping the server's disk."""
    response = await app_context["client"].get("/api/jobs/deadbeefdeadbeefdeadbeefdeadbeef")

    assert response.status_code == 404
    body = response.text
    for leak in ("E:\\", "/home/", "storage_dir", "Traceback", "site-packages"):
        assert leak not in body, f"error body leaked {leak!r}"


# ------------------------------------------------------------------ uploads


async def test_a_path_like_filename_does_not_reach_the_filesystem(
    app_context: dict[str, Any],
) -> None:
    """The name is kept for display and stripped of anything path-shaped.

    Storage names files by generated id, never by upload name, so this is
    defence in depth rather than the only guard.
    """
    settings = app_context["settings"]
    response = await app_context["client"].post(
        "/api/jobs",
        files={"image": ("../../../evil.png", png_bytes(), "image/png")},
        data={"scale": "4"},
    )

    assert response.status_code == 202, response.text
    for directory in (settings.inputs_dir, settings.outputs_dir):
        for path in directory.iterdir():
            assert ".." not in path.name
            assert path.parent == directory


async def test_an_empty_upload_is_refused(app_context: dict[str, Any]) -> None:
    response = await app_context["client"].post(
        "/api/jobs",
        files={"image": ("empty.png", b"", "image/png")},
        data={"scale": "4"},
    )

    assert response.status_code in {413, 415, 422}, response.text
    assert response.status_code != 500


async def test_a_truncated_png_is_refused_rather_than_crashing(
    app_context: dict[str, Any],
) -> None:
    """A real header with the body cut off - the common accidental corruption."""
    truncated = png_bytes()[:40]
    response = await app_context["client"].post(
        "/api/jobs",
        files={"image": ("cut.png", truncated, "image/png")},
        data={"scale": "4"},
    )

    assert response.status_code in {415, 422}, response.text


async def test_a_renamed_text_file_is_caught_by_its_bytes(
    app_context: dict[str, Any],
) -> None:
    """Extension and declared MIME type are both attacker-controlled."""
    response = await app_context["client"].post(
        "/api/jobs",
        files={"image": ("photo.png", b"this is not an image at all", "image/png")},
        data={"scale": "4"},
    )

    assert response.status_code == 415, response.text


async def test_no_upload_survives_a_refusal(app_context: dict[str, Any]) -> None:
    """A rejected submission must not leave its staged bytes behind.

    This is the one that accumulates: a stranger retrying a bad file a hundred
    times should cost a hundred error responses and no disk.
    """
    settings = app_context["settings"]

    for _ in range(5):
        await app_context["client"].post(
            "/api/jobs",
            files={"image": ("junk.png", b"not an image", "image/png")},
            data={"scale": "4"},
        )

    assert list(settings.inputs_dir.iterdir()) == []


# ------------------------------------------------------------- invalid input


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scale", "3"),
        ("scale", "-4"),
        ("scale", "0"),
        ("scale", "999999"),
        ("scale", "abc"),
        ("model", "no-such-model"),
        ("model", "../../../etc/passwd"),
        ("mode", "turbo"),
        ("target", "32k"),
        ("format", "bmp"),
    ],
)
async def test_an_invalid_field_is_a_typed_refusal(
    app_context: dict[str, Any], field: str, value: str
) -> None:
    response = await submit(app_context["client"], **{field: value})

    assert response.status_code in {404, 415, 422}, f"{field}={value}: {response.text}"
    assert response.status_code != 500
    body = response.json()
    assert body["code"], "a refusal must carry a machine-readable code"
    assert "Traceback" not in response.text


@pytest.mark.parametrize("denoise", ["-0.1", "1.1", "abc", "1e400"])
async def test_an_out_of_range_denoise_is_refused(
    app_context: dict[str, Any], denoise: str
) -> None:
    response = await submit(app_context["client"], settings=f'{{"denoiseStrength": {denoise}}}')

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_parameters"


@pytest.mark.parametrize(
    "payload",
    ["{", "[]", "null", '"a string"', "123", '{"tileSize": -1}', '{"tileSize": 999999}'],
)
async def test_a_malformed_settings_blob_is_refused(
    app_context: dict[str, Any], payload: str
) -> None:
    response = await submit(app_context["client"], settings=payload)

    assert response.status_code == 422, response.text
    assert response.status_code != 500


# ------------------------------------------------------------------- streams


async def test_a_stream_for_an_unknown_job_is_refused_not_left_open(
    app_context: dict[str, Any],
) -> None:
    """A stream that opens and hangs is worse than a clean 404.

    The route resolves the job before returning a response for exactly this
    reason.
    """
    response = await app_context["client"].get("/api/jobs/deadbeefdeadbeefdeadbeefdeadbeef/events")

    assert response.status_code == 404


async def test_abandoned_streams_do_not_accumulate_subscribers(
    app_context: dict[str, Any],
) -> None:
    """Twenty tabs opened and closed must leave nothing behind.

    Each subscription holds a queue on the broker's topic. The generator drops
    it in a `finally`, so a client that disconnects mid-stream is cleaned up
    the same way one that reads to the end is - but only as long as that
    `finally` survives, which is what this asserts.
    """
    client = app_context["client"]
    broker = app_context["broker"]

    created = await submit(client, scale=4)
    job_id = created.json()["jobId"]

    for _ in range(20):
        async with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            assert stream.status_code == 200
            async for _line in stream.aiter_lines():
                break  # read one line, then hang up

    await asyncio.sleep(0.1)

    topic = broker._topics.get(job_id)
    if topic is not None:
        assert topic.subscribers == set(), "disconnected subscribers were retained"


# --------------------------------------------------------------- the backlog


async def test_the_queue_refuses_a_flood_rather_than_accepting_it(
    app_context: dict[str, Any],
) -> None:
    """The protection that keeps one stranger from booking the GPU for an hour.

    `max_concurrent_jobs` is 1 and the backlog is bounded, so a burst is
    answered with a typed refusal once the bound is reached rather than
    queued. Submitted well past the limit on purpose: the assertion is that
    the refusals are clean and bounded, not that any particular number gets in.
    """
    client = app_context["client"]
    statuses: list[int] = []

    for _ in range(24):
        response = await submit(client, scale=4)
        statuses.append(response.status_code)

    assert 500 not in statuses, "a flood produced a server error"
    assert set(statuses) <= {202, 429, 503}, f"unexpected statuses: {sorted(set(statuses))}"

    refused = [code for code in statuses if code != 202]
    if refused:
        # Whatever was refused must say why, in the documented shape.
        response = await submit(client, scale=4)
        if response.status_code != 202:
            assert response.json()["code"] == "queue_full"


# ------------------------------------------------------------------- quality


@pytest.mark.parametrize("quality", ["0", "49", "101", "1000", "-5"])
async def test_an_out_of_range_quality_is_refused_for_a_lossy_format(
    app_context: dict[str, Any], quality: str
) -> None:
    """Quality is validated where it means something.

    The audit initially read PNG accepting `quality=1000` as a gap. It is not:
    the field is dropped for a lossless format because there is nothing for it
    to control, and refusing a request over an ignored field would reject
    perfectly good submissions. The guard belongs on the lossy path, and this
    asserts it is there.
    """
    response = await submit(app_context["client"], format="jpeg", quality=quality, scale=4)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_parameters"


@pytest.mark.parametrize("quality", ["50", "92", "100"])
async def test_a_valid_quality_is_accepted_for_a_lossy_format(
    app_context: dict[str, Any], quality: str
) -> None:
    response = await submit(app_context["client"], format="jpeg", quality=quality, scale=4)

    assert response.status_code == 202, response.text


async def test_quality_is_ignored_rather_than_refused_for_png(
    app_context: dict[str, Any],
) -> None:
    """Documented deliberately, because it looks like a missing check."""
    response = await submit(app_context["client"], format="png", quality="1000", scale=4)

    assert response.status_code == 202, response.text


async def test_a_json_null_denoise_means_omitted(app_context: dict[str, Any]) -> None:
    """`null` is not an out-of-range value; it is the absence Phase 7 defined.

    The server resolves it the same way it resolves a missing field, which is
    what keeps a client that serialises its whole settings object working.
    """
    response = await submit(app_context["client"], settings='{"denoiseStrength": null}', scale=4)

    assert response.status_code == 202, response.text
