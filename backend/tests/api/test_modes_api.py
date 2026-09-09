"""Modes and target resolutions, end to end through the ASGI app.

Two things are being checked here. That the new fields do what they say - a
mode picks a model, a target picks a size - and, at least as important, that a
request written before either existed still behaves exactly as it did.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.mode_planner import CREATIVE_DENOISE, CREATIVE_MODEL, STANDARD_MODEL
from tests.api.conftest import png_bytes, submit, wait_for_status

pytestmark = pytest.mark.anyio


async def created_job(client: Any, **fields: Any) -> dict[str, Any]:
    response = await submit(client, **fields)
    assert response.status_code == 202, response.text
    body: dict[str, Any] = await wait_for_status(client, response.json()["jobId"])
    return body


# ----------------------------------------------------------------- the modes


async def test_standard_selects_the_model_that_already_shipped(
    app_context: dict[str, Any],
) -> None:
    job = await created_job(app_context["client"], mode="standard", scale=4)

    assert job["status"] == "completed"
    assert job["model"] == STANDARD_MODEL


async def test_creative_selects_the_detail_first_model(app_context: dict[str, Any]) -> None:
    job = await created_job(app_context["client"], mode="creative", scale=4)

    assert job["status"] == "completed"
    assert job["model"] == CREATIVE_MODEL


async def test_an_explicit_model_overrides_the_mode(app_context: dict[str, Any]) -> None:
    """A mode supplies defaults; it must not overrule what was asked for."""
    job = await created_job(app_context["client"], mode="creative", model=STANDARD_MODEL, scale=4)

    assert job["model"] == STANDARD_MODEL


async def test_an_unknown_mode_is_refused(app_context: dict[str, Any]) -> None:
    response = await submit(app_context["client"], mode="cinematic", scale=4)

    assert response.status_code == 422


# ------------------------------------------------------- target resolutions


@pytest.mark.parametrize(
    ("preset", "long_edge"),
    [("2k", 1920), ("4k", 3840)],
)
async def test_a_target_produces_exactly_that_long_edge(
    app_context: dict[str, Any], preset: str, long_edge: int
) -> None:
    """600x450 reaches 2K with a 4x pass and 4K with an 8x one, in both cases
    overshooting and then being resampled down to land exactly."""
    job = await created_job(app_context["client"], target=preset, content=png_bytes(600, 450))

    assert job["status"] == "completed"
    output = job["output"]
    assert max(output["width"], output["height"]) == long_edge


async def test_a_target_preserves_the_aspect_ratio(app_context: dict[str, Any]) -> None:
    job = await created_job(app_context["client"], target="2k", content=png_bytes(600, 450))

    output = job["output"]
    assert output["width"] / output["height"] == pytest.approx(600 / 450, rel=1e-2)


async def test_a_target_and_a_scale_together_are_refused(app_context: dict[str, Any]) -> None:
    """They answer the same question two different ways, so one must be wrong."""
    response = await submit(app_context["client"], target="4k", scale=4)

    assert response.status_code == 422
    assert "target resolution" in response.json()["detail"]


async def test_a_target_smaller_than_the_source_is_refused(
    app_context: dict[str, Any],
) -> None:
    """Enhancement cannot make an image smaller, and refusing says so."""
    response = await submit(app_context["client"], target="2k", content=png_bytes(2400, 1600))

    assert response.status_code == 422
    assert "already" in response.json()["detail"]


async def test_a_target_out_of_reach_is_refused_before_any_work(
    app_context: dict[str, Any],
) -> None:
    """64 px would need 120x for 8K; the largest factor composable is 16x."""
    response = await submit(app_context["client"], target="8k")

    assert response.status_code == 422
    body = response.json()
    assert "largest factor" in body["detail"]
    assert body["context"]["maximumScale"] == 16
    # 64 x 16, so the refusal says what this image *can* reach rather than
    # only what it cannot.
    assert body["context"]["reachableLongEdge"] == 1024


async def test_an_unknown_target_is_refused(app_context: dict[str, Any]) -> None:
    """32k is not a preset. Deliberately not 16k, which is one - naming a real
    preset here would pass on the source being too small to reach it and stop
    testing the enum at all."""
    response = await submit(app_context["client"], target="32k")

    assert response.status_code == 422


# ---------------------------------------------------- backward compatibility


async def test_a_request_with_neither_mode_nor_target_is_unchanged(
    app_context: dict[str, Any],
) -> None:
    """The exact shape every existing client sends."""
    job = await created_job(app_context["client"], model="realesr-general-x4v3", scale=4)

    assert job["status"] == "completed"
    assert job["model"] == "realesr-general-x4v3"
    assert job["scale"] == 4
    assert job["output"]["width"] == 64 * 4


async def test_a_bare_submission_still_uses_the_configured_defaults(
    app_context: dict[str, Any],
) -> None:
    """No model, no scale, no mode, no target - the oldest possible request."""
    job = await created_job(app_context["client"])

    assert job["status"] == "completed"
    assert job["scale"] == 4


async def test_the_existing_settings_object_still_applies(
    app_context: dict[str, Any],
) -> None:
    job = await created_job(
        app_context["client"],
        model="realesr-general-x4v3",
        scale=4,
        settings='{"sharpenStrength": 0.5, "denoiseStrength": 0.3}',
    )

    assert job["status"] == "completed"


async def test_an_explicit_denoise_survives_a_mode(app_context: dict[str, Any]) -> None:
    """Creative supplies 0.25 only when nothing was asked for."""
    job = await created_job(
        app_context["client"],
        mode="creative",
        scale=4,
        settings='{"denoiseStrength": 0.8}',
    )

    assert job["status"] == "completed"
    assert job["model"] == CREATIVE_MODEL


async def test_creatives_denoise_is_not_the_global_default(
    app_context: dict[str, Any],
) -> None:
    """The guard against this phase moving DEFAULT_DENOISE by the back door."""
    assert CREATIVE_DENOISE == 0.25


async def test_a_mode_combines_with_a_target(app_context: dict[str, Any]) -> None:
    job = await created_job(
        app_context["client"], mode="creative", target="2k", content=png_bytes(600, 450)
    )

    assert job["status"] == "completed"
    assert job["model"] == CREATIVE_MODEL
    assert max(job["output"]["width"], job["output"]["height"]) == 1920


# ------------------------------------------- how a finished job describes itself


async def test_a_standard_job_reports_its_mode(app_context: dict[str, Any]) -> None:
    job = await created_job(app_context["client"], mode="standard", scale=4)

    assert job["mode"] == "standard"
    assert job["outputType"] == "scale"
    assert job["target"] is None


async def test_a_creative_job_reports_its_mode(app_context: dict[str, Any]) -> None:
    job = await created_job(app_context["client"], mode="creative", scale=4)

    assert job["mode"] == "creative"
    assert job["outputType"] == "scale"


async def test_a_target_job_reports_the_preset_it_was_asked_for(
    app_context: dict[str, Any],
) -> None:
    """The preset, not only the pixels it resolved to - "4K" is what was asked."""
    job = await created_job(
        app_context["client"], mode="creative", target="2k", content=png_bytes(600, 450)
    )

    assert job["outputType"] == "target"
    assert job["target"] == "2k"
    assert job["mode"] == "creative"


async def test_a_job_with_no_mode_reports_none_rather_than_guessing(
    app_context: dict[str, Any],
) -> None:
    """The shape of every job stored before this phase existed."""
    job = await created_job(app_context["client"], model="realesr-general-x4v3", scale=4)

    assert job["mode"] is None
    # It was a scale job, because that was the only way to ask.
    assert job["outputType"] == "scale"
    assert job["target"] is None


async def test_history_carries_the_same_description(app_context: dict[str, Any]) -> None:
    """The list endpoint and the detail endpoint must not disagree."""
    client = app_context["client"]
    await created_job(client, mode="creative", target="2k", content=png_bytes(600, 450))

    listing = (await client.get("/api/jobs")).json()
    entry = listing["items"][0]

    assert entry["mode"] == "creative"
    assert entry["outputType"] == "target"
    assert entry["target"] == "2k"
