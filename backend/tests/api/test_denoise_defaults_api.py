"""Denoise default semantics, end to end and as persisted.

`test_mode_planner` covers the policy in isolation. This covers the thing that
actually shipped wrong: what a real submission resolves to, and what gets
written to the job so a finished job can say what it ran.

The bug being guarded against was one of omission rather than logic. A request
naming `realesr-general-x4v3` without a denoise value resolved to `None`, and
`None` at the model manager means "load the standard weights unblended" - which
for this model is DNI 1.00, the strongest denoising available. Phase 3B
confirmed the two are byte-identical, so the shipped default was the one
setting F2 and Phase 2.5 had both ruled out, chosen by nobody.

Every test here reads the stored `enhance_options` rather than the response
body, because that is the record the worker acts on and the only place the
resolved value is observable.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.runtime import repository_scope
from app.services.mode_planner import (
    CREATIVE_DENOISE,
    CREATIVE_MODEL,
    DEFAULT_DENOISE,
    STANDARD_MODEL,
)
from tests.api.conftest import submit, wait_for_status

pytestmark = pytest.mark.anyio


async def stored_options(job_id: str) -> dict[str, Any]:
    """The `enhance_options` JSON as the worker will read it."""
    async with repository_scope() as repository:
        job = await repository.get(job_id)
    assert job is not None, f"job {job_id} was not persisted"
    options: dict[str, Any] = job.enhance_options or {}
    return options


async def submitted(client: Any, **fields: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Submit, wait, and return the finished job alongside its stored options."""
    response = await submit(client, **fields)
    assert response.status_code == 202, response.text
    job_id = response.json()["jobId"]
    body: dict[str, Any] = await wait_for_status(client, job_id)
    return body, await stored_options(job_id)


# ------------------------------------------------- the default that was wrong


async def test_the_capable_model_without_a_mode_no_longer_lands_on_full_denoise(
    app_context: dict[str, Any],
) -> None:
    """The exact request that was broken.

    No mode, no settings, the denoise-capable model named directly - which is
    what the client sends when the slider has not been touched. This used to
    persist no denoise at all and run DNI 1.00.
    """
    job, options = await submitted(app_context["client"], model=CREATIVE_MODEL, scale=4)

    assert job["status"] == "completed"
    assert job["model"] == CREATIVE_MODEL
    assert options["denoiseStrength"] == DEFAULT_DENOISE
    assert options["denoiseStrength"] != 1.0


async def test_standard_mode_overridden_to_a_capable_model_also_gets_the_default(
    app_context: dict[str, Any],
) -> None:
    """The second path that fell through, which the audit had not named.

    Standard's own model has no denoise pair, so `plan_mode` returns `None` for
    it. Overriding the model to a capable one left that `None` looking like
    "unset" and it landed on DNI 1.00 the same way.
    """
    _, options = await submitted(
        app_context["client"], mode="standard", model=CREATIVE_MODEL, scale=4
    )

    assert options["denoiseStrength"] == DEFAULT_DENOISE


async def test_creative_still_supplies_its_own_value(app_context: dict[str, Any]) -> None:
    job, options = await submitted(app_context["client"], mode="creative", scale=4)

    assert job["model"] == CREATIVE_MODEL
    assert options["denoiseStrength"] == CREATIVE_DENOISE


# ---------------------------------------------------- explicit always survives


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
async def test_an_explicit_value_is_persisted_exactly(
    app_context: dict[str, Any], strength: float
) -> None:
    """Both ends included, and they are the two that matter.

    0.0 is a real setting - fully the `wdn` weights - and a truth test would
    read it as absent. 1.0 is the value the old default produced by accident
    and must stay reachable on purpose.
    """
    _, options = await submitted(
        app_context["client"],
        model=CREATIVE_MODEL,
        scale=4,
        settings=f'{{"denoiseStrength": {strength}}}',
    )

    assert options["denoiseStrength"] == strength


async def test_an_explicit_zero_is_not_swallowed_by_the_default(
    app_context: dict[str, Any],
) -> None:
    """Stated separately because it is the failure a careless fix produces.

    Writing the resolution as `if not denoise: use_default()` passes every
    other test in this file and silently rewrites 0.0 to 0.25.
    """
    _, options = await submitted(
        app_context["client"],
        mode="creative",
        model=CREATIVE_MODEL,
        scale=4,
        settings='{"denoiseStrength": 0.0}',
    )

    assert options["denoiseStrength"] == 0.0
    assert options["denoiseStrength"] != DEFAULT_DENOISE


async def test_an_explicit_one_still_reaches_full_denoise(
    app_context: dict[str, Any],
) -> None:
    """Nothing here removes 1.0 as a choice; it removes it as an accident."""
    _, options = await submitted(
        app_context["client"],
        model=CREATIVE_MODEL,
        scale=4,
        settings='{"denoiseStrength": 1.0}',
    )

    assert options["denoiseStrength"] == 1.0


# ------------------------------------------------------- the model without one


async def test_the_standard_model_persists_no_denoise_at_all(
    app_context: dict[str, Any],
) -> None:
    """`RealESRGAN_x4plus` declares no denoise pair, so no default applies.

    The capability tier returns before any default is considered, which is what
    keeps this change from touching Standard.
    """
    job, options = await submitted(app_context["client"], mode="standard", scale=4)

    assert job["model"] == STANDARD_MODEL
    assert "denoiseStrength" not in options


async def test_a_bare_submission_is_unchanged(app_context: dict[str, Any]) -> None:
    """No mode, no model, no settings - the oldest shape of request there is.

    It resolves to the configured default model, which has no denoise pair, so
    it carries no denoise. Backward compatibility for the majority path.
    """
    _, options = await submitted(app_context["client"], scale=4)

    assert "denoiseStrength" not in options


async def test_a_denoise_value_for_an_incapable_model_is_still_refused(
    app_context: dict[str, Any],
) -> None:
    """Explicit is checked before capability, so this reaches the validator.

    Dropping it silently would be worse: the user asked for something the model
    cannot do and deserves to be told.
    """
    response = await submit(
        app_context["client"],
        model=STANDARD_MODEL,
        scale=4,
        settings='{"denoiseStrength": 0.5}',
    )

    assert response.status_code == 422, response.text
