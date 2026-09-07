"""The queue, the progress broker, and the progress arithmetic.

These are the pieces between HTTP and the GPU. They are tested without either:
the queue runs trivial handlers, and the broker moves plain events, so what is
under test is the coordination rather than the inference.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings
from app.core.exceptions import QueueFullError
from app.models.enums import JobStage
from app.workers.progress import ProgressBroker, ProgressEvent, progress_event, stage_event
from app.workers.queue import QUEUE_DEPTH_PER_WORKER, InProcessJobQueue
from app.workers.runner import TilePlan, predict_tiles


def make_settings(**overrides: object) -> Settings:
    return Settings(environment="test", **overrides)  # type: ignore[arg-type]


# ------------------------------------------------------------------- events


def test_a_progress_event_carries_the_measured_tile_counts() -> None:
    event = progress_event(52, JobStage.RUNNING_INFERENCE, tiles_done=1, tiles_total=2)

    assert event.name == "progress"
    assert event.data == {
        "progress": 52,
        "stage": "running_inference",
        "tilesDone": 1,
        "tilesTotal": 2,
    }


def test_tile_counts_are_omitted_when_there_are_none_to_report() -> None:
    """Outside inference there is no tile count, so none is invented."""
    event = progress_event(10, JobStage.PREPROCESSING)

    assert "tilesDone" not in event.data
    assert "tilesTotal" not in event.data


def test_a_stage_event_reports_only_its_own_boundary() -> None:
    event = stage_event(JobStage.ENCODING, 95)

    assert event.name == "stage"
    assert event.data == {"stage": "encoding", "progress": 95}


@pytest.mark.parametrize("name", ["completed", "failed", "cancelled"])
def test_terminal_events_are_recognised(name: str) -> None:
    assert ProgressEvent(name, {}).is_terminal is True


def test_progress_is_not_terminal() -> None:
    assert ProgressEvent("progress", {}).is_terminal is False


# ------------------------------------------------------------------- broker


async def test_a_subscriber_receives_published_events() -> None:
    broker = ProgressBroker()
    received: list[ProgressEvent] = []

    async def listen() -> None:
        async for event in broker.subscribe("job-1"):
            received.append(event)

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)

    await broker.publish("job-1", progress_event(50, JobStage.RUNNING_INFERENCE))
    await broker.publish("job-1", ProgressEvent("completed", {"jobId": "job-1"}))
    await asyncio.wait_for(task, timeout=2)

    assert [event.name for event in received] == ["progress", "completed"]


async def test_the_stream_closes_after_a_terminal_event() -> None:
    """Without this the browser would hold an EventSource open forever."""
    broker = ProgressBroker()

    async def listen() -> list[str]:
        return [event.name async for event in broker.subscribe("job-1")]

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await broker.publish("job-1", ProgressEvent("failed", {"code": "internal_error"}))

    assert await asyncio.wait_for(task, timeout=2) == ["failed"]


async def test_a_late_subscriber_is_told_the_current_state_immediately() -> None:
    """A reconnecting client must not sit on an empty bar until the next tile."""
    broker = ProgressBroker()
    await broker.publish("job-1", progress_event(75, JobStage.RUNNING_INFERENCE))

    stream = broker.subscribe("job-1")
    try:
        first = await asyncio.wait_for(anext(stream), timeout=2)
    finally:
        await stream.aclose()

    assert first.data["progress"] == 75


async def test_a_subscriber_joining_after_the_end_still_gets_the_outcome() -> None:
    broker = ProgressBroker()
    await broker.publish("job-1", ProgressEvent("completed", {"jobId": "job-1"}))

    names = [event.name async for event in broker.subscribe("job-1")]

    assert names == ["completed"]


async def test_several_subscribers_all_receive_the_same_events() -> None:
    broker = ProgressBroker()

    async def listen() -> list[str]:
        return [event.name async for event in broker.subscribe("job-1")]

    tasks = [asyncio.create_task(listen()) for _ in range(3)]
    await asyncio.sleep(0)
    await broker.publish("job-1", ProgressEvent("completed", {}))

    for task in tasks:
        assert await asyncio.wait_for(task, timeout=2) == ["completed"]


async def test_publishing_from_a_thread_reaches_the_subscribers() -> None:
    """The worker runs off-loop, so this is the path every tile update takes."""
    broker = ProgressBroker()
    broker.bind_loop(asyncio.get_running_loop())

    async def listen() -> list[str]:
        return [event.name async for event in broker.subscribe("job-1")]

    task = asyncio.create_task(listen())
    await asyncio.sleep(0)

    await asyncio.to_thread(broker.publish_threadsafe, "job-1", ProgressEvent("completed", {}))

    assert await asyncio.wait_for(task, timeout=2) == ["completed"]


async def test_finished_jobs_are_forgotten_once_nobody_is_listening() -> None:
    """Retaining every job's last event forever would be a slow leak."""
    broker = ProgressBroker()
    await broker.publish("job-1", ProgressEvent("completed", {}))

    # Draining the retained terminal event is what releases the topic.
    names = [event.name async for event in broker.subscribe("job-1")]

    assert names == ["completed"]
    assert broker.tracked_jobs == 0


async def test_forgetting_a_job_drops_its_state() -> None:
    broker = ProgressBroker()
    await broker.publish("job-1", progress_event(10, JobStage.VALIDATING))

    broker.forget("job-1")

    assert broker.tracked_jobs == 0


# -------------------------------------------------------------------- queue


async def test_submitting_runs_the_handler() -> None:
    handled: list[str] = []

    async def handler(job_id: str) -> None:
        handled.append(job_id)

    queue = InProcessJobQueue(make_settings(), handler)
    await queue.start()
    try:
        await queue.submit("job-1")
        await asyncio.sleep(0.1)
    finally:
        await queue.stop()

    assert handled == ["job-1"]


async def test_jobs_run_in_submission_order() -> None:
    handled: list[str] = []

    async def handler(job_id: str) -> None:
        handled.append(job_id)

    queue = InProcessJobQueue(make_settings(max_concurrent_jobs=1), handler)
    await queue.start()
    try:
        for index in range(4):
            await queue.submit(f"job-{index}")
        await asyncio.sleep(0.2)
    finally:
        await queue.stop()

    assert handled == ["job-0", "job-1", "job-2", "job-3"]


async def test_the_queue_position_reflects_the_backlog() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(job_id: str) -> None:
        started.set()
        await release.wait()

    queue = InProcessJobQueue(make_settings(max_concurrent_jobs=1), handler)
    await queue.start()
    try:
        await queue.submit("job-0")
        await asyncio.wait_for(started.wait(), timeout=2)
        second = await queue.submit("job-1")

        assert second == 1
        assert queue.position_of("job-1") == 1
    finally:
        release.set()
        await queue.stop()


async def test_a_full_queue_is_refused_rather_than_growing_without_bound() -> None:
    release = asyncio.Event()

    async def handler(job_id: str) -> None:
        await release.wait()

    queue = InProcessJobQueue(make_settings(max_concurrent_jobs=1), handler)
    await queue.start()
    try:
        for index in range(QUEUE_DEPTH_PER_WORKER):
            await queue.submit(f"job-{index}")

        with pytest.raises(QueueFullError, match="busy"):
            await queue.submit("one-too-many")
    finally:
        release.set()
        await queue.stop()


async def test_a_handler_that_raises_does_not_stop_the_queue() -> None:
    """One bad job must not take every later job down with it."""
    handled: list[str] = []

    async def handler(job_id: str) -> None:
        handled.append(job_id)
        if job_id == "bad":
            raise RuntimeError("boom")

    queue = InProcessJobQueue(make_settings(), handler)
    await queue.start()
    try:
        await queue.submit("bad")
        await queue.submit("good")
        await asyncio.sleep(0.2)
    finally:
        await queue.stop()

    assert handled == ["bad", "good"]


async def test_cancellation_is_visible_to_a_worker_thread() -> None:
    """The tiling loop reads this between tiles, off the event loop."""

    async def handler(job_id: str) -> None:
        return None

    queue = InProcessJobQueue(make_settings(), handler)

    queue.request_cancel("job-1")
    assert await asyncio.to_thread(queue.is_cancelled, "job-1") is True

    queue.clear_cancel("job-1")
    assert await asyncio.to_thread(queue.is_cancelled, "job-1") is False


async def test_a_finished_job_leaves_the_backlog() -> None:
    async def handler(job_id: str) -> None:
        return None

    queue = InProcessJobQueue(make_settings(), handler)
    await queue.start()
    try:
        await queue.submit("job-1")
        await asyncio.sleep(0.1)

        assert queue.depth == 0
        assert queue.position_of("job-1") is None
    finally:
        await queue.stop()


# ------------------------------------------------------- progress arithmetic


def test_a_single_pass_maps_fractions_onto_tiles() -> None:
    plan = TilePlan(per_pass=(4,))

    assert plan.total == 4
    assert plan.tiles_done(0.0) == 0
    assert plan.tiles_done(0.25) == 1
    assert plan.tiles_done(0.75) == 3
    assert plan.tiles_done(1.0) == 4


def test_a_two_pass_plan_counts_both_passes() -> None:
    """An 8x job runs 4x then 2x, and the second pass sees a larger image."""
    plan = TilePlan(per_pass=(2, 6))

    assert plan.total == 8
    assert plan.tiles_done(0.0) == 0
    assert plan.tiles_done(0.5) == 2  # first pass finished
    assert plan.tiles_done(0.75) == 5
    assert plan.tiles_done(1.0) == 8


def test_tile_counts_never_exceed_the_total() -> None:
    plan = TilePlan(per_pass=(3,))

    assert plan.tiles_done(1.5) == 3
    assert plan.tiles_done(-1.0) == 0


def test_prediction_uses_the_configured_tile_size() -> None:
    settings = make_settings(tile_size=256, tile_pad=16)

    plan = predict_tiles(
        320, 240, [4], settings=settings, tile_size=None, tile_pad=None, free_mb=None
    )

    assert plan.per_pass == (2,)  # a 2x1 grid across 320 px


def test_prediction_follows_a_per_job_tile_override() -> None:
    settings = make_settings(tile_size=256, tile_pad=16)

    plan = predict_tiles(320, 240, [4], settings=settings, tile_size=128, tile_pad=8, free_mb=None)

    assert plan.per_pass == (6,)  # 3x2


def test_prediction_grows_the_image_between_passes() -> None:
    """The second pass of an 8x job works on the 4x result, so it has more tiles."""
    settings = make_settings(tile_size=256, tile_pad=16)

    plan = predict_tiles(
        320, 240, [4, 2], settings=settings, tile_size=None, tile_pad=None, free_mb=None
    )

    assert plan.per_pass[0] == 2
    assert plan.per_pass[1] > plan.per_pass[0]


def test_prediction_respects_the_vram_budget() -> None:
    """A nearly full card gets smaller tiles, and therefore more of them."""
    settings = make_settings(tile_size=512, tile_pad=16)

    roomy = predict_tiles(
        1024, 1024, [4], settings=settings, tile_size=None, tile_pad=None, free_mb=11000
    )
    tight = predict_tiles(
        1024, 1024, [4], settings=settings, tile_size=None, tile_pad=None, free_mb=900
    )

    assert tight.total > roomy.total
