"""Progress publication, from the worker thread to SSE subscribers.

Inference runs on a worker thread; SSE subscribers live on the event loop. The
only supported way across that boundary is
`asyncio.run_coroutine_threadsafe`, which is what `publish_threadsafe` does.

Every event here describes something that actually happened. Nothing in this
module invents a value, interpolates between two real measurements, or advances
a bar on a timer — a progress report that is not a measurement is a lie the
user cannot check.

The last event per job is retained so a subscriber that connects late, or
reconnects after a dropped stream, is told the current state immediately rather
than waiting for the next tile to finish.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.models.enums import JobStage

logger = get_logger(__name__)

# How many events a slow subscriber may fall behind before the oldest are
# dropped. Progress events are snapshots, so dropping stale ones loses nothing;
# blocking the worker thread on a slow HTTP client would lose a lot.
SUBSCRIBER_QUEUE_SIZE = 32


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One SSE event. `name` is the event type the browser dispatches on."""

    name: str
    data: dict[str, Any]

    @property
    def is_terminal(self) -> bool:
        """Whether the stream should close after this event."""
        return self.name in {"completed", "failed", "cancelled"}


def progress_event(
    percent: int,
    stage: JobStage,
    *,
    tiles_done: int | None = None,
    tiles_total: int | None = None,
) -> ProgressEvent:
    """A measured progress report.

    `tilesDone`/`tilesTotal` are included only while there is a real tile count
    to report; outside inference they are absent rather than guessed.
    """
    data: dict[str, Any] = {"progress": percent, "stage": stage.value}

    if tiles_done is not None and tiles_total is not None:
        data["tilesDone"] = tiles_done
        data["tilesTotal"] = tiles_total

    return ProgressEvent("progress", data)


def stage_event(stage: JobStage, percent: int) -> ProgressEvent:
    """A stage boundary - the honest report for work that cannot be subdivided."""
    return ProgressEvent("stage", {"stage": stage.value, "progress": percent})


@dataclass
class _Topic:
    """Subscribers for one job, plus the most recent event."""

    subscribers: set[asyncio.Queue[ProgressEvent]] = field(default_factory=set)
    latest: ProgressEvent | None = None
    finished: bool = False


class ProgressBroker:
    """Fan-out of job events to any number of SSE streams."""

    def __init__(self) -> None:
        self._topics: dict[str, _Topic] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the loop the worker threads will publish into."""
        self._loop = loop

    # ---------------------------------------------------------------- publish

    async def publish(self, job_id: str, event: ProgressEvent) -> None:
        """Deliver an event to every current subscriber."""
        topic = self._topics.setdefault(job_id, _Topic())
        topic.latest = event
        if event.is_terminal:
            topic.finished = True

        for queue in list(topic.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest and retry: the newest snapshot is the one
                # worth having.
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                    logger.debug("dropped a progress event for a slow subscriber")

    def publish_threadsafe(self, job_id: str, event: ProgressEvent) -> None:
        """Publish from a worker thread.

        Deliberately fire-and-forget: the worker must not block on subscriber
        delivery, because a stalled HTTP client would then stall inference.
        """
        loop = self._loop
        if loop is None or loop.is_closed():  # pragma: no cover - shutdown race
            return

        try:
            asyncio.run_coroutine_threadsafe(self.publish(job_id, event), loop)
        except RuntimeError:  # pragma: no cover - loop closed mid-publish
            logger.debug("could not publish progress; the loop is gone")

    # -------------------------------------------------------------- subscribe

    async def subscribe(self, job_id: str) -> AsyncIterator[ProgressEvent]:
        """Yield events for a job until a terminal one arrives.

        The most recent event is replayed first, so a client that connects
        after work has already started is not left with an empty bar.
        """
        topic = self._topics.setdefault(job_id, _Topic())
        queue: asyncio.Queue[ProgressEvent] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        topic.subscribers.add(queue)

        try:
            if topic.latest is not None:
                yield topic.latest
                if topic.latest.is_terminal:
                    return

            while True:
                event = await queue.get()
                yield event
                if event.is_terminal:
                    return
        finally:
            topic.subscribers.discard(queue)
            self._discard_if_done(job_id)

    def _discard_if_done(self, job_id: str) -> None:
        """Forget a finished job once nobody is listening.

        The retained event exists to catch late subscribers; keeping it for
        every job ever run would be a slow memory leak.
        """
        topic = self._topics.get(job_id)
        if topic is not None and topic.finished and not topic.subscribers:
            del self._topics[job_id]

    def forget(self, job_id: str) -> None:
        """Drop a job's topic outright, e.g. when its record is deleted."""
        self._topics.pop(job_id, None)

    @property
    def tracked_jobs(self) -> int:
        """How many jobs have retained state. Used by tests and diagnostics."""
        return len(self._topics)
