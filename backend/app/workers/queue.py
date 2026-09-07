"""The job queue and its worker pool.

One `asyncio.Queue` feeding a `ThreadPoolExecutor` sized at
`MAX_CONCURRENT_JOBS` (docs/architecture.md §3, §4). PyTorch inference is
blocking native code: running it in a coroutine would freeze SSE streams and
health checks for the duration of a job, so it goes to a thread, where the GIL
is released inside PyTorch's kernels and the loop genuinely keeps running.

A pool of one also serialises GPU work for free, with no extra locking - which
is what a single GPU wants anyway.

`JobQueue` is an interface so that swapping to Celery later means writing one
implementation rather than touching the services above it.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

from app.core.config import Settings
from app.core.exceptions import QueueFullError
from app.core.logging import get_logger

logger = get_logger(__name__)

JobHandler = Callable[[str], Awaitable[None]]

# Each worker may have this many jobs waiting behind it before submissions are
# refused. Small on purpose: a queue that accepts hundreds of jobs it cannot
# start for an hour is worse than one that says no.
QUEUE_DEPTH_PER_WORKER = 8


class JobQueue(ABC):
    """Somewhere to put a job id and have it processed."""

    @abstractmethod
    async def submit(self, job_id: str) -> int:
        """Enqueue a job. Returns its position in the queue."""

    @abstractmethod
    def position_of(self, job_id: str) -> int | None:
        """Where a job sits in the queue, or None once it has been taken up."""

    @abstractmethod
    def request_cancel(self, job_id: str) -> None:
        """Ask a job to stop. Cooperative: it stops at the next checkpoint."""

    @abstractmethod
    def is_cancelled(self, job_id: str) -> bool:
        """Whether cancellation has been requested. Safe to call from a thread."""

    @abstractmethod
    def clear_cancel(self, job_id: str) -> None:
        """Forget a cancellation request once the job has finished."""


class InProcessJobQueue(JobQueue):
    """asyncio.Queue plus a thread pool, inside the API process."""

    def __init__(self, settings: Settings, handler: JobHandler) -> None:
        self._settings = settings
        self._handler = handler
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending: list[str] = []
        self._consumers: list[asyncio.Task[None]] = []
        self._executor: ThreadPoolExecutor | None = None
        # A plain set behind a lock rather than an asyncio primitive: this is
        # read from the worker thread between tiles, where awaiting is not an
        # option.
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the consumers and the pool that runs inference."""
        if self._running:  # pragma: no cover - defensive
            return

        workers = self._settings.max_concurrent_jobs
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="pixelforge-inference"
        )
        self._consumers = [
            asyncio.create_task(self._consume(index), name=f"job-consumer-{index}")
            for index in range(workers)
        ]
        self._running = True
        logger.info("job queue started", extra={"workers": workers})

    async def stop(self) -> None:
        """Stop accepting work and let anything in flight wind down.

        Running jobs are asked to cancel rather than killed: a half-written
        output file is worse than a job that took a few more seconds to stop.
        """
        if not self._running:
            return
        self._running = False

        with self._cancel_lock:
            self._cancelled.update(self._pending)

        for task in self._consumers:
            task.cancel()

        await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers.clear()

        if self._executor is not None:
            # wait=True so a job in a thread finishes its current tile and
            # closes its files before the process exits.
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        logger.info("job queue stopped")

    @property
    def executor(self) -> ThreadPoolExecutor:
        """The pool inference runs on."""
        if self._executor is None:  # pragma: no cover - programming error
            raise RuntimeError("the job queue has not been started")
        return self._executor

    # ---------------------------------------------------------------- submit

    async def submit(self, job_id: str) -> int:
        """Enqueue a job, refusing once the backlog is at its bound."""
        if not self._running:  # pragma: no cover - defensive
            raise QueueFullError(
                "The server is not accepting jobs at the moment.",
                technical="queue not running",
            )

        limit = self._settings.max_concurrent_jobs * QUEUE_DEPTH_PER_WORKER
        if len(self._pending) >= limit:
            raise QueueFullError(
                "The server is busy with other images. Try again in a moment.",
                technical=f"queued={len(self._pending)} limit={limit}",
                context={"queueLength": len(self._pending)},
            )

        position = len(self._pending)
        self._pending.append(job_id)
        await self._queue.put(job_id)
        logger.info("job queued", extra={"job_id": job_id, "position": position})
        return position

    def position_of(self, job_id: str) -> int | None:
        try:
            return self._pending.index(job_id)
        except ValueError:
            return None

    @property
    def depth(self) -> int:
        """Jobs waiting or in flight."""
        return len(self._pending)

    # --------------------------------------------------------- cancellation

    def request_cancel(self, job_id: str) -> None:
        with self._cancel_lock:
            self._cancelled.add(job_id)
        logger.info("cancellation requested", extra={"job_id": job_id})

    def is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled

    def clear_cancel(self, job_id: str) -> None:
        with self._cancel_lock:
            self._cancelled.discard(job_id)

    # -------------------------------------------------------------- consumer

    async def _consume(self, index: int) -> None:
        """Take jobs one at a time and run them to completion."""
        while True:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return

            try:
                await self._handler(job_id)
            except asyncio.CancelledError:
                logger.info("consumer cancelled mid-job", extra={"job_id": job_id})
                raise
            except Exception:
                # The handler is responsible for recording failures; reaching
                # here means it failed to, and the consumer must survive it or
                # the queue stops processing anything else.
                logger.exception("job handler raised", extra={"job_id": job_id})
            finally:
                if job_id in self._pending:
                    self._pending.remove(job_id)
                self.clear_cancel(job_id)
                self._queue.task_done()
