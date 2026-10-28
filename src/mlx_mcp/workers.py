"""Lazily built shared resources and a serialized worker thread for blocking calls."""

import asyncio
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from threading import Event, Lock
from typing import cast

# A queued call: its result future and the bound call.
type _Job = tuple[Future[object], Callable[[], object]]


class LazyResource[T]:
    """Construct once even on concurrent cold calls; never cache failed attempts."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._lock = Lock()
        self._value: tuple[T] | None = None

    def get(self) -> T:
        """Return the resource, constructing it if no attempt has succeeded yet."""
        with self._lock:
            if self._value is None:
                self._value = (self._factory(),)
            return self._value[0]

    def peek(self) -> T | None:
        """Return an existing resource without constructing one."""
        with self._lock:
            return None if self._value is None else self._value[0]

    def clear(self) -> None:
        """Release ownership; the next get constructs a fresh resource."""
        with self._lock:
            self._value = None


class BlockingWorker:
    """Run blocking calls one at a time on a thread, behind a bounded queue.

    Any event loop may submit calls. Only the running call is handed to the
    executor, so queued calls can still be withdrawn when their caller cancels.
    """

    def __init__(self, name: str, capacity: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._lock = Lock()
        self._queue: deque[_Job] = deque()
        self._busy = False
        self._capacity = capacity
        self._closed = False
        self._idle = Event()
        self._idle.set()

    def _dispatch(self) -> None:
        # Called under _lock. Only the active job is submitted to the executor.
        if not self._busy and self._queue:
            job = self._queue.popleft()
            self._busy = True
            self._idle.clear()
            job[0].set_running_or_notify_cancel()
            self._pool.submit(self._execute, job)

    def _execute(self, job: _Job) -> None:
        result, function = job
        try:
            try:
                value = function()
            finally:
                # Settle admission before waking callers that may request idle cleanup.
                with self._lock:
                    self._busy = False
                    self._dispatch()
                    if not self._busy:
                        self._idle.set()
        except BaseException as exc:  # noqa: BLE001 - transport the worker's original exception
            result.set_exception(exc)
        else:
            result.set_result(value)

    async def run[R, *Args](self, function: Callable[[*Args], R], *args: *Args) -> R:
        """Run function(*args) on the worker thread and return its result.

        Raise ValueError when the queue is full and RuntimeError once closed.
        Cancelling the caller withdraws a queued call; a running call finishes
        regardless.
        """
        result: Future[object] = Future()
        job: _Job = result, partial(function, *args)
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker is closed")
            if len(self._queue) >= self._capacity:
                raise ValueError(
                    "Too many requests at once. Try again when the current ones finish."
                )
            self._queue.append(job)
            self._dispatch()
        wrapped = asyncio.wrap_future(result)
        # Consume failures of abandoned running calls; their request is canceled.
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            # This heterogeneous queue entry belongs only to the supplied function.
            return cast(R, await asyncio.shield(wrapped))
        except asyncio.CancelledError:
            with self._lock:
                if result.cancel():
                    self._queue.remove(job)
            raise

    async def maintain(self, function: Callable[[], None]) -> None:
        """Run idle maintenance without occupying a slot needed by a request."""
        result: Future[object] = Future()
        with self._lock:
            if self._closed or self._busy or self._queue:
                return
            self._queue.append((result, function))
            self._dispatch()
        wrapped = asyncio.wrap_future(result)
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        await asyncio.shield(wrapped)

    def close(self, finalizer: Callable[[], None] | None = None) -> None:
        """Drain calls, run finalizer on the worker thread, then shut it down.

        Reject new submissions as soon as closing begins. This call blocks the
        calling thread until queued/running work and any finalizer finish; call
        it outside the worker thread. A finalizer failure propagates after
        executor shutdown.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._idle.wait()
        try:
            if finalizer is not None:
                self._pool.submit(finalizer).result()
        finally:
            self._pool.shutdown(wait=True)
