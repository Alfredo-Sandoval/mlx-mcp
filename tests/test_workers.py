"""Lazy resources and the serialized inference worker."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from mlx_mcp.workers import BlockingWorker, LazyResource


def test_canceled_queue_is_removed_and_running_capacity_is_held() -> None:
    worker = BlockingWorker("test-inference", capacity=1)
    release = Event()
    calls: list[object] = []

    async def check() -> None:
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def blocking(name: str) -> str:
            calls.append(name)
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return name

        first = asyncio.create_task(worker.run(blocking, "running"))
        await asyncio.wait_for(started.wait(), 2)
        queued = asyncio.create_task(worker.run(blocking, "canceled"))
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="Too many requests"):
            await worker.run(blocking, "overflow")
        first.cancel()
        queued.cancel()
        for task in (first, queued):
            with pytest.raises(asyncio.CancelledError):
                await task
        next_task = asyncio.create_task(worker.run(blocking, "next"))
        await asyncio.sleep(0)
        assert calls == ["running"]
        # Saturated inference does not occupy the default executor.
        assert await asyncio.wait_for(asyncio.to_thread(lambda: "fetch"), 1) == "fetch"
        release.set()
        assert await asyncio.wait_for(next_task, 2) == "next"

    try:
        asyncio.run(check())
    finally:
        release.set()
        worker.close()
    assert calls == ["running", "next"]


def test_lazy_resource_serializes_concurrent_constructors() -> None:
    calls: list[object] = []
    barrier = Barrier(12)

    def factory():
        result = object()
        calls.append(result)
        return result

    resource = LazyResource(factory)

    def worker(_: int):
        barrier.wait(timeout=5)
        return resource.get()

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(worker, range(12)))
    assert len(calls) == 1
    assert all(result is calls[0] for result in results)
