import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final

import pytest
import redis.asyncio as async_redis
from integration._support.redis_process import owned_redis

from litellm.caching.redis_cache import RedisCache

_DrainScenario = Callable[[RedisCache, async_redis.Redis], Awaitable[None]]


def _run_against_real_redis(tmp_path: Path, redis_version: str | float, scenario: _DrainScenario) -> None:
    async def drive() -> None:
        with owned_redis(tmp_path) as redis_server:
            redis_cache: Final = RedisCache(host=redis_server.host, port=redis_server.port, namespace="drain-test")
            redis_cache.redis_version = redis_version
            redis_client: Final = async_redis.Redis(host=redis_server.host, port=redis_server.port)
            try:
                await scenario(redis_cache, redis_client)
            finally:
                await redis_client.aclose()
                await redis_cache.disconnect()

    asyncio.run(drive())


@pytest.mark.parametrize("redis_version", ["6.0", 6.1])
def test_legacy_redis_drain_keeps_rows_pending_in_order_until_acknowledged(
    redis_version: str | float, tmp_path: Path
) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3", "row-4", "row-5")

        first_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        processing_key: Final = "drain-test:buffer:processing:worker-1"
        assert first_batch == ["row-1", "row-2"]
        assert await redis_client.llen(processing_key) == 2
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-3", b"row-4", b"row-5"]

        pending_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        assert pending_batch == ["row-1", "row-2"]
        assert await redis_client.lrange(processing_key, 0, -1) == [b"row-1", b"row-2"]

        await redis_cache.async_ack_processing_list(key="buffer", worker_id="worker-1", ttl_seconds=60)
        assert await redis_cache.async_processing_list_state(key="buffer", worker_id="worker-1") == "acknowledged"
        next_batch: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )
        assert next_batch == ["row-3", "row-4"]
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-5"]

    _run_against_real_redis(tmp_path, redis_version, scenario)


def test_lua_drain_as_half_open_probe_closes_the_circuit_breaker(tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        breaker: Final = redis_cache._circuit_breaker
        await redis_client.rpush("drain-test:buffer", "row-1")
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        assert breaker.state == breaker.OPEN
        breaker.recovery_timeout = 0
        await asyncio.sleep(0.01)

        drained: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=1, ttl_seconds=60
        )

        assert drained == ["row-1"]
        assert breaker.state == breaker.CLOSED
        await redis_cache.async_set_cache("after-probe", "ok")
        assert await redis_client.get("drain-test:after-probe") is not None

    _run_against_real_redis(tmp_path, "6.0.16", scenario)


@pytest.mark.parametrize("redis_version", ["Unknown", "latest", ""])
def test_unknown_redis_version_drains_through_lua_without_lmove(redis_version: str, tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        await redis_client.execute_command("ACL", "SETUSER", "default", "-lmove")
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3")

        drained: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="worker-1", count=2, ttl_seconds=60
        )

        assert drained == ["row-1", "row-2"]
        assert await redis_client.lrange("drain-test:buffer:processing:worker-1", 0, -1) == [b"row-1", b"row-2"]
        assert await redis_client.lrange("drain-test:buffer", 0, -1) == [b"row-3"]

    _run_against_real_redis(tmp_path, redis_version, scenario)


@pytest.mark.parametrize("redis_version", ["6.0", "7.2"])
def test_abandoned_processing_list_returns_to_the_buffer_for_another_worker(redis_version: str, tmp_path: Path) -> None:
    async def scenario(redis_cache: RedisCache, redis_client: async_redis.Redis) -> None:
        if redis_version == "6.0":
            await redis_client.execute_command("ACL", "SETUSER", "default", "-lmove")
        await redis_client.rpush("drain-test:buffer", "row-1", "row-2", "row-3")
        abandoned: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="old-leader", count=2, ttl_seconds=60
        )
        assert abandoned == ["row-1", "row-2"]

        fresh_list_untouched: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="new-leader", count=3, ttl_seconds=60, recovery_seconds=60
        )
        assert fresh_list_untouched == ["row-3"]
        await redis_cache.async_ack_processing_list(key="buffer", worker_id="new-leader", ttl_seconds=60)

        recovered: Final = await redis_cache.async_drain_list_to_processing(
            key="buffer", worker_id="new-leader", count=3, ttl_seconds=60, recovery_seconds=0
        )
        assert recovered == ["row-1", "row-2"]
        assert await redis_cache.async_processing_list_state(key="buffer", worker_id="old-leader") == "missing"

    _run_against_real_redis(tmp_path, redis_version, scenario)
