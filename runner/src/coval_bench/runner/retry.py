# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Exponential-backoff retry helper for provider calls.

Pure function, no global state. Logs at WARNING per retry attempt and ERROR
on final failure. Re-raises the last exception on exhaustion.

Usage::

    result = await with_retry(
        lambda: provider.measure_ttft(audio_bytes, ...),
        max_attempts=3,
    )
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import structlog

_log = structlog.get_logger("coval_bench.runner")

_DEFAULT_RETRY_ON: tuple[type[BaseException], ...] = (
    TimeoutError,  # asyncio.TimeoutError is an alias for builtin TimeoutError (Python 3.11+)
    OSError,
    ConnectionError,
)


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = _DEFAULT_RETRY_ON,
) -> T:
    """Call *fn* up to *max_attempts* times with exponential backoff + full jitter.

    Delay formula: ``min(base_delay_s * 2**attempt, max_delay_s)`` with full jitter
    (uniform random in ``[0, cap]``).

    Args:
        fn: Zero-argument async callable to invoke.
        max_attempts: Total number of attempts (including the first).
        base_delay_s: Base delay in seconds before the first retry.
        max_delay_s: Maximum delay cap in seconds.
        retry_on: Exception types that trigger a retry. All others propagate immediately.

    Returns:
        The return value of the first successful *fn* call.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retry_on as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                _log.error(
                    "retry exhausted",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    exc_info=exc,
                )
                break
            cap = min(base_delay_s * (2**attempt), max_delay_s)
            # Full jitter — non-cryptographic, used only for backoff scheduling
            delay = random.uniform(0, cap)  # noqa: S311
            _log.warning(
                "provider call failed — retrying",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_s=round(delay, 3),
                exc_info=exc,
            )
            await asyncio.sleep(delay)

    # last_exc is always set when we reach here (max_attempts >= 1 and the
    # loop only breaks after catching an exception)
    assert last_exc is not None  # noqa: S101 — unreachable otherwise
    raise last_exc
