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
import time
from collections.abc import Awaitable, Callable, Mapping

import structlog

logger = structlog.get_logger("coval_bench.runner")

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
    retry_event: str = "provider_call_retry",
    exhaustion_event: str = "retry_exhausted",
    retry_state: Callable[[], Mapping[str, object]] | None = None,
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
        retry_event: Structured event name emitted for retry attempts.
        exhaustion_event: Structured event name emitted when retries are exhausted.
        retry_state: Optional synchronous callback returning diagnostic state captured around
            each attempt. Callback failures are ignored.

    Returns:
        The return value of the first successful *fn* call.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        state_before: Mapping[str, object] = {}
        if retry_state is not None:
            try:
                state_before = retry_state()
            except Exception:  # Diagnostics must never affect operation behavior.
                state_before = {}
        started = time.monotonic()
        try:
            return await fn()
        except retry_on as exc:
            last_exc = exc
            attempt_elapsed_ms = round((time.monotonic() - started) * 1000)
            state_after: Mapping[str, object] = {}
            if retry_state is not None:
                try:
                    state_after = retry_state()
                except Exception:
                    state_after = {}
            if attempt + 1 >= max_attempts:
                logger.error(
                    exhaustion_event,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    exc_info=exc,
                    exception_type=type(exc).__name__,
                    attempt_elapsed_ms=attempt_elapsed_ms,
                    state_before=state_before,
                    state_after=state_after,
                )
                break
            cap = min(base_delay_s * (2**attempt), max_delay_s)
            # Full jitter — non-cryptographic, used only for backoff scheduling
            delay = random.uniform(0, cap)  # noqa: S311
            logger.warning(
                retry_event,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_s=round(delay, 3),
                exc_info=exc,
                exception_type=type(exc).__name__,
                attempt_elapsed_ms=attempt_elapsed_ms,
                state_before=state_before,
                state_after=state_after,
            )
            await asyncio.sleep(delay)

    # last_exc is always set when we reach here (max_attempts >= 1 and the
    # loop only breaks after catching an exception)
    assert last_exc is not None  # noqa: S101 — unreachable otherwise
    raise last_exc
