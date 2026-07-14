# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for response compression.

The aggregates payload for wide windows exceeds Cloud Run's 32 MiB response cap
uncompressed, which returns a 500. GZipMiddleware keeps it well under.
"""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.api.conftest import _fill_buckets, _insert_result, _insert_run, _refresh_mv


async def test_large_aggregates_response_is_gzipped(client: AsyncClient, postgresql: Any) -> None:
    """A response over the size threshold is gzip-encoded and still decodes."""
    run_id = await _insert_run(postgresql)
    for i in range(12):
        for value in (1.0, 2.0, 3.0, 4.0):
            await _insert_result(
                postgresql,
                run_id,
                provider=f"prov{i}",
                model=f"model{i}",
                metric_type="WER",
                metric_value=value,
            )
    await _refresh_mv(postgresql)
    await _fill_buckets(postgresql)

    response = await client.get(
        "/v1/results/aggregates",
        params={"benchmark": "STT", "window": "30d"},
        headers={"Accept-Encoding": "gzip"},
    )

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    # Body still decodes through the transparent gunzip.
    assert len(response.json()["model_stats"]) == 12


async def test_small_response_is_not_compressed(client: AsyncClient) -> None:
    """Responses under the size threshold are sent uncompressed."""
    response = await client.get(
        "/v1/results/aggregates",
        params={"benchmark": "STT"},
        headers={"Accept-Encoding": "gzip"},
    )

    assert response.status_code == 200
    assert "content-encoding" not in response.headers
