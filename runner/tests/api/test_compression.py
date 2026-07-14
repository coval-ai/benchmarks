# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for response compression.

The aggregates payload for wide windows exceeds Cloud Run's 32 MiB response cap
uncompressed, which returns a 500. GZipMiddleware keeps it well under.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.api.conftest import _fill_buckets, _insert_result, _insert_run, _refresh_mv

AppFactory = Callable[[dict[str, str] | None], Awaitable[FastAPI]]


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


async def test_clips_are_not_compressed(app_factory: AppFactory, tmp_path: Path) -> None:
    """Static clip responses stay identity-encoded.

    Gzipping them would return a 206 whose ``Content-Range`` describes offsets
    in the original file while the body is compressed, and would drop
    ``Content-Length`` from full-file responses.
    """
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip.wav").write_bytes(b"\x00" * 4096)
    app = await app_factory({"ARENA_AUDIO_DIR": str(tmp_path)})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        full = await c.get("/clips/clip.wav", headers={"Accept-Encoding": "gzip"})
        assert full.status_code == 200
        assert "content-encoding" not in full.headers
        assert full.headers["content-length"] == "4096"

        partial = await c.get(
            "/clips/clip.wav",
            headers={"Accept-Encoding": "gzip", "Range": "bytes=0-1499"},
        )
        assert partial.status_code == 206
        assert "content-encoding" not in partial.headers
        assert partial.headers["content-length"] == "1500"
        assert partial.content == b"\x00" * 1500
