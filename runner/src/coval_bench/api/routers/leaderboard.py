# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/leaderboard — aggregated benchmark leaderboard.

Metric/benchmark compatibility:
- WER  + STT
- TTFT + STT
- TTFS + STT
- TTFA + TTS
- V2V  + S2S
- TTFT + LLM

Every window reads its published dashboard snapshot and ranks on the
aggregates' headline value.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from posthog import Posthog
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench import scenarios
from coval_bench.api.common import (
    MIN_SCORED_SAMPLES,
    BenchmarkLiteral,
    WindowLiteral,
    has_enough_samples,
)
from coval_bench.api.dashboard_windows import dashboard_read, require_window_state
from coval_bench.api.deps import capture_api_event, get_pool, get_posthog
from coval_bench.api.internal import hidden_early_access
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import LeaderboardEntry, LeaderboardResponse
from coval_bench.config import DATASET_ALL
from coval_bench.db.dashboard_windows import WINDOW_VIEWS
from coval_bench.registries import is_metric_excluded
from coval_bench.registries.benchmarks import Benchmark

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["leaderboard"])

MetricLiteral = Literal["WER", "TTFA", "TTFT", "TTFS", "V2V"]

# Valid (metric, benchmark) combinations — lower is better for all.
_VALID_COMBOS: set[tuple[str, str]] = {
    ("WER", "STT"),
    ("TTFT", "STT"),
    ("TTFS", "STT"),
    ("TTFA", "TTS"),
    ("V2V", "S2S"),
    ("TTFT", "LLM"),
}

# The headline S2S board is the Ultra Bank instruction-following set, which every
# S2S agent runs daily. Dental and the multi-turn set are frozen rather than
# retired: nothing writes to them, but they stay reachable through the aggregates
# ``dataset`` param, as does ``__all__`` for callers that want every S2S condition
# pooled. LLM runs the same bank scenario over text.
_PRIMARY_DATASET_BY_BENCHMARK = {
    b.value: scenarios.ACTIVE.primary_dataset(b) for b in (Benchmark.S2S, Benchmark.LLM)
}

_LEADERBOARD_SQL = """
    SELECT provider, model,
           avg_value AS avg,
           p50,
           p95,
           sample_count AS n
    FROM {view} v JOIN benchmarks_v2.metrics m ON m.id = v.metric_id
    WHERE m.code = %(metric)s
      AND benchmark = %(benchmark)s
      AND dataset_id = %(dataset)s
    ORDER BY sample_count < %(min_samples)s, avg_value ASC
"""


@router.get("/leaderboard", response_model=LeaderboardResponse)
@limiter.limit("60/minute")
async def get_leaderboard(
    request: Request,  # required by slowapi
    metric: MetricLiteral = Query(...),
    benchmark: BenchmarkLiteral = Query(...),
    window: WindowLiteral = Query(default="24h"),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
    posthog_client: Posthog | None = Depends(get_posthog),
    hidden: frozenset[tuple[str, str]] = Depends(hidden_early_access),
) -> LeaderboardResponse:
    """Return leaderboard entries sorted ascending by average metric value.

    Entries under the modality's sample floor are flagged and sink below every
    ranked entry, so a model that scored once with a fast time cannot lead.

    Args:
        metric: One of WER, TTFA, TTFT, TTFS, V2V.
        benchmark: One of STT, TTS, S2S, LLM.
        window: Time window — each is served by its published snapshot.

    Returns:
        ``{"metric": ..., "window": ..., "entries": [LeaderboardEntry, ...]}``

    Raises:
        400: If the metric/benchmark combination is incompatible.
    """
    if (metric, benchmark) not in _VALID_COMBOS:
        raise HTTPException(
            400,
            f"metric={metric!r} is not compatible with benchmark={benchmark!r}. "
            f"Valid combinations: WER+STT, TTFT+STT, TTFS+STT, TTFA+TTS, V2V+S2S, TTFT+LLM.",
        )

    params: dict[str, Any] = {
        "metric": metric,
        "benchmark": benchmark,
        "dataset": _PRIMARY_DATASET_BY_BENCHMARK.get(benchmark, DATASET_ALL),
        "min_samples": MIN_SCORED_SAMPLES.get(benchmark, 0),
    }
    sql = _LEADERBOARD_SQL.format(view=WINDOW_VIEWS[window])
    async with dashboard_read(pool) as conn:
        snapshot = await require_window_state(conn)
        rows = await conn.execute(sql, params)
        entry_rows = await rows.fetchall()

    entries = [
        LeaderboardEntry.model_validate(
            {**r, "insufficient_samples": not has_enough_samples(benchmark, r["n"])}
        )
        for r in entry_rows
        if (r["provider"], r["model"]) not in hidden
        and not is_metric_excluded(r["provider"], r["model"], metric)
    ]
    capture_api_event(
        posthog_client,
        "leaderboard_queried",
        {
            "metric": metric,
            "benchmark": benchmark,
            "window": window,
            "entry_count": len(entries),
            "$process_person_profile": False,
        },
    )
    return LeaderboardResponse(
        metric=metric,
        window=window,
        entries=entries,
        snapshot=asdict(snapshot),
    )
