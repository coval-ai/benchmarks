# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/results — filtered list of benchmark results.

Always filters ``r.status = 'success'`` (result-row level). This is independent
of the parent-run status, which is controlled by ``include_failed``:

* ``include_failed=False`` (default): only returns results whose parent run has
  ``status IN ('succeeded', 'partial')``. Failed and running runs are excluded.
* ``include_failed=True``: results from *all* parent runs (including failed and
  running) are included, provided the result row itself has ``status='success'``.

The ``status`` field on ``ResultOut`` reflects the *parent run* status (uppercase),
not the result-row status. It is denormalized here at the API boundary via SQL JOIN.

Note on window routing:
* ``window=24h``: queries the live ``benchmarks_v2.results`` table directly with a
  24-hour filter. The ``results_24h`` materialized view is NOT used here because it
  aggregates by (provider, model, benchmark, metric_type) and lacks row-level columns
  (id, voice, metric_value, etc.) required by ``ResultOut``. The existing index
  ``results_provider_model_idx`` supports this path.
* ``window=7d|30d``: live table join with interval filter.
* ``since``/``until``: live table join with explicit timestamp bounds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import psycopg.rows
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request

from coval_bench.api.deps import get_pool
from coval_bench.api.ratelimit import limiter
from coval_bench.api.schemas import ResultOut, ResultsResponse

logger = structlog.get_logger("coval_bench.api")

router = APIRouter(tags=["results"])

# ``MetricLiteral`` is kept as the type for the legacy ``metric`` param.
# The canonical FE-facing param is ``metric_type`` (plain ``str``).
MetricLiteral = Literal["WER", "TTFA", "TTFT", "RTF", "AUDIO_TO_FINAL"]
BenchmarkLiteral = Literal["STT", "TTS"]
WindowLiteral = Literal["24h", "7d", "30d"]

# Fixed interval strings — looked up by Python, never user-interpolated into SQL.
_WINDOW_INTERVALS: dict[str, str] = {
    "24h": "24 hours",
    "7d": "7 days",
    "30d": "30 days",
}

# Base SELECT used by all three query paths.
_SELECT = (
    "SELECT r.id, r.run_id, r.provider, r.model, r.voice, r.benchmark,"
    " r.metric_type, r.metric_value, r.metric_units, r.audio_filename,"
    " r.created_at, UPPER(rn.status) AS status"
    " FROM benchmarks_v2.results r"
    " JOIN benchmarks_v2.runs rn ON rn.id = r.run_id"
)


@router.get("/results", response_model=ResultsResponse)
@limiter.limit("60/minute")
async def list_results(
    request: Request,  # required by slowapi
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    metric: MetricLiteral | None = Query(
        default=None,
        description=(
            "Deprecated alias for metric_type. Kept for backward compatibility with legacy "
            "callers. Use metric_type for new integrations. If both are provided and equal, "
            "the request succeeds. If they differ, a 400 is returned."
        ),
    ),
    metric_type: str | None = Query(
        default=None,
        description="Filter on metric_type (e.g. WER, TTFA, TTFT, RTF). Canonical FE-facing name.",
    ),
    benchmark: BenchmarkLiteral | None = Query(default=None),
    window: WindowLiteral | None = Query(
        default=None,
        description=(
            "Time window for results. One of '24h', '7d', '30d'. "
            "Defaults to '7d' when neither window nor since/until are supplied. "
            "Mutually exclusive with since/until."
        ),
    ),
    since: datetime | None = Query(
        default=None,
        description="Lower bound on created_at (ISO 8601). Mutually exclusive with window.",
    ),
    until: datetime | None = Query(
        default=None,
        description="Upper bound on created_at (ISO 8601). May be combined with since.",
    ),
    include_failed: bool = Query(
        default=False,
        description=(
            "If false (default), only returns results whose parent run is SUCCEEDED or PARTIAL. "
            "If true, results from FAILED and RUNNING parent runs are also included. "
            "The result row's own status='success' filter is always applied regardless."
        ),
    ),
    limit: int = Query(
        default=100000,
        ge=1,
        le=100000,
        description="Maximum rows to return (1–100000, default 100000).",
    ),
    pool: AsyncConnectionPool[Any] = Depends(get_pool),
) -> ResultsResponse:
    """Return a newest-first page of successful benchmark results.

    All optional filters are ANDed together.

    **Result-row vs run-level status distinction:**
    ``include_failed`` controls whether results from *failed runs* are included.
    The result-row ``status='success'`` filter is always applied — there is no
    way to opt in to failed result rows in this phase.

    **``metric`` param is deprecated** — use ``metric_type`` for new code.
    Both are accepted; if both are provided and equal the request succeeds.
    If they differ a 400 is returned.

    **``window`` and ``since``/``until`` are mutually exclusive.** Providing both
    returns 400.
    """
    # -- Validation: since/until ordering
    if since is not None and until is not None and since > until:
        raise HTTPException(400, "since must not be after until")

    # -- Validation: window is mutually exclusive with since/until
    if window is not None and (since is not None or until is not None):
        raise HTTPException(400, "window cannot be combined with since/until")

    # Apply default window of '7d' when no time constraint is provided at all.
    if window is None and since is None and until is None:
        window = "7d"

    # -- Validation: metric / metric_type alias reconciliation
    resolved_metric: str | None = None
    if metric is not None and metric_type is not None:
        if metric != metric_type:
            raise HTTPException(
                400,
                "metric and metric_type are aliases — pass only one",
            )
        # Both provided and equal — treat as single value
        resolved_metric = metric
    elif metric is not None:
        resolved_metric = metric
    elif metric_type is not None:
        resolved_metric = metric_type

    # Build WHERE clause dynamically — parameterised only, no f-string SQL injection.
    conditions: list[str] = ["r.status = 'success'"]
    params: dict[str, Any] = {"limit": limit}

    if provider is not None:
        conditions.append("r.provider = %(provider)s")
        params["provider"] = provider
    if model is not None:
        conditions.append("r.model = %(model)s")
        params["model"] = model
    if resolved_metric is not None:
        conditions.append("r.metric_type = %(metric_type)s")
        params["metric_type"] = resolved_metric
    if benchmark is not None:
        conditions.append("r.benchmark = %(benchmark)s")
        params["benchmark"] = benchmark

    # -- Run-status filter (include_failed controls PARTIAL/FAILED run inclusion)
    if not include_failed:
        conditions.append("rn.status IN ('succeeded', 'partial')")

    # -- Time window / since-until conditions
    if window is not None:
        # Use a fixed-string interval from the lookup dict — never user-interpolated.
        interval_str = _WINDOW_INTERVALS[window]
        conditions.append(f"r.created_at >= NOW() - INTERVAL '{interval_str}'")  # noqa: S608
    else:
        # Path C: explicit since/until bounds (window is None when either is set)
        if since is not None:
            conditions.append("r.created_at >= %(since)s")
            params["since"] = since
        if until is not None:
            conditions.append("r.created_at <= %(until)s")
            params["until"] = until

    where_clause = " AND ".join(conditions)
    # S608 false-positive: where_clause is built entirely from pre-defined static
    # SQL fragments (constants in the ``conditions`` list). No raw user input is
    # interpolated into the SQL string itself; user values go through %(param)s.
    sql = f"{_SELECT} WHERE {where_clause} ORDER BY r.created_at DESC, r.id DESC LIMIT %(limit)s"  # noqa: S608

    async with pool.connection() as conn:
        conn.row_factory = psycopg.rows.dict_row
        rows = await conn.execute(sql, params)
        result_rows = await rows.fetchall()

    results = [ResultOut.model_validate(r) for r in result_rows]
    return ResultsResponse(results=results)
