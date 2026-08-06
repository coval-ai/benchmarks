# Internal cost tracking

Every benchmark item with a resolvable rate writes a sibling `COST_USD` row to
`benchmarks_v2.results` (metric-as-a-row, so spend rides the existing
`results → results_by_bucket → matviews` pipeline), and every run rolls up to
`runs.total_cost_usd` / `runs.judge_cost_usd` at `finish_run`.

How the number is computed (`coval_bench/metrics/cost.py`):

- rates come from `benchmarks_v2.model_pricing` (append-only; the rate
  effective at run time), loaded once per run;
- resolution order: token rates × token counts → duration rates ×
  `billable_seconds` (falling back to measured audio duration) → per-1M-chars
  × `characters_in`;
- no rate or no quantity → **no row** (never a zero). Failed items write no
  cost row either — whether a provider bills a failed call is unknowable, so
  the rollup undercounts by failed calls; treat it as a lower bound.

`COST_USD` is internal: the API filters it out of `/v1/results` and both
aggregates endpoints (`INTERNAL_METRICS` in `registries/metrics.py`). Public
price display uses list prices from `model_pricing`, not measured spend.

## Canonical queries

Spend per model per run (the "one GROUP BY away" query):

```sql
SELECT run_id, provider, model, SUM(metric_value) AS cost_usd
FROM benchmarks_v2.results
WHERE metric_type = 'COST_USD' AND status = 'success'
GROUP BY 1, 2, 3
ORDER BY run_id DESC, cost_usd DESC;
```

Run totals (items + judge, stamped at finish):

```sql
SELECT id, started_at, dataset_id, status, total_cost_usd, judge_cost_usd
FROM benchmarks_v2.runs
WHERE total_cost_usd IS NOT NULL
ORDER BY started_at DESC;
```

Daily spend by provider (from the series rollup — `value_sum` is the bucket's
total spend; the pooled `__all__` bucket rows would double-count, so filter):

```sql
SELECT date_trunc('day', bucket_at) AS day, provider,
       SUM(value_sum) AS cost_usd
FROM benchmarks_v2.results_by_bucket
WHERE metric_type = 'COST_USD' AND dataset_id = '__all__'
GROUP BY 1, 2
ORDER BY 1 DESC, cost_usd DESC;
```

Judge spend over time:

```sql
SELECT date_trunc('day', started_at) AS day, SUM(judge_cost_usd)
FROM benchmarks_v2.runs
GROUP BY 1 ORDER BY 1 DESC;
```

Per-run spend also lands in PostHog (project `public-benchmarks`) as one
`benchmark_run_cost` event per run: `total_cost_usd`, `judge_cost_usd`, and
the top-5 model costs.
