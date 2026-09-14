# Saved dashboard aggregates

BENCH-899 implements run-completion precomputation for the existing normalized
read flag. Raw observations remain the source of truth. The migration, job, and
read cutover have not been applied to production.

## Storage and metric rules

Migration `20260914_0034` creates the base saved aggregate tables and views in
[PR #645](https://github.com/coval-ai/benchmarks/pull/645); the original aggregate
application is in [PR #634](https://github.com/coval-ai/benchmarks/pull/634).
Migration `20260914_0035` is isolated in
[PR #650](https://github.com/coval-ai/benchmarks/pull/650). Compatible metric-ID
readers, writers, and regression tests are in
[PR #649](https://github.com/coval-ai/benchmarks/pull/649), stacked on that
migration branch. The original 0034 migration remains unchanged.

| Object | Purpose |
| --- | --- |
| `metrics` | Generated IDs, immutable codes, and mutable display names (0035) |
| `normalized_results_24h`, `_7d`, `_30d` | Exact saved summary statistics, including all six current percentiles |
| `dashboard_summary_state` | Atomic generation, window boundary, publication time, definition identity |
| `dashboard_hourly_aggregates` | Primary sums/counts, ratio operands, coverage, latest source time |
| `dashboard_hourly_state` | Successful coverage, including empty hours, and pending rebuild status |
| `dashboard_source_refreshes` | Durable requests to rebuild source buckets after run completion |

Migration 0035 creates `metrics`, a database-local dimension with generated
`BIGINT` IDs, immutable canonical `code` values, and mutable `display_name`
labels. The twelve current metrics are seeded by the migration.
Application publication registers any future metric with a matching enum/spec/
contract definition using `ON CONFLICT DO NOTHING` before writing aggregates,
then resolves the complete code-to-ID mapping. Definitions are retained for
historical rows: IDs and codes cannot be changed, and rows cannot be deleted or
truncated.

Rows retain provider, model, benchmark, dataset (including pooled `__all__`),
metric ID, version, and evaluation variant as typed dimensions. Numeric statistics
and timestamps are typed columns. Small `metadata JSONB` objects currently hold
`schema_version: 1`; they are not used for dashboard filtering or arithmetic.

Summary membership is `[as_of - duration, as_of)`, using observation capture time.
All three views publish in one repeatable-read transaction with one generation.
Their statistics preserve the current WER pooling/fallback, TTFA components,
means, counts, spread, extrema, and exact p25/p50/p75/p90/p95/p99. Percentiles are
computed from individual observations during refresh, never combined from
previous percentiles. BENCH-884 remains the separate percentile UI follow-up.

New mean metrics share the existing row schema. Hourly ratios use registered
operand names, units, scale, and fallback rules; every source bucket must have
matching operand coverage. The current summary projection supports WER's declared
ratio and rejects unsupported ratio definitions until the projection is extended.
A fingerprint of registered contracts prevents reads using old saved definitions.
Changing materialization semantics also requires a definition revision change.
The hourly table has a real foreign key to `metrics.id`, so an unknown ID fails
the write. Materialized views cannot carry foreign keys; their defining SQL uses
`metric_id_for_code` and fails when a grouped source code has no retained
definition. Readers join the dimension and return the canonical code as the
public `metric_type`, preserving the existing API and raw table contract.
Hourly and summary publication only considers source rows with metric version
`v1` and evaluation variant `default`; other identities remain isolated.

## Publication, repair, and readers

Once migration 0034 is applied, run completion durably enqueues its source bucket
in the completion transaction. If a runner image arrives before the migration,
an enqueue savepoint isolates missing-table errors so the run's status, finish
time, and error still commit. The writer emits
`dashboard_source_refresh_enqueue_skipped` with the run ID and required migration.
Other enqueue errors still propagate and roll back the completion transaction.

After applying 0034, explicitly repair source buckets for runs completed during
the pre-0034 gap before enabling saved reads. Use the existing
`repair-dashboard-aggregates --bucket ...` command with each distinct non-null
`runs.scheduled_at` for runs that finished during the gap and have normalized
observations. Include failed runs, since rebuilding also removes contributions.
Scheduled maintenance cannot discover requests skipped while the queue was absent.

Source replacement takes the UTC-hour lock before the source-bucket lock. It
claims the pending request before reading source observations, replaces the
bucket, and marks its hour dirty in one transaction. A failed replacement keeps
both previous data and the pending request. Hourly rebuilds replace entire hours,
so retries do not double-count. Synchronous normalized backfills use the same
lock order and publish affected hours after committing their source updates.

Summary publication uses a separate advisory lock and can coalesce running
siblings. Normalized summary failure is independent of legacy maintenance and
run outcome. Scheduled reconciliation drains pending source requests and missing
or dirty hours, then refreshes summaries even if another maintenance stage fails.
Each phase has a time limit; completed hour repairs remain committed. A failed
stage makes the maintenance command fail so it can be retried.

Normalized summary consumers read saved rows and state in one repeatable-read
transaction. Missing, uninitialized, or incompatible storage returns
`503 dashboard_snapshot_not_ready`. Successful empty snapshots return empty data
with a generation. Summary and averaged-timeline responses bypass the old TTL
cache. The frontend indicates when saved data is stale. Summary snapshots become
stale after two hours, allowing two hourly maintenance intervals.

The 24h timeline retains per-run points and local zoom. The 7d view uses hourly
averages; 30d combines those sufficient statistics into four-hour averages. Zoom
can fetch a smaller range with one hour as the finest interval. Exact requested
bounds combine complete saved hours with at most two partial source-hour ranges;
a partial hour is never included twice. Source freshness remains the latest
actual source timestamp, separate from materialization time.

## Commands and rollout

The runner image supports:

```sh
python -m coval_bench db refresh-dashboard-aggregates
python -m coval_bench db repair-dashboard-aggregates \
  --bucket 2026-09-14T10:00:00Z --bucket 2026-09-14T11:00:00Z
```

Repair accepts exact source-bucket timestamps. Include both old and new buckets
when changing scheduled time or moving data, including an old bucket that is now
empty. `--as-of` fixes summary membership for validation/replay. Normal scheduling
uses the current time so old observations expire without new ingestion.

The infrastructure change defines a database-only Cloud Run job every hour,
with a 600-second timeout, one retry, and image updates through the existing
runner image workflow. Its scheduler is created paused.

Merge order is #645 (tables), #634 (aggregate application), #650 (metric-ID
migration), then #649 (metric-ID application). Retarget each dependent PR to
`main` after its prerequisites merge. Merging a migration PR does not apply it
to the database.

1. Apply the unchanged 0034 prerequisite migration if needed. Disable normalized
   reads and pause/drain aggregate maintenance and backfill workers before
   applying 0035, then deploy the compatible BENCH-906 application. Migration
   0035 recreates saved views unpopulated, preserves hourly
   rows and summary generation, and invalidates readiness; an unknown historical
   hourly code fails the whole migration transaction.
2. Apply the reviewed infrastructure plan through Atlantis and install the
   compatible runner image in the maintenance job. Keep its scheduler paused.
3. If writers ran before 0034, repair their skipped source buckets as described
   above. Run maintenance to initialize source/hour coverage and all summary
   views, then resume the scheduler and confirm successful recurring refreshes.
4. Verify production read latency and refresh load, then enable the normalized
   read flag. Disable it to return to legacy reads if needed.

Do not infer production performance from the small runtime smoke test. Local
validation exercised real Alembic migration, normalized RunWriter completion,
the maintenance CLI, and all four HTTP consumers. With 12 observations in one
source bucket, maintenance initialized 721 hours and published one generation;
all HTTP requests returned 200, the source queue drained, and PostgreSQL stopped.
Focused tests cover ratios/percentiles, missing and empty state, refresh rollback,
locks, retries, exact zoom boundaries, metadata, and affected ingestion paths.

An earlier representative synthetic prototype used 120,016 observations and
240,018 projected evaluations. All 54 raw/saved comparisons matched; warm STT
7d execution/fetch median was 52.80 ms raw versus 0.44 ms saved, and 30d was
200.80 ms versus 0.43 ms. Refreshing all windows took 0.80 s. These are local
measurements; the production cutover still requires its own timing check.
