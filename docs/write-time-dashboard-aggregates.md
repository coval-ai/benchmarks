# Saved dashboard aggregates

BENCH-899 implements run-completion precomputation for the existing normalized
read flag. Raw observations remain the source of truth. Production rollout
requires verifying the migration, initializing saved data, and enabling the job
before the read cutover.

## Storage and metric rules

Migration `20260914_0034` creates three materialized views and four tables. Its
separate [PR #645](https://github.com/coval-ai/benchmarks/pull/645) has merged into
`main`; verify that the migration has also been applied to the database before
deploying the application. The application was reviewed in
[PR #634](https://github.com/coval-ai/benchmarks/pull/634) and is delivered through
a follow-up PR targeting `main`.

| Object | Purpose |
| --- | --- |
| `normalized_results_24h`, `_7d`, `_30d` | Exact saved summary statistics, including all six current percentiles |
| `dashboard_summary_state` | Atomic generation, window boundary, publication time, definition identity |
| `dashboard_hourly_aggregates` | Primary sums/counts, ratio operands, coverage, latest source time |
| `dashboard_hourly_state` | Successful coverage, including empty hours, and pending rebuild status |
| `dashboard_source_refreshes` | Durable requests to rebuild source buckets after run completion |

Rows retain provider, model, benchmark, dataset (including pooled `__all__`),
metric, version, and evaluation variant as typed dimensions. Numeric statistics
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

### Stable metric identities

Migration `20260914_0035`, delivered in [PR #650](https://github.com/coval-ai/benchmarks/pull/650),
adds the `metrics` dimension and generated IDs to hourly and summary storage.
Both `metric_id` and `metric_type` remain available; the hourly trigger keeps
them consistent so existing code-based writers remain compatible. Raw
evaluations and source buckets continue to store canonical metric codes.

[PR #649](https://github.com/coval-ai/benchmarks/pull/649) uses those IDs in
dashboard publication and joins them back to codes for API responses. Before
publication, registration inserts missing registry definitions without replacing
existing IDs or display names. Unknown or unsupported source definitions fail
the transaction, preserving the previous publication.

Metric-ID application rollout requires 0035 and does not change its SQL. It
advances the application definition revision from 1 to 2, so a refresh completed
by the older application must be repeated after deploying the new application.
The schema migration's compatibility columns do not make old publication
fingerprints compatible with the new application.

## Publication, repair, and readers

Once migration 0034 is applied, run completion durably enqueues its source bucket
in the completion transaction. If a runner image arrives before the migration,
an enqueue savepoint isolates missing-table errors so the run's status, finish
time, and error still commit. The writer emits
`dashboard_source_refresh_enqueue_skipped` with the run ID and required migration.
Other enqueue errors still propagate and roll back the completion transaction.

After applying the migration, explicitly repair source buckets for runs completed
during that gap before enabling saved reads. Use the existing
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
siblings. The STT/TTS completion hook publishes only after succeeded or partial
runs. Failed runs keep their queued source repairs for hourly maintenance without
waiting for summary publication. If the last sibling fails after earlier siblings
deferred publication, the hourly job publishes their results. Normalized summary
failure is independent of legacy maintenance and run outcome.
Scheduled reconciliation drains pending source requests and missing
or dirty hours, then refreshes summaries even if another maintenance stage fails.
Each phase has a time limit; completed hour repairs remain committed. A failed
stage makes the maintenance command fail so it can be retried.

Normalized summary consumers read saved rows and state in one repeatable-read
transaction. Missing, uninitialized, or incompatible storage returns
`503 dashboard_snapshot_not_ready`. Successful empty snapshots return empty data
with a generation. Summary and averaged-timeline responses bypass the old TTL
cache. The frontend indicates when saved data is stale. Summary snapshots become
stale after two hours, allowing two hourly maintenance intervals.

Averaged timelines require every complete hour to have a compatible publication
record, including hours with no samples. Missing records still return 503: run
completion queues a source repair before the rebuild marks its hour dirty, so a
missing hour is not proof that no data exists. Maintenance alerts cover failed
executions and no successful execution within 90 minutes.

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
The existing alerting module pages on any failed execution in the last hour and
on no successful execution within 90 minutes. The heartbeat also fires before
the first successful run, so complete initialization and resume the scheduler as
part of the same rollout; a deliberately paused job will continue to alert.

1. Verify migration `20260914_0034` from the already-merged PR #645 has been
   applied. Merge the application follow-up PR targeting `main`, and deploy
   compatible writers with normalized reads still disabled. Views are created
   without initial population. Deploy
   [frontend PR #95](https://github.com/coval-ai/benchmarks-web/pull/95) alongside
   or immediately afterward: legacy 7d/30d timelines also switch to averages,
   independently of the normalized read flag.
2. If writers ran before the migration, repair their skipped source buckets as
   described above. Run maintenance to initialize source/hour coverage and all
   summary views.
3. Apply the reviewed infrastructure plan through Atlantis, install the compatible
   runner image, and resume the scheduler. Confirm successful recurring refreshes.
   Verify the `dashboard-aggregates` failure and 90-minute heartbeat alerts.
4. Verify production read latency and refresh load, then enable the normalized
   read flag. Disable it to return to legacy reads if needed.

### Metric-ID application cutover and 30-day backfill

1. Verify migration `20260914_0035` is applied. Disable normalized saved reads
   and pause/drain aggregate maintenance and backfill workers while deploying
   the metric-ID application to the runner, API, and maintenance job. Mixed
   application revisions can publish incompatible fingerprints.
2. Repair missing or skipped source buckets needed for the last 30 days with
   `python -m coval_bench db repair-dashboard-aggregates --bucket <ISO_TIME>`.
   Repeat `--bucket` for additional timestamps. Maintenance cannot discover
   historical source buckets that were never queued.
3. Run `python -m coval_bench db refresh-dashboard-aggregates` using the new
   application. It rebuilds rolling 30-day hourly coverage and publishes the
   24h, 7d, and 30d summaries. Each maintenance phase has a time limit; completed
   hours remain committed, so repeat while pending work decreases. Older dirty
   or queued work can also be processed; 30 days is the required coverage,
   not a strict processing cutoff. This rebuild uses saved observations and
   does not rerun benchmarks or scoring.
4. Verify source repairs are drained, required hourly coverage is complete,
   summaries are published, and representative 7d/30d API responses preserve
   metric codes and expected values. Resume scheduling and enable saved reads
   after these checks pass.

Converting metric identity in the other normalized tables is a separate
follow-up in [BENCH-911](https://linear.app/coval/issue/BENCH-911/extend-metric-ids-to-normalized-evaluations-and-source-rollups).

### Normalized metric-ID backfill

Migration `0036` adds nullable `metric_id` columns to normalized evaluations,
their successful dashboard projections, and source buckets. Apply it before
running the operator command below. The command registers only definitions
already known to the runner and preserves the existing catalog IDs.

```sh
python -m coval_bench migrate backfill-normalized-metric-ids
python -m coval_bench migrate backfill-normalized-metric-ids --apply \
  --batch-size 1000 --max-runtime-seconds 600
# For large tables, build/recover temporary null-ID indexes concurrently first:
python -m coval_bench migrate backfill-normalized-metric-ids --apply \
  --create-pending-indexes
```

The first invocation is a read-only preflight. `--apply` is required for
writes. Evaluations are hydrated first, projections copy the parent effective
identity second, and source buckets are hydrated last. Each batch is a separate
transaction, selects null IDs with `SKIP LOCKED`, and can be
rerun after interruption. `--max-batches` and `--max-runtime-seconds` bound an
operator run; the final report includes pending, unknown-code, mismatch, and
skipped-batch counts. `verification_complete: false` means the time limit
prevented final verification; rerun the read-only preflight before using the
coverage counts as final. Unknown codes and code/ID conflicts fail closed for manual
reconciliation. The command changes only `metric_id`; scores, dimensions,
timestamps, and payloads are never rewritten.

Run the read-only readiness checker after each bounded run. It reports ID
coverage alongside the existing raw and rollup parity checks:

```sh
python runner/scripts/check_normalized_readiness.py --database-url "$DATABASE_URL"
```

Non-null enforcement requires a separate migration. After the preflight reports zero pending,
unknown, and mismatched rows across repeated runs, validate foreign keys and
projection/parent agreement, then add the non-null and ID-based uniqueness
constraints. Keep the code columns and compatibility keys until every writer
has been upgraded. To roll back before non-null enforcement, stop the
backfill and deploy the prior application; leave nullable columns in place so
the retained code keys continue to serve retries. Do not drop or rewrite
historical rows as part of rollback.

The backfill is intentionally excluded from ordinary startup and Alembic
upgrade execution. Measure table sizes, index use, lock wait behavior, and
batch throughput in the target environment first. If a large deployment needs
additional acceleration, an operator may create temporary partial indexes
concurrently and remove them after reconciliation. The explicit
`--create-pending-indexes` mode removes invalid leftovers from an interrupted
concurrent build before retrying.

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
