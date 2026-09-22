# Results API v2

`GET /v2/results` returns one item per successful default metric evaluation on
a succeeded or partial parent run. See the API service's `/docs` or
`/openapi.json` for the complete response schema and query parameters.

## Examples

Primary values for one run and metric:

```text
/v2/results?run_id=789&metric_type=TTFA
```

Include named components and select a metric version:

```text
/v2/results?run_id=789&metric_type=TTFA&metric_version=v1&evaluation_variant=default&include_components=true
```

Export a time range across runs:

```text
/v2/results?benchmark=TTS&since=2026-09-01T00:00:00Z&until=2026-09-08T00:00:00Z&limit=1000
```

## Response and filters

- The response contains `results` and a nullable `next_cursor`.
- Each result contains `captured_at`, provider, model, voice, benchmark, dataset,
  metric type/version, and the stored primary `value`/`unit`.
- `value` and `unit` are guaranteed to be present for returned rows. Values are
  returned without conversion.
- `components` is omitted by default. With `include_components=true`, it maps
  component names to `{value, unit}` pairs; no components produces `{}`.
- `evaluation_variant` is restricted to `default` and `evaluation_status` to
  `succeeded`; incompatible explicit values return HTTP 422. `run_status` may
  be `succeeded` or `partial` and defaults to both.
- Filters combine with AND. Model access restrictions apply to every page.

## Pagination and time ranges

The page size defaults to 100 evaluations, with a maximum of 1000. Results sort
by capture time descending, then evaluation ID descending. Components do not
change page membership or consume page slots.

Repeat the original request with its `cursor` set to the returned `next_cursor`
until that value is null. Keep filters and authorization unchanged; `limit` and
`include_components` may change. Cursors are authenticated, opaque, and freeze
the original time range. Invalid, legacy, tampered, or mismatched cursors return
HTTP 400. The API requires a shared Fernet `RESULTS_CURSOR_KEY`; a missing or
invalid key returns HTTP 503 before database access. Key replacement invalidates
existing cursors.

Time bounds use timezone-aware ISO 8601 timestamps: `since` is inclusive and
`until` exclusive. Alternatively, use `window=24h`, `7d`, or `30d`; a window
cannot be combined with explicit bounds. A run-ID query without time bounds
searches all available history for that run. Other unbounded queries default to
seven days. This default is not a limit on queryable history. Available normalized
history can differ from v1; retirement does not backfill older measurements.

Pages read live data. Late arrivals ahead of an already-passed position require
restarting the range. For complete exports, repeat a bounded range after
ingestion settles and replace that range's earlier export. Public fields are
not a unique row identity: identical measurements can legitimately repeat.

## Moving from v1

`/v1/results` is retired and returns HTTP 410 with a link to `/v2/results`.
It does not redirect. Legacy query strings are accepted so callers
receive the migration response rather than a validation error. The aggregate,
timeline, and aggregate-by-dataset routes under `/v1/results/*` remain available.

V2 returns one evaluation per item. The stored value with role `primary` is
exposed as top-level `value`/`unit`; named values with role `component` appear
only in `components` when requested. Response items intentionally do not expose
internal IDs, sample paths, statuses, or evaluation variants.

Clients must exhaust `next_cursor`; the maximum page size is 1000. `since` is
inclusive and `until` exclusive. The only supported evaluation status and
variant are `succeeded` and `default`; use `run_status=succeeded` or
`run_status=partial` to select one allowed parent state.
