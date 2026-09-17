# Results API v2

`GET /v2/results` returns one item per metric evaluation, including its
observation, metric version, and variant. See the API service's `/docs` or
`/openapi.json` for the complete response schema and query parameters.

## Examples

Primary values for one run and metric:

```text
/v2/results?run_id=789&metric_type=TTFA
```

Include named components and select a metric version and variant:

```text
/v2/results?run_id=789&metric_type=TTFA&metric_version=v1&evaluation_variant=default&include_components=true
```

Export a time range across runs:

```text
/v2/results?benchmark=TTS&since=2026-09-01T00:00:00Z&until=2026-09-08T00:00:00Z&limit=1000
```

## Response and filters

- The response contains `results` and a nullable `next_cursor`.
- Each result has its own `evaluation_id`, `observation_id`, and `run_id`, plus
  metric, status, provider, model, dataset, sample, and capture-time fields.
- `value` and `unit` describe the stored primary value. Both are null when it is
  absent; a component is never substituted. Values are returned without conversion.
- `components` is omitted by default. With `include_components=true`, it maps
  component names to `{value, unit}` pairs; no components produces `{}`.
- Version and variant filters are exact matches. Omitting them returns all
  versions and variants as separate evaluations.
- Evaluations default to `succeeded`; parent runs default to `succeeded` and
  `partial`. These filters are independent. To include failed evaluations from
  any run state, use `evaluation_status=failed&run_status=all`.
- Filters combine with AND. Model access restrictions apply to every page.

## Pagination and time ranges

The page size defaults to 100 evaluations, with a maximum of 1000. Results sort
by capture time descending, then evaluation ID descending. Components do not
change page membership or consume page slots.

Repeat the original request with its `cursor` set to the returned `next_cursor`
until that value is null. Keep filters and authorization unchanged; `limit` and
`include_components` may change. Cursors freeze the original time range. Invalid
cursors or mismatched filters/access return HTTP 400.

Time bounds use timezone-aware ISO 8601 timestamps: `since` is inclusive and
`until` exclusive. Alternatively, use `window=24h`, `7d`, or `30d`; a window
cannot be combined with explicit bounds. A run-ID query without time bounds
searches all available history for that run. Other unbounded queries default to
seven days. This default is not a limit on queryable history.

Pages read live data. Late arrivals or status changes can affect later pages;
newly eligible results ahead of an already-passed position require restarting
the range. For complete exports, repeat a bounded range after ingestion settles
and deduplicate by `evaluation_id`. Historical coverage can differ from v1.

## Moving from v1

`/v1/results` remains available. V2 changes the response shape and uses explicit
run/evaluation status filters in place of `include_failed`. Update response
parsing and follow pagination when switching to v2.
