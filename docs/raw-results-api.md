# Raw results API

`GET /v2/results` reads normalized metric evaluations. Each result represents one
evaluation of one observation, including its metric version and variant.
`GET /v1/results` remains available with its existing response during migration.

## Response

```json
{
  "results": [
    {
      "evaluation_id": "99408542-2ff3-4ceb-a019-d24566e9a17c",
      "observation_id": "2a2307a2-4a32-46d1-b19e-97bc2b40af6d",
      "run_id": 789,
      "provider": "example",
      "model": "example-tts",
      "voice": null,
      "benchmark": "TTS",
      "dataset_id": "tts-v1",
      "sample_id": "sample-1",
      "captured_at": "2026-09-15T12:00:00Z",
      "metric_type": "TTFA",
      "metric_version": "v1",
      "evaluation_variant": "default",
      "evaluation_status": "succeeded",
      "run_status": "succeeded",
      "value": 120,
      "unit": "milliseconds",
      "components": {
        "roundtrip": { "value": 100, "unit": "milliseconds" },
        "leading_silence": { "value": 20, "unit": "milliseconds" }
      }
    }
  ],
  "next_cursor": null
}
```

This example includes components. By default, `components` is omitted. Request
`include_components=true` to receive a map of component names to stored values
and units; evaluations with no components return `{}`.

The top-level `value` and `unit` come from the stored primary value's **role**,
regardless of its name. Both are `null` when that value is absent. A component is
never substituted for it. Values and units are returned as stored, without
conversion or recalculation.

One observation can have several evaluations. Each keeps its own ID, metric,
version, and variant. The API does not choose a latest version or combine values
across versions. Provider errors, transcripts, private artifact locations, and
provider metadata are not included.

## Filters

All supplied filters are combined with AND. Enum values are case-sensitive.

| Parameter | Behavior |
| --- | --- |
| `run_id` | Positive run ID. Without a time filter, searches all available history for that run. |
| `provider`, `model` | Exact stored provider/model names. |
| `benchmark` | `STT`, `TTS`, `S2S`, or `LLM`. |
| `dataset` | Exact observation dataset ID, such as `tts-v1`. |
| `metric_type` | Metric catalog code, such as `TTFA` or `WER`. |
| `metric_version`, `evaluation_variant` | Exact matches when supplied; otherwise all stored versions/variants. |
| `evaluation_status` | `succeeded` by default; also `queued`, `running`, `failed`, or `all`. |
| `run_status` | Omitted: `succeeded` and `partial`. Also accepts an individual `running`, `succeeded`, `partial`, `failed`, or `all`. |
| `window` | `24h`, `7d`, or `30d`. Defaults to `7d` if no run ID or time bounds are supplied. |
| `since`, `until` | Timezone-aware ISO 8601 bounds on observation `captured_at`: `since` inclusive, `until` exclusive. Cannot be combined with `window`. |
| `include_components` | `false` by default. Adds components to the same evaluations. |
| `limit` | Evaluations per page: 1–1000, default 100. |
| `cursor` | Opaque continuation token from the preceding response. |

Run and evaluation status are independent. For example,
`evaluation_status=failed` still restricts parent runs to succeeded/partial
unless `run_status` is also supplied. Use `run_status=all` to include evaluations
from every parent-run state. V2 uses these explicit filters instead of v1's
`include_failed` parameter.

Existing model visibility rules apply to every page, before the page limit.
Use the same authorization context throughout a traversal. Cursors do not grant
access to models.

## Examples

Primary values for one run and metric:

```text
/v2/results?run_id=789&metric_type=TTFA
```

The same evaluations with components, restricted to one version and variant:

```text
/v2/results?run_id=789&metric_type=TTFA&metric_version=v1&evaluation_variant=default&include_components=true
```

Failed evaluations from any run state:

```text
/v2/results?evaluation_status=failed&run_status=all&window=24h
```

A bounded export across runs:

```text
/v2/results?benchmark=TTS&since=2026-09-01T00:00:00Z&until=2026-09-08T00:00:00Z&limit=1000
```

Repeat the export request with its `cursor` parameter set to `next_cursor`,
URL-encoded by your HTTP client. Continue until `next_cursor` is `null`.

## Pagination and history

Results are ordered by capture time descending, then evaluation UUID descending
to break ties. Pagination counts evaluations. Components never consume page
slots or split an evaluation across pages.

Keep row filters and authorization context unchanged when continuing a request.
The cursor freezes effective time bounds so a relative window does not move
between pages. You may change `limit` or `include_components`. Invalid cursors
or cursors used with different filters/visibility return HTTP 400.

Pages read current database state. They do not form a database snapshot. Static
eligible evaluations appear once during a complete traversal. Late inserts,
backfills, and status changes can make additional rows eligible: rows after the
current cursor position may appear on later pages; rows before a position you
have already passed require a fresh traversal. For exports that must include
late arrivals, repeat a bounded range after ingestion has settled and deduplicate
by `evaluation_id`.

V2 returns available normalized history. There is no 30-day cutoff and no
implicit union with legacy results. An empty historical range does not prove
that the legacy store is empty or that its data has been backfilled. Storage
retention and archival policies are separate from the query defaults.

## Release and rollback

Before production release:

1. Apply schema migration `0036` before deploying v2. The reader uses stored
   metric IDs and resolves historical null IDs through the metric catalog, so
   the metric-ID backfill does not need to finish first. Reads do not update IDs.
2. Verify the API database role can read the normalized observations, evaluations,
   values, shared runs, and metric catalog used by the query. Existing dashboard
   access alone does not prove access to these source tables.
3. Verify normalized write coverage and the history ranges consumers need.
   Check real query plans and latency for run lookups and bounded exports at
   representative production volume.
4. Deploy the additive v2 route and matching OpenAPI documentation, then publish
   generated client types. The dashboard's normalized-read flag does not select
   this route or change the v1 response.
5. Move callers to v2 deliberately, updating their response parsing and following
   pagination until the cursor is null. Confirm visibility and status filters
   with their actual authorization context.

While v1 remains available, rollback means returning callers to their previous
v1 request and response handling or reverting the additive API release. V1 and
v2 shapes differ, so changing only the URL is insufficient.

This release does not retire legacy writes or readers. Arena provider health and
import deduplication still use legacy results. Arena health requires complete
latest-run success and failure coverage to preserve provider exclusion and
recovery; successful normalized dashboard samples do not establish that.
Legacy-write shutdown, backfill, retention changes, and table removal require
separate work.
