# coval-bench

Runner + API for Coval voice-AI benchmarks. Implements:

- A Cloud Run Job that runs STT/TTS providers against a pinned dataset every 30 min and writes results to Cloud SQL.
- A FastAPI service that serves the public results API at `https://benchmarks.coval.ai`.

## What we benchmark

**STT** — providers are scored against two frozen datasets, each run as its
own execution per cycle: [LibriSpeech `test-clean`](https://www.openslr.org/12/)
(`stt-v1`, CC-BY-4.0 read English speech, the easy set) and pipecat's
conversational benchmark data (`stt-v3`, 897 spontaneous voice-agent clips,
the hard set). Metrics: WER, TTFT, TTFS, audio→final latency, RTF. Headline
stats pool both datasets; absolute WER on `stt-v1` runs low (most providers
train on LibriSpeech) and `stt-v3` references are model-generated.

**TTS** — providers are scored on 30 short English customer-service transcripts
(order tracking, appointments, account verification, tech support; Apache-2.0).
Metrics: TTFA, RTF, end-to-end synthesis latency. No reference-audio quality
metric (voices differ by provider).

See `src/coval_bench/datasets/manifests/README.md` for full dataset details.

## Local development

### Tests (offline, no creds, no DB)

```bash
uv sync
uv run pytest -q
```

VCR cassettes + fakes — never hit the network.

### Full stack (Postgres + API + runner image, real provider APIs)

From the repo root:

```bash
cp .env.example .env             # add provider keys you want to exercise
docker compose up -d db          # Postgres on :5432
docker compose run --rm migrate  # alembic upgrade head
docker compose up -d api         # FastAPI on http://localhost:8000

# Trigger a single-item benchmark run (writes to the local Postgres):
docker compose run --rm runner run --smoke --kind tts

# Probe one TTS provider without DB writes:
docker compose run --rm runner tts-smoke \
  --provider cartesia --model sonic-3 --voice <voice-id> --text "hello"
```

The web FE lives in the private `coval-ai/benchmarks-web` repo — run it against `NEXT_PUBLIC_API_URL=http://localhost:8000`.

All env vars are documented in `src/coval_bench/config.py`. Provider keys are optional; tests don't need them.

Normalized observation dual writes are additive, private, and disabled by default.
Set both `BENCHMARK_ARTIFACT_BUCKET` and `NORMALIZED_DUAL_WRITE_ENABLED=true` to
enable the normalized rollout; legacy result writes remain the source of truth.

### Required normalized capture and recovery

`NORMALIZED_CAPTURE_REQUIRED=true` adds a fail-closed durability boundary while
preserving legacy writes. It also requires `NORMALIZED_DUAL_WRITE_ENABLED=true`
and `BENCHMARK_ARTIFACT_BUCKET`. Before provider warmup or Coval run fetches, the
runner verifies the dataset hashes, normalized database schema, and private GCS
create/read/list access.

For each completed provider call, the first acknowledged durable write is an
immutable GCS envelope containing the frozen legacy rows, normalized evaluation
shape, original timestamps and artifact bytes. Database and artifact replay then
uses that envelope. A storage or database backlog leaves an otherwise successful
run `partial` with `normalized capture pending`; recovery promotes it to the
provider-derived sealed status only after all receipts exist and the dashboard
repair enqueue commits. Genuine provider failures remain captured failures.

Inspect one run without exposing payloads:

```bash
coval-bench db capture status --run-id 123 --limit 100
```

Replay a bounded page. Pass a returned `next_cursor` to continue:

```bash
coval-bench db capture recover --run-id 123 --limit 100
coval-bench db capture recover --run-id 123 --limit 100 --cursor 'gs://...'
```

An unsealed run indicates cancellation or process loss before normal finalization.
Replay it only after confirming provider work has stopped, using
`--abandoned`. Conflicting immutable claims fail recovery and require inspection;
the command logs only run IDs and digests, never transcripts or audio.

There is one unavoidable window: a process can die after provider completion but
before its first envelope upload. The run manifest makes the missing expected
capture visible, but bytes that never reached durable storage cannot be replayed.
Envelopes and receipts are retained; remote deletion requires a separately
approved lifecycle policy. Required mode is not enabled by this change and must
be rolled out separately.

### API auth

Public data endpoints need no auth. Two things do.

**Admin routes** (`/v1/admin/models`, `/v1/admin/pricing`, and the PATCH on a model) accept either proof:

- A Clerk session token with the coval org active. This is what the web admin page sends.
- A Google identity token for an email in `ADMIN_GOOGLE_EMAILS`. In prod that list is `local.dev_members` in benchmark-infra `envs/prod/humans.tf`, so every dev there can drive the admin API from a shell:

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  "$API_URL"/v1/admin/models
```

The token lasts an hour and needs only a normal `gcloud auth login`. Model history records the caller's subject and email either way. A 401 means the token could not be verified; a 403 means it verified but the account is not staff.

**Early-access models** are stripped from every data endpoint unless the request carries a Clerk session token. The coval org sees everything, a partner org sees what `CLERK_ORG_PROVIDERS` or `CLERK_ORG_EXCLUSIVE` names for it, and anything else gets the public view. The response says which case applied in `X-EA-Token-Status`.

### Normalized read-index benchmark

With Docker Postgres running, compare the baseline and the two candidate indexes
against one million synthetic metric rows and a pagination-sized result limit:

```bash
docker compose up -d db
cd runner
uv run python scripts/benchmark_normalized_queries.py --rows 1000000 --result-limit 1000
uv run python scripts/benchmark_normalized_queries.py --rows 1000000 --result-limit 1000 --candidate-indexes
uv run python scripts/benchmark_normalized_queries.py --rows 1000000 --result-limit 100000 --candidate-indexes
```

In a local PostgreSQL 16 run, dashboard series measured about 2.63 ms for
legacy, 3.29 ms for normalized baseline, and 2.86 ms with the composite series
index. At a 1,000-row limit, normalized recent results improved from about 23.3
ms to 6.3 ms with the observation index. At 100,000 rows, normalized recent
results measured about 154.0 ms and PostgreSQL ignored that index; this is the
negative/control case. The benchmarks web app currently requests
`/v1/results/aggregates`, not `/v1/results`, so the observation index prepares
for a future normalized paginated-results cutover; the series index maps to the
current dashboard request shape.

Apache-2.0.

### Metric catalog cutover runbook

The application rollout precedes the destructive schema cleanup. Deploy the
ID-only writer and catalog-join readers, then verify deployed consumers and the
normalized readiness report for the agreed rollback period. Preserve the
current normalized dashboard read configuration during this verification. The legacy
`results` and `results_by_bucket` paths remain available during rollback.

The default database boot migration is capped at revision `20261005_0043`.
The cleanup revision is applied separately after the deployed-consumer and
rollback-period gates pass:

```bash
uv run coval-bench db migrate --revision 20261007_0044
uv run alembic -x allow_metric_code_cleanup=true upgrade 20261007_0044
```

Deploy the artifact containing revision `20261007_0044` to every consumer
before applying it, so later restarts recognize the installed Alembic revision.
Never use an implicit head upgrade for this cutover. To roll back the
application, restore the previous interface first, then run
`uv run alembic downgrade 20261005_0043` before deploying an older application
binary.
Record the deployed runner, API,
dashboard, and maintenance-job versions with the readiness report.
