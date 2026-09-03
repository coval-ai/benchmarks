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
enable the STT/TTS rollout; legacy result writes remain the source of truth.

### Normalized-storage backfill operator runbook

Run the production backfill as a dry run first. This command adds no `--apply`
flag, so it makes no normalized-storage writes; the Cloud Run timeout override
applies only to this execution.

```bash
gcloud run jobs execute benchmarks-runner \
  --project=coval-benchmarks-prod \
  --region=us-east1 \
  --task-timeout=48h \
  --wait \
  --args='migrate,backfill-normalized-storage,--min-result-id=1,--batch-size=100'
```

The dry run resolves one database UTC timestamp, freezes the preceding inclusive
start/exclusive end 168-hour window, and reports its exact UTC start, end, and
maximum result ID. The 48-hour timeout leaves a safe margin beyond the observed
17h31m48s dry run. Record all three values for any separately authorized apply:

```bash
gcloud run jobs execute benchmarks-runner \
  --project=coval-benchmarks-prod \
  --region=us-east1 \
  --task-timeout=48h \
  --wait \
  --args='migrate,backfill-normalized-storage,--min-result-id=1,--window-start=2026-08-20T00:00:00Z,--window-end=2026-08-27T00:00:00Z,--max-result-id=N,--batch-size=100,--apply'
```

Progress events are JSON on stderr; the final JSON report remains stdout's final line.

`--batch-size` is a run-ID read page and also bounds apply-plan transaction
batches. Larger pages reduce round trips, but increase memory, transaction
duration, rollback and retry work, artifact exposure, and the spacing between
safe checkpoints. A local PostgreSQL 16 dry-run comparison over 5,000 synthetic
STT runs with one result each measured 1.15--1.26 s / 95.2 MB peak RSS at 25,
1.03--1.06 s / 95.6 MB at 100, and 0.92--0.98 s / 97.6--97.8 MB at 400 after a
warm-up run. The small synthetic gain does not model production row fanout,
artifact handling, transaction duration, or rollback exposure, so the default
remains 100. Compare candidate sizes in production only with sequential dry
runs over the same frozen window, using each phase's elapsed time and throughput
together with Cloud Run peak memory.

### Normalized S2S-storage backfill operator runbook

S2S has a separate, database-only backfill: it never instantiates GCS and it
never writes artifacts, preprocessing artifacts, evaluation inputs, or metric
artifacts. Start with this read-only dry run; it freezes and reports the S2S
maximum result ID only. Progress JSON is written to stderr and the final report
is stdout's final line.

```bash
gcloud run jobs execute benchmarks-runner \
  --project=coval-benchmarks-prod \
  --region=us-east1 \
  --task-timeout=24h \
  --wait \
  --args='migrate,backfill-normalized-s2s-storage,--min-result-id=1,--batch-size=100'
```

An apply requires explicit approval and the frozen maximum from the dry run:
add `--max-result-id=N,--apply`. Progress phases are `qualifying_run_count`,
`source_reconciliation`, `post_write_verification`, `public_parity`, and
`rollup_verification`, bracketed by the `operation` phase. Events include both
completed and durably committed run/result checkpoints. On failure, restart the
same frozen window: deterministic backfill IDs and exact natural-key
reconciliation make committed pages idempotent.

The default is 100 run IDs/page. A page bounds planning memory and one top-level
write transaction; a failed page rolls back its observations, evaluations,
values, and bucket refreshes together while earlier pages remain committed.
Larger pages reduce query and checkpoint overhead but increase memory,
transaction duration, rollback work, and the distance between durable
checkpoints. Each affected scheduled bucket is refreshed under the existing
per-bucket advisory transaction lock.

The final verification is independent of source-pass counters. It freshly
re-plans the frozen complete-run cohort, compares the complete normalized S2S
observation population, compares public legacy values with normalized primary
values exactly, and compares every stored rollup field (including `value_sum`)
with a fresh aggregate. Bounded mismatch details accompany exact mismatch
counts. `backfill_complete` means this frozen migration window reconciles; it
does not claim global cutover readiness.

The local helper creates, migrates, seeds, benchmarks, and removes a uniquely
named disposable database. Its default seed uses 450 runs, 50 conversations per
run, and one to three metrics per conversation; each batch size runs in a fresh
child process so peak RSS is comparable. Query duration is cumulative
client-observed SQL execution time. The administrative URL must be loopback and
its role must be allowed to create and drop databases.

```bash
uv run python scripts/benchmark_normalized_s2s_backfill.py \
  --admin-database-url postgresql://postgres:postgres@127.0.0.1:5432/postgres \
  --batch-sizes 25,100,400 \
  --warmups 1 \
  --iterations 1 \
  --format markdown
```

No production-size measurement is recorded here yet, so 100 remains the
conservative default. Roll out the read-only dry run, obtain explicit approval
for the frozen apply, then use the separate readiness/cutover process. Do not
enable normalized dashboard reads, broaden the STT/TTS backfill, or combine
this runbook with the asynchronous-runner follow-up.

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
