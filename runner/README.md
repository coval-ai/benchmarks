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
docker compose run --rm runner coval-bench run --smoke --kind tts

# Probe one TTS provider without DB writes:
docker compose run --rm runner coval-bench tts-smoke \
  --provider cartesia --model sonic-3 --voice <voice-id> --text "hello"
```

The web FE lives in the private `coval-ai/benchmarks-web` repo — run it against `NEXT_PUBLIC_API_URL=http://localhost:8000`.

All env vars are documented in `src/coval_bench/config.py`. Provider keys are optional; tests don't need them.

Normalized observation dual writes are additive, private, and disabled by default.
Set both `BENCHMARK_ARTIFACT_BUCKET` and `NORMALIZED_DUAL_WRITE_ENABLED=true` to
enable the STT/TTS rollout; legacy result writes remain the source of truth.

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
