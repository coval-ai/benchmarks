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

# Optional: write a portable JSONL artifact for the run:
docker compose run --rm -e RUN_ARTIFACT_DIR=/tmp/artifacts runner \
  coval-bench run --smoke --kind tts

# Probe one TTS provider without DB writes:
docker compose run --rm runner coval-bench tts-smoke \
  --provider cartesia --model sonic-3 --voice <voice-id> --text "hello"
```

The web FE (`web/`) is a Next.js app — run `pnpm dev` against `NEXT_PUBLIC_API_URL=http://localhost:8000`.

All env vars are documented in `src/coval_bench/config.py`. Provider keys are optional; tests don't need them.

Apache-2.0.
