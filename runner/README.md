# coval-bench

Runner + API for Coval voice-AI benchmarks. Implements:

- A Cloud Run Job that runs STT/TTS providers against a pinned dataset every 30 min and writes results to Cloud SQL.
- A FastAPI service that serves the public results API at `https://benchmarks.coval.ai`.

## What we benchmark

**STT** — providers are scored against a frozen 50-utterance sample of
[LibriSpeech `test-clean`](https://www.openslr.org/12/) (CC-BY-4.0, English read
speech, 2.0–15.0 s utterances). Selection is round-robin by speaker in lex
order, so all 40 source-pool speakers (20 F / 20 M) contribute, with the
lex-first 10 providing 2 utterances each (deterministic per ADR-020). Metrics:
WER, TTFT, audio→final latency, RTF. Total audio per run ≈ 5–6 min. Absolute
WER is artificially low (most providers train on LibriSpeech) — the value is
in latency and per-provider regression-over-time.

**STT v2** — optional frozen **60-clip** MInDS-14–style US banking customer-service
audio (`DATASET_ID=stt-v2`); see `docs/methodology.md` and
`src/coval_bench/datasets/manifests/README.md`.

**TTS** — providers are scored on 30 short English customer-service transcripts
(order tracking, appointments, account verification, tech support — Coval-internal,
not redistributable). Metrics: TTFA, RTF, end-to-end synthesis latency. No
reference-audio quality metric (voices differ by provider).

See `src/coval_bench/datasets/manifests/README.md` for full dataset details and
`tmp/DECISIONS.md` ADR-020 for the rationale.

## Local development

```bash
uv sync
uv run python -m coval_bench --help
uv run pytest -q
```

Tests use VCR cassettes + fakes and never hit the network. Provider API keys are
loaded from environment variables (or a `.env` file) — see
`src/coval_bench/config.py` for the full list. All keys are optional locally.

### Full stack (Postgres + API + runner image, real provider APIs)

From the repo root:

```bash
# Create a `.env` in the repo root with POSTGRES_* and any provider keys you need
# (variable names in `src/coval_bench/config.py`).
docker compose up -d db          # Postgres on :5432
# `migrate` and `runner` use Compose profiles — include them or Compose may not see the service:
docker compose --profile init run --rm migrate   # alembic upgrade head
docker compose up -d api         # FastAPI on http://localhost:8000

# Trigger a single-item benchmark run (writes to the local Postgres):
docker compose --profile tools run --rm runner coval-bench run --smoke --kind tts

# Probe one TTS provider without DB writes:
docker compose --profile tools run --rm runner coval-bench tts-smoke \
  --provider cartesia --model sonic-3 --voice <voice-id> --text "hello"

# Probe one STT provider without DB writes (no Postgres required):
docker compose --profile tools run --rm runner coval-bench stt-preview \
  --provider deepgram --model nova-2
```

The web FE (`web/`) is a Next.js app — run `pnpm dev` against `NEXT_PUBLIC_API_URL=http://localhost:8000`.

All env vars are documented in `src/coval_bench/config.py`. Provider keys are optional for offline tests.

Apache-2.0.
