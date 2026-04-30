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

Provider API keys are loaded from environment variables (or a `.env` file) — see `src/coval_bench/config.py` for the full list. All keys are optional locally; tests use VCR cassettes and never hit the network.

Apache-2.0.
