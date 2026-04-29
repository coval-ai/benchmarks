# coval-bench

Runner + API for Coval voice-AI benchmarks. Implements:

- A Cloud Run Job that runs STT/TTS providers against a pinned dataset every 30 min and writes results to Cloud SQL.
- A FastAPI service that serves the public results API at `https://benchmarks.coval.ai`.

## Local development

```bash
uv sync
uv run python -m coval_bench --help
uv run pytest -q
```

Provider API keys are loaded from environment variables (or a `.env` file) — see `src/coval_bench/config.py` for the full list. All keys are optional locally; tests use VCR cassettes and never hit the network.

Apache-2.0.
