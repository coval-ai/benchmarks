# Contributing

Contributions are welcome. Please open an issue first to discuss significant changes. All code must pass `ruff`, `mypy --strict`, and `pytest` before merging. By submitting a pull request you agree your contribution is licensed under Apache-2.0.

## Updating your model's price

The prices shown on [benchmarks.coval.ai](https://benchmarks.coval.ai) live in this repo, one JSON file per provider per benchmark under [`runner/src/coval_bench/registries/pricing/data/<benchmark>/`](runner/src/coval_bench/registries/pricing/data/) (`tts/`, `stt/`). Providers are encouraged to keep their own rates current:

1. Edit (or add) `<benchmark>/<your-provider>.json` — the file may only contain entries for your own provider on that benchmark, keyed by the same `(benchmark, provider, model)` as [`registries/models.py`](runner/src/coval_bench/registries/models.py):

   ```json
   [
     {
       "benchmark": "STT",
       "provider": "deepgram",
       "model": "nova-3",
       "unit": "per_minute",
       "price_usd": "0.0048",
       "effective_from": "2026-08-24",
       "source_url": "https://deepgram.com/pricing",
       "notes": "Pay As You Go streaming rate."
     }
   ]
   ```

2. Use your *native* published unit — characters for TTS (`per_1m_chars`, `per_1k_chars`, `per_char`, `per_second_audio_out`), duration of input audio for STT (`per_minute`, `per_hour`, `per_second_audio_in`). The API normalizes server-side to the figure the site shows: $ per 1M characters for TTS, $ per 1,000 minutes for STT — never across the two, which would take an assumed speaking rate. Write `price_usd` as a string so the decimal is exact. `source_url` must point at your public pricing page, and `effective_from` is the date the rate took effect. For STT, use the rate of the tier we benchmark: streaming/real-time pay-as-you-go, and say so in `notes`.
3. Open a PR. CI validates the schema and that every entry matches a registered model; a model without a pricing entry simply shows no price on the site — never submit an estimated rate.

The new price appears on the site with the next deploy — no code change needed.
