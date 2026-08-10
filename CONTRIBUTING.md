# Contributing

Contributions are welcome. Please open an issue first to discuss significant changes. All code must pass `ruff`, `mypy --strict`, and `pytest` before merging. By submitting a pull request you agree your contribution is licensed under Apache-2.0.

## Updating your model's price

The prices shown on [benchmarks.coval.ai](https://benchmarks.coval.ai) live in this repo, one JSON file per provider under [`runner/src/coval_bench/registries/pricing/data/tts/`](runner/src/coval_bench/registries/pricing/data/tts/). Providers are encouraged to keep their own rates current:

1. Edit (or add) `<your-provider>.json` — the file may only contain entries for your own provider, keyed by the same `(benchmark, provider, model)` as [`registries/models.py`](runner/src/coval_bench/registries/models.py):

   ```json
   [
     {
       "benchmark": "TTS",
       "provider": "deepgram",
       "model": "aura-2-thalia-en",
       "unit": "per_1k_chars",
       "price_usd": "0.030",
       "effective_from": "2026-08-10",
       "source_url": "https://deepgram.com/pricing",
       "notes": "Pay As You Go rate."
     }
   ]
   ```

2. Use your *native* published unit (`per_1m_chars`, `per_1k_chars`, `per_char`, or `per_second_audio_out`); the API normalizes to $ per 1M characters server-side. Write `price_usd` as a string so the decimal is exact. `source_url` must point at your public pricing page, and `effective_from` is the date the rate took effect.
3. Open a PR. CI validates the schema and that every entry matches a registered model; a model without a pricing entry simply shows no price on the site — never submit an estimated rate.

The new price appears on the site with the next deploy — no code change needed.
