# Contributing

Contributions are welcome. Please open an issue first to discuss significant changes. All code must pass `ruff`, `mypy --strict`, and `pytest` before merging. By submitting a pull request you agree your contribution is licensed under Apache-2.0.

## Updating your model's price

The prices shown on [benchmarks.coval.ai](https://benchmarks.coval.ai) are served from the benchmarks database — `benchmarks_v2.pricing_rates`, an append-only log in which every rate ever in force stays on record — and are maintained by Coval staff through the admin page, so a change takes effect without a deploy. The JSON ratesheets under [`runner/src/coval_bench/registries/pricing/data/<benchmark>/`](runner/src/coval_bench/registries/pricing/data/) (`tts/`, `stt/`) are the open, reviewable mirror of that log and its seed: they are how anyone can check where a figure came from, and a pull request against them is how a provider proposes a change.

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

2. Use your *native* published unit, and write `price_usd` as a string so the decimal is exact. The API normalizes server-side to the one figure the site compares models on — $ per 1M characters for TTS, $ per 1,000 minutes of input audio for STT — never across the two, which would take an assumed speaking rate:

   | Benchmark | Units | Normalized to |
   |---|---|---|
   | TTS | `per_1m_chars`, `per_1k_chars`, `per_char` | $ / 1M characters |
   | STT | `per_minute`, `per_hour`, `per_second_audio_in` | $ / 1,000 minutes |
   | TTS | `per_second_audio_out` | nothing — see below |

   `per_second_audio_out` bills synthesized audio rather than characters sent, so it has no character equivalence without assuming a speaking rate. A rate in it is served and shown in its own unit, but it is left out of the $ / 1M characters figure and so out of the comparison charts. Prefer a character unit if you publish one.

3. Price the tier we benchmark, and name it in `notes`: for STT that is your streaming/real-time pay-as-you-go rate. If you publish no pay-as-you-go rate at all, use the *marginal* rate on your entry-level paid plan — the add-on or overage price of usage past what the plan includes — and say which plan it is. `source_url` must point at the public page that prints the figure, and `effective_from` is the date the rate took effect. Do not submit a rate computed from a plan's price divided by its included allowance, or from a credit rate we would have to look up elsewhere: if your public pricing does not state a usage rate, leave the model unpriced until it does.

4. Open a PR. CI validates the schema and that every entry matches a registered model; a model without a pricing entry simply shows no price on the site — never submit an estimated rate.

Once merged, `coval-bench pricing sync` records the new entry into the log (staff can also enter it directly in the admin page while the PR is open). The two never fight: a figure the log has held once is not recorded again by the sync, so a correction made in the admin page is not undone by the files, and the files keep one entry per model — the rate in force today — while the log keeps them all. When a provider stops publishing a usage rate, staff record a *delisting* (no known public rate from a given day) in the admin page; remove the entry from the ratesheet in the same PR so the mirror agrees.

The site reads the log live and caches its pages for about an hour, so a recorded change is public within the hour — no deploy involved.
