<div align="center">

# 🎙️ Coval Benchmarks

### Voice AI benchmarks in real-world conditions.

**Open-source, always-on leaderboards for speech-to-text, text-to-speech, and speech-to-speech models.**

[**benchmarks.coval.ai →**](https://benchmarks.coval.ai)

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](runner/pyproject.toml)
[![Live leaderboards](https://img.shields.io/badge/leaderboards-live-1db098.svg)](https://benchmarks.coval.ai)
[![Methodology](https://img.shields.io/badge/methodology-open-8A2BE2.svg)](docs/methodology.md)

</div>

---

Most voice AI benchmarks are a one-time snapshot on clean studio audio. We run ours **every 30 minutes**, against **almost 40 providers**, on audio that sounds like real calls: accents, background noise, echoey rooms, far-away mics, clipped signals, and phone-line compression.

Every number on the site comes from this repository. You can rerun it with your own API keys.

## ⚡ What's on the board

| | Benchmark | What we measure |
|---|---|---|
| 🗣️ | **Text-to-Speech** | **TTFA** (time to first *audible* sample, including front-loaded silence), real-time factor, end-to-end synthesis latency, and **WER** on the synthesized audio to check that the speech is intelligible and complete |
| 👂 | **Speech-to-Text** | **WER** after Whisper's `EnglishTextNormalizer`, **TTFT** (first partial), **TTFS** (time to final segment, from a shared VAD end-of-speech anchor), audio→final latency, real-time factor |
| 🔁 | **Speech-to-Speech** | **Voice-to-voice latency** for realtime models, measured in simulated caller conversations across personas that range from easy to extra-hard |
| 🥊 | **Voice Arena** | Blind A/B votes on which voice sounds more natural, fit with a Bradley-Terry model extended for ties (Davidson) and bootstrapped confidence intervals |

Leaderboards rank by the **median** and show the spread from p25 to p99, so a model with a fast median and a slow tail can't hide it.

## 🌪️ Audio that fights back

Most STT providers train on clean studio speech, so nearly all of them look good on it. We test clean speech as a control, then test the conditions that break things in production:

| Dataset | Condition |
|---|---|
| `clean` | Clean read speech (FLEURS), the undistorted control |
| `accent` | 845 clips across demographic accents |
| `noisegap` | Intermittent background-noise bursts |
| `reverb` | Room echo |
| `farfield` | Mic at a distance |
| `clipping` | Peak distortion |
| `phonecodec` | Telephony compression |
| `stt-v3` | 897 spontaneous voice-agent turns, with fragments and fillers |
| `tts-v1` | 30 short customer-service prompts: order tracking, appointments, account verification, tech support |

Every clip is loudness-normalized to −20 dBFS RMS and **pinned to a SHA-256 hash**. If the audio changes, the runner refuses to score it.

## 🔬 Built so you can trust it

- **Exact model versions.** Every entry uses the provider's versioned identifier (`nova-3`, `gpt-4o-mini-tts`, `sonic-3`), never a floating alias.
- **Apples-to-apples runs.** Each run draws one random sample, and every model is scored on that same draw. Over time, the random draws cover the whole dataset.
- **The model, not the network.** TCP, TLS, and protocol handshakes are excluded from every provider's latency the same way. HTTP clients are pre-warmed over HTTP/2, and dedicated endpoints are scaled up before timing starts.
- **Honest exclusions.** If a provider's wire protocol would make a metric measure something other than the model (a fixed emission tick, say), we withhold that metric instead of publishing a misleading number.
- **Methodology changes are dated, not hidden.** When scoring changes, results from before and after are marked incomparable rather than quietly blended. The [changelog is in the methodology](docs/methodology.md).
- **Pinned scoring libraries.** `jiwer` and `whisper-normalizer` are locked in `uv.lock`.

Read the full reproducibility contract in [`docs/methodology.md`](docs/methodology.md).

## 🧩 Providers

AssemblyAI · Alibaba · Azure · Baseten · Cartesia · Deepdub · Deepgram · ElevenLabs · Fish Audio · Fluxions · Gemini · Gladia · Google · Gradium · Groq · Hume · Inworld · LMNT · MiniMax · Mistral · Modulate · Murf · Nari · OpenAI · Palabra · Rev.ai · Rime · Smallest · Soniox · Speechify · Speechmatics · Together AI · xAI · Zoom · *and more*

Missing a provider? Adapters live in [`runner/src/coval_bench/providers/`](runner/src/coval_bench/providers/), one file each. [Open an issue](https://github.com/coval-ai/benchmarks/issues) and we'll talk about adding it.

## 🚀 Run it yourself

Offline tests need no keys, database, or network:

```bash
cd runner
uv sync
uv run pytest -q
```

To run the full stack (Postgres, API, and runner) against real providers:

```bash
cp .env.example .env               # add keys for the providers you want
docker compose up -d db
docker compose run --rm migrate
docker compose up -d api           # http://localhost:8000
docker compose run --rm runner run --smoke --kind tts
```

To check one model without writing to the database:

```bash
docker compose run --rm runner tts-smoke \
  --provider cartesia --model sonic-3 --voice <voice-id> --text "hello"
```

See [`runner/README.md`](runner/README.md) for the CLI, the API, and the other runbooks.

## 🏗️ How it works

```
 Scheduler       ──▶ Runner (Cloud Run Job) ──▶ Provider APIs
   every 30 min        │  sample dataset, verify SHA-256,
   + daily suite       │  stream audio / text, time it, score it
                       ▼
                   Postgres ──▶ FastAPI (/v1/leaderboard, /v1/results, …) ──▶ benchmarks.coval.ai
```

## 🤝 Contributing

Contributions are welcome. Please open an issue before making a big change. All code must pass `ruff`, `mypy --strict`, and `pytest`. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## ⚖️ Caveats and license

This is a research benchmark. Results reflect specific pinned datasets, model versions, and scoring pipelines, and may not generalize to your production traffic. Our workers run in `us-east-1`, so providers serving from other regions carry that round-trip. Latency and per-provider drift over time are the most informative signals. Full caveats are in [`docs/methodology.md`](docs/methodology.md#caveats).

Licensed under [Apache-2.0](LICENSE) (warranty disclaimer in §7). Third-party provider SDKs are covered by their own licenses.

<div align="center">

Built by [Coval](https://coval.dev), which does simulation and evaluation for voice and chat agents.

</div>
