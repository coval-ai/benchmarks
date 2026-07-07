# Methodology

A leaderboard number on [benchmarks.coval.ai](https://benchmarks.coval.ai) is a
function of four pinned inputs. Reproducing a number requires reproducing all
four. This document describes each one and where it lives.

## 1. Dataset

**STT benchmark.** A 50-utterance subset of
[LibriSpeech `test-clean`](https://www.openslr.org/12/) (OpenSLR-12,
`CC-BY-4.0`). Selection rule: utterances filtered to the 2.0–15.0 s duration
range, then round-robin by speaker in lexicographic order — all 40 speakers
(20 F / 20 M) contribute, with the lex-first 10 providing 2 utterances each.
The selection is deterministic and reproducible from the raw `test-clean`
download.

**TTS benchmark.** A 30-prompt set of short text inputs. Source and selection
rule are documented in `runner/src/coval_bench/datasets/manifests/tts-v1.json`.

**Per-run sampling.** Each scheduled run draws a random sample of
`dataset_sample_size` items (default 10) from each manifest before any provider
is called; the STT and TTS pools are sampled independently. The sample is drawn
once at the start of the run and shared across every model, so all models are
scored on the identical subset within a run (parity); the draw is independent
across runs, so the full manifest is covered over time. The manifests still
carry the complete 50 / 30 items — sampling only controls how many run each
cycle. Set `DATASET_SAMPLE_SIZE` ≥ the manifest size to run everything.

**SHA pinning.** Every audio file referenced by `stt-v1.json` carries a
`sha256` field. The runner verifies the SHA after fetching from GCS and raises
`DatasetIntegrityError` on mismatch (see `runner/src/coval_bench/datasets/loader.py`).
TTS items are text-only and have no SHA.

**Versioning.** Dataset manifests live at
`runner/src/coval_bench/datasets/manifests/{stt-v1,tts-v1}.json` and carry a
`version` field. Bumping the dataset re-pins by minting a new manifest
(`stt-v2.json` etc.); the old manifest stays for historical reproducibility.

**Rebuilding from scratch.** `runner/scripts/build_dataset.py` downloads the
LibriSpeech archive, applies the selection rule, transcodes to the canonical
audio format, uploads to GCS, and writes the manifest with fresh SHAs. See
`runner/src/coval_bench/datasets/manifests/README.md` for the full procedure.

**Selection rationale.** See [ADR-020](#adr-references) below.

## 2. Provider model versions

Each `(provider, model, voice)` tuple the runner exercises is declared in
`runner/src/coval_bench/runner/config.py` (`DEFAULT_STT_MATRIX` and
`DEFAULT_TTS_MATRIX`). Model identifiers are exact provider strings — not
aliases — wherever the provider exposes a versioned identifier (e.g.
`nova-3`, `flux-general-en`, `gpt-4o-mini-tts`, `aura-2-thalia-en`, `mistv3`).

**Server-side aliases.** Where a provider resolves a name server-side (e.g.
Rime `arcana` → Arcana v3), the resolution is documented inline in
`config.py` next to the matrix entry.

**Disabled entries.** Some entries are intentionally inert (`disabled=True`).
The runner skips them; the API exposes them so that a frontend can grey out
models that are catalogued but not currently producing data. See
[ADR-011](#adr-references) for the policy.

## 3. Normalization pipeline

WER is computed after a deterministic normalization applied to both the
reference transcript and the provider hypothesis: `whisper_normalizer`'s
`EnglishTextNormalizer`, the de facto standard for published WER, wrapped by
`normalize_text` in `runner/src/coval_bench/metrics/wer.py`. It folds spoken
and written forms of numbers, ordinals, dates, currency, and percentages into
a canonical form, expands contractions, drops filler words, maps British
spellings to American, and strips diacritics and punctuation.

**Version constant.** `wer.NORM_VERSION` (currently `"2"`) is bumped on any
behavioural change to the pipeline. The constant is included on every
`WERResult` as `norm_version`, so a result row can be unambiguously
attributed to the pipeline version that produced it.

**Methodology change (2026-07).** Version `"1"` was a hand-rolled pipeline
that corrupted many number forms (e.g. "thirty six" → `3006`, reported in
[issue #218](https://github.com/coval-ai/benchmarks/issues/218)), putting a
WER floor under providers whose inverse text normalization emits digits while
leaving spelled-out-number providers unaffected. It was replaced wholesale
with `EnglishTextNormalizer` per [ADR-021](#adr-references). WER values
before and after the change are **not comparable** — rows are distinguished
by the `runner_sha` on the run.

## 4. Latency metrics (TTFA)

The TTS latency metric is **TTFA (Time-To-First-Audio)**, reported in
milliseconds. It is *perceived* first-audible latency, the goal being to measure 
what an enqueue-and-play client actually waits before it hears sound:

    TTFA = (first audio chunk arrival − synthesis start)
           + leading silence inside the stream before the first audible sample

The arrival term is wall-clock (`time.monotonic`) from the synthesis trigger to
the first non-empty audio chunk. The leading-silence term is computed from the
assembled PCM by `runner/src/coval_bench/metrics/ttfa.py`
(`first_audible_offset_ms`, an RMS-threshold onset detector). Both terms are
combined in one place — `providers/tts/_common.py:finalize_tts_result` — which
every provider routes through. The offset is best-effort: if it cannot be
computed, TTFA degrades to arrival-only and the audio is still kept.

**Methodology change (2026-05).** TTFA previously measured *network arrival
only* (time to first chunk) and ignored any leading silence a provider
front-loads into the stream. As of this change it is the perceived value above.
This shifts every provider's reported TTFA upward by its leading-silence offset
(near-zero for providers that emit audible audio immediately; several hundred ms
for those that front-load silence), so numbers recorded before the change are
**not comparable** with later ones. There is no schema or metric-name change —
the existing `TTFA` metric simply carries the perceived definition from this
point forward, distinguished by the `runner_sha` on the run.

## 5. Library versions

The DP edit-distance computation is delegated to `jiwer` and text
normalization to `whisper-normalizer`. Both are pinned in `runner/uv.lock`
(currently `jiwer == 4.0.0`, `whisper-normalizer == 0.1.12`). `numpy` is also
locked. Every `uv.lock` change is reviewed against the SPDX license-set
policy before merge.

## 5. Latency measurement convention

TTFT (STT) and TTFA (TTS) measure model inference time. Pre-t0 deployment
setup — TCP, TLS, protocol handshake, optional session-setup RTT — is
excluded for every provider. The rule is applied uniformly across cohorts;
the magnitude of what each cohort excludes varies by deployment architecture,
but the rule is the same.

| Cohort | What is excluded from t0 |
|---|---|
| WS streaming (Deepgram, AssemblyAI, ElevenLabs Scribe, Speechmatics, Gradium STT, Cartesia, Deepgram aura-2, Rime, Gradium TTS, Hume) | TLS + WS upgrade + optional session-setup RTT (~50–200 ms). Handshake naturally completes inside `measure_ttft` / `synthesize` before t0. |
| HTTP TTS (ElevenLabs HTTP, OpenAI `gpt-4o-mini-tts`) | TLS + TCP via a shared `httpx.AsyncClient` pre-warmed once per run (~80–200 ms). |

HTTP-pool warming lives in `runner/src/coval_bench/providers/_http_session.py`.
Providers opt in by overriding `Provider.warmup()` in `providers/base.py`; the
orchestrator invokes warmup on every enabled provider class before the
dataset loop runs and tears down the pool in the run's `finally` block.

### HTTP/2 for the HTTP TTS cohort

The shared pool uses HTTP/2 so that every concurrent request to a host
multiplexes over a single connection. The connection is opened once by
`warmup()`, and the dataset loop — run at a concurrency of 8 — reuses it, so
TCP+TLS is paid once and excluded from every TTFA row rather than just the
first. `http2=True` and the pool limits are set on `TimedTransport`, not on
`AsyncClient`: httpx ignores both when a custom transport is supplied. Both
hosts (`api.openai.com`, `api.elevenlabs.io`) negotiate HTTP/2; under 8-way
concurrency the pool stays at one socket per host. Requires the `h2` package
(`httpx[http2]`) — a build without it raises at client construction rather
than downgrading silently.

The protocol is chosen by TLS ALPN at connect time. It can negotiate HTTP/1.1
instead — a TLS-terminating proxy or middlebox, an `HTTPS_PROXY` tunnel, or an
HTTP/1.1-only endpoint. On HTTP/1.1 there is no multiplexing, so concurrent
requests open additional connections that `warmup()` did not warm, and TTFA
reabsorbs TCP+TLS for those rows. `warmup()` logs `http_version` and emits a
`*_prewarm_no_http2` warning on any non-h2 connection, and every HTTP TTS
result row records the negotiated `http_version`, so HTTP/1.1 rows can be
filtered out of analysis rather than silently inflating the published TTFA.

**Follow-ups, not yet in this implementation:**

- *NVCF gRPC hosted cohort (Nvidia Nemotron, Magpie).* These providers ship a
  per-channel warmup probe that excludes TLS + gRPC channel + NVCF GPU
  dispatch (~500–1000 ms) before t0. The probe uses the same
  `Provider.warmup()` hook documented above and will be wired in when
  `nvidia_hosted.py` lands on `main`; the cohort table will gain a third row
  then.
- *Connect-time isolation at the transport layer.* `TimedTransport` records
  `submit-to-headers` time per request, surfaced on each HTTP TTS result row
  as `submit_to_headers_ms` — small and stable values confirm the pool stayed
  warm, while a spike flags a row whose connection reconnected mid-run and
  reabsorbed connect time. It does not isolate TCP + TLS time from server
  processing time. True per-call isolation would require subclassing
  `httpcore.AsyncNetworkBackend` (one layer below httpx's transport).
  Deferred — pool reuse achieves the same end result by construction, and the
  recorded interval is sufficient to detect and filter pool eviction.

## Reproducing a result

To reproduce a single `(provider, model, voice, metric)` cell:

1. Clone the repo at the commit SHA from the leaderboard row
   (`/v1/runs/<id>` exposes `runner_sha`).
2. `uv sync` inside `runner/`.
3. Set the relevant provider API key in `.env`.
4. `coval-bench run --kind stt --providers <provider> --models <model>`
   (or analogous for TTS). The runner downloads the same dataset audio from
   GCS, verifies the SHA, runs against the same model identifier, and applies
   the same normalization pipeline.

Numbers should match the leaderboard within provider-side variance (latency
metrics in particular are session-dependent).

## ADR references

- ADR-011 — Provider scope and the `disabled` flag policy.
- ADR-020 — Dataset selection rule (round-robin by speaker, 2.0–15.0 s window).
- ADR-021 — WER text normalization delegated to `whisper_normalizer`'s
  `EnglishTextNormalizer`.

ADR rationale is referenced inline in the relevant source files and READMEs
where the decision context is load-bearing.

## Caveats

- Absolute WER on LibriSpeech `test-clean` is artificially low for most
  providers because the corpus is widely included in training data.
  Latency and per-provider drift over time are the more informative
  signals.
- This is a research benchmark. Results reflect a specific dataset and
  methodology and may not generalize to production workloads. See the
  Apache-2.0 `LICENSE` for the warranty disclaimer.
