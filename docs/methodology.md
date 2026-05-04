# Methodology

A leaderboard number on [benchmarks.coval.ai](https://benchmarks.coval.ai) is a
function of four pinned inputs. Reproducing a number requires reproducing all
four. This document describes each one and where it lives.

## 1. Dataset

### STT v1 (`stt-v1`)

A 50-utterance subset of
[LibriSpeech `test-clean`](https://www.openslr.org/12/) (OpenSLR-12,
`CC-BY-4.0`). Selection rule: utterances filtered to the 2.0–15.0 s duration
range, then round-robin by speaker in lexicographic order — all 40 speakers
(20 F / 20 M) contribute, with the lex-first 10 providing 2 utterances each.
The selection is deterministic and reproducible from the raw `test-clean`
download.

### STT v2 (`stt-v2`)

A frozen STT corpus sourced from Hugging Face
[`PolyAI/minds14`](https://huggingface.co/datasets/PolyAI/minds14) (`en-US`,
`train`), **`CC-BY-4.0`**. The packaged manifest `stt-v2.json` is **authored and
frozen**: each row keys off the Hub `path`, carries a human-reviewed reference
transcript (and editorial metadata such as intent labels where present), and
pins **`sha256`** for the mirrored WAV bytes. The JSON on disk remains the
source of truth for transcripts and paths (reviewed text, not raw Hub
captions).

**Why this corpus (vs `stt-v1`).** `stt-v1` is clean read audiobook speech — ideal
for latency baselines and regression tracking, but not representative of
**narrowband / telephony-shaped** audio, **accent and speaker variability**, or
**messy real-world wording**. `stt-v2` fills that gap: **60** frozen clips from
MInDS-14-style banking customer service (**30 male / 30 female** speakers in
the packaged manifest), **one distinct speaker per clip**, grouped by
**intent** (12 categories in the packaged manifest, e.g. pay bill, freeze
card, app error). Clips are **resampled to
16 kHz mono** for the runner, but upstream captures retain **telephone-style
bandwidth and codec/DSP character** — lower acoustic fidelity than LibriSpeech
— on purpose, so WER and timings reflect **call-center STT** more than studio
read speech. English is **US-market customer service** with natural
speaker-to-speaker variation (including **regional accents** common in that
setting).

**Runtime object layout.** The runner loads audio from
`gs://<dataset_bucket>/stt-v2/<item.path>` (see
`runner/src/coval_bench/datasets/loader.py`). There is no separate `/audio/`
prefix unless paths inside the manifest include it.

**Rebuilding / verifying.** From `runner/`, with the optional HF extra
installed, run (see `manifests/README.md` for the full flags):

```bash
uv run python -m coval_bench.datasets.scripts.build_dataset build \
  --dataset stt-v2 --bucket <bucket> [--dry-run]
```

Dry-run checks SHA256
against the packaged manifest without GCS writes; omit `--dry-run` to upload.
Optional **`HF_MINDS14_REVISION`** pins the Hub snapshot for byte-stable
rebuilds.

**Metrics on v2.** The same STT metrics apply as for `stt-v1` (WER after the
normalization pipeline in `runner/src/coval_bench/metrics/wer.py`, TTFT,
audio→final latency, RTF). Utterance durations are wider than LibriSpeech v1
(see `duration_sec` in the manifest); interpret absolute WER and latency in
that context.

**ADR scope.** ADR-020 governs **LibriSpeech `stt-v1` construction** (round-robin
by speaker in the 2.0–15.0 s window). **`stt-v2` item order and inclusion are
defined only by the committed manifest**, not by that ADR.

**TTS benchmark.** A 30-prompt set of short text inputs. Source and selection
rule are documented in `runner/src/coval_bench/datasets/manifests/tts-v1.json`.

**SHA pinning.** Every audio item in an STT manifest (`stt-v1.json`,
`stt-v2.json`, …) carries a `sha256` field. The runner verifies the SHA after
fetching from GCS and raises `DatasetIntegrityError` on mismatch (see
`runner/src/coval_bench/datasets/loader.py`). TTS items are text-only and have
no SHA.

**Versioning.** Dataset manifests live at
`runner/src/coval_bench/datasets/manifests/{stt-v1,stt-v2,tts-v1}.json` and carry a
`version` field. Bumping the dataset re-pins by minting a new manifest
(`stt-v2.json` etc.); the old manifest stays for historical reproducibility.

**Rebuilding from scratch.** From `runner/`, run the packaged builder
`python -m coval_bench.datasets.scripts.build_dataset` (see
`runner/src/coval_bench/datasets/scripts/build_dataset.py`): for **`stt-v1`**
it downloads LibriSpeech, applies the ADR-020 selection rule, transcodes,
uploads to GCS, and writes the manifest with fresh SHAs. For **`stt-v2`** use
`publish_reviewed_stt_v2` (see **Rebuilding / publishing** under **`stt-v2`**
above). See `runner/src/coval_bench/datasets/manifests/README.md` for an
operator checklist when present.

**Selection rationale.** See [ADR-020](#adr-references) below.

## 2. Provider model versions

Each `(provider, model, voice)` tuple the runner exercises is declared in
`runner/src/coval_bench/runner/config.py` (`DEFAULT_STT_MATRIX` and
`DEFAULT_TTS_MATRIX`). Model identifiers are exact provider strings — not
aliases — wherever the provider exposes a versioned identifier (e.g.
`nova-3`, `flux-general-en`, `tts-1-hd`, `aura-2-thalia-en`, `mistv3`).

**Server-side aliases.** Where a provider resolves a name server-side (e.g.
Rime `arcana` → Arcana v3), the resolution is documented inline in
`config.py` next to the matrix entry.

**Disabled entries.** Some entries are intentionally inert (`disabled=True`).
The runner skips them; the API exposes them so that a frontend can grey out
models that are catalogued but not currently producing data. See
[ADR-011](#adr-references) for the policy.

## 3. Normalization pipeline

WER is computed after a deterministic normalization pipeline applied to both
the reference transcript and the provider hypothesis. The pipeline lives in
`runner/src/coval_bench/metrics/wer.py` and covers currency expansion,
ordinals, dates, time expressions, dehyphenation, and unicode folding.

**Version constant.** `wer.NORM_VERSION` (currently `"1"`) is bumped on any
behavioural change to the pipeline. The constant is included on every
`WERResult` as `norm_version`, so a result row can be unambiguously
attributed to the pipeline version that produced it.

A future revision may A/B-test against `whisper_normalizer`. Any such change
will increment `NORM_VERSION` and ship a separate ADR.

## 4. Library versions

The DP edit-distance computation is delegated to `jiwer`. Pinned in
`runner/uv.lock` (currently `jiwer == 4.0.0`). `numpy` is also locked. Every
`uv.lock` change is reviewed against the SPDX license-set policy before
merge.

## Reproducing a result

To reproduce a single `(provider, model, voice, metric)` cell:

1. Clone the repo at the commit SHA from the leaderboard row
   (`/v1/runs/<id>` exposes `runner_sha`).
2. `uv sync` inside `runner/`.
3. Set the relevant provider API key in `.env`.
4. From `runner/`, run `coval-bench run --kind stt` (add `--smoke` for one
   utterance × the full enabled matrix). The CLI does **not** accept
   `--providers` / `--models`; rows come from `DEFAULT_STT_MATRIX` /
   `DEFAULT_TTS_MATRIX` in `runner/src/coval_bench/runner/config.py`. Point
   `DATASET_ID` / `DATASET_BUCKET` at the same corpus as the run you are
   reproducing (e.g. `stt-v1` or `stt-v2`). For a single `(provider, model)`
   probe **without Postgres**, use `coval-bench stt-preview`. The runner
   fetches STT audio from GCS when applicable, verifies SHA256, and applies the
   same WER normalization pipeline.

Numbers should match the leaderboard within provider-side variance (latency
metrics in particular are session-dependent).

## ADR references

- ADR-011 — Provider scope and the `disabled` flag policy.
- ADR-020 — Dataset selection rule (round-robin by speaker, 2.0–15.0 s window).

ADR rationale is referenced inline in the relevant source files and READMEs
where the decision context is load-bearing.

## Caveats

- Absolute WER on LibriSpeech `test-clean` is artificially low for most
  providers because the corpus is widely included in training data.
  Latency and per-provider drift over time are the more informative
  signals.
- **`stt-v2`** (MInDS-14–derived) is conversational, **phone-style**,
  customer-intent English with different speakers per clip; WER and latency are
  not directly comparable to `stt-v1` leaderboard cells — compare providers
  **within** the same dataset id and manifest version.
- This is a research benchmark. Results reflect a specific dataset and
  methodology and may not generalize to production workloads. See the
  Apache-2.0 `LICENSE` for the warranty disclaimer.
