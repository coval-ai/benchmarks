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
4. `coval-bench run --kind stt --providers <provider> --models <model>`
   (or analogous for TTS). The runner downloads the same dataset audio from
   GCS, verifies the SHA, runs against the same model identifier, and applies
   the same normalization pipeline.

Numbers should match the leaderboard within provider-side variance (latency
metrics in particular are session-dependent).

## ADR references

- ADR-011 — Provider scope and the `disabled` flag policy.
- ADR-020 — Dataset selection rule (round-robin by speaker, 2.0–15.0 s window).

The full ADR set is published under `docs/adr/` as decisions are promoted
from internal status to public commitment.

## Caveats

- Absolute WER on LibriSpeech `test-clean` is artificially low for most
  providers because the corpus is widely included in training data.
  Latency and per-provider drift over time are the more informative
  signals.
- This is a research benchmark. Results reflect a specific dataset and
  methodology and may not generalize to production workloads. See the
  Apache-2.0 `LICENSE` for the warranty disclaimer.
