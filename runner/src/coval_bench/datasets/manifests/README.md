# Dataset Manifests

These JSON files are packaged inside the `coval-bench` wheel and loaded via
`importlib.resources` at runtime.  They are the **SHA-pinned source of truth**
for each dataset version — the runner validates every downloaded file against
the recorded hash before use.

## How each manifest is produced

### The WildASR environment family (`stt-wildasr-*`)

Six index-aligned STT datasets built from the `environment_degradation` splits of
[bosonai/WildASR](https://huggingface.co/datasets/bosonai/WildASR): `clean` plus
five degradations of the same recordings (`clipping`, `far_field`, `noise_gap`,
`phone_codec`, `reverberation`). Because every manifest selects the same
utterances in the same order with the same filenames, `audio/0042.wav` is the
same utterance in every family dataset — clean-vs-degraded comparisons pair by
filename.

**Family selection** (shared by all six builds; implemented in
`datasets/scripts/wildasr.py`, deterministic from the frozen source):

- Utterance identity is the clean transcript. The source repeats most utterances
  as exact duplicate rows (same `audio_hash_id`), so rows are deduped by hash per
  (split, transcript); a split's *conditions* for an utterance are its distinct
  degraded audios. Deterministic degradations collapse to their true condition
  count (clipping 1, phone_codec 2, reverberation 3); randomized ones keep every
  distinct draw (far_field 6 or 3, noise_gap 8 or 4).
- Multi-condition splits contribute one row per utterance, rotated by the
  utterance's clean-order ordinal mod its condition count and advancing to the
  first in-band condition, so conditions are evenly represented without
  discarding utterances whose rotation-point condition runs long. Each item
  records `condition_idx`/`condition_count` and the source `audio_hash_id`.
- Filters are family-wide: an utterance stays only if EVERY split has a
  condition inside the 2.0–15.0 s band (noise_gap inserts silence that pushes
  long clips over) and its transcript has ≥ 3 words. An utterance whose
  transcript maps to more than one distinct clean recording would be ambiguous
  and is dropped (none in the current source).
- Result: **284 utterances** — every one the duration band permits — from the
  source's 350 (30 have out-of-band clean audio, 36 have a split where no
  condition fits the band), ~41 minutes of clean audio, transcripts unique and
  lexicographically ordered.
- Recording level: loudness-normalized during transcode (RMS target −20 dBFS,
  peak-guarded), same as `stt-v2`/`stt-v3` — the source audio is published very
  quiet.

**What we measure.** Same as the other STT sets: WER, TTFT, audio→final latency,
TTFS, RTF. The degraded sets exist for targeted robustness comparisons against
`stt-wildasr-clean` on the same utterances.

To rebuild any family dataset (reads the repo parquet, needs the `hf-parquet`
extra; `--keep-cache` keeps the ~2 GB source for sibling builds):

```bash
uv run --extra hf-parquet coval-build-dataset stt-wildasr-clean --keep-cache
```

Then fill `speech_end_offset_ms` (the TTFS anchor) with
`scripts/precompute_vad_offsets.py --write` against the built 16 kHz WAVs.

**Never overwrite a family manifest.** A source refresh changes the selection,
which would silently break cross-family filename alignment for recorded runs —
expansions get new ids.

License: `Apache-2.0` (WildASR); audio derived from FLEURS (`CC-BY-4.0`).

#### `stt-wildasr-clean.json`

The clean (undegraded) baseline of the family and the paired reference for the
five degradation datasets. Succeeds `stt-v2` (same source split, which stays
frozen): built via the family selection above instead of stt-v2's 50-clip
first-100-rows window.

### `stt-v3.json`

**What it is.** A frozen 897-clip set — the full usable pool of
[pipecat-ai/stt-benchmark-data](https://huggingface.co/datasets/pipecat-ai/stt-benchmark-data),
the corpus behind pipecat's [stt-benchmark](https://github.com/pipecat-ai/stt-benchmark)
project. The runner selects it via `DATASET_ID=stt-v3` and samples
`DATASET_SAMPLE_SIZE` clips per run as usual.

**Sample composition (stt-benchmark-data, the source pool):**

- **Language:** English; conversational voice-agent utterances (commands and
  questions a user says to an assistant), 1,000 rows of 16 kHz audio. The audio
  is drawn from `pipecat-ai/smart-turn-data-v3.1-train`, pipecat's
  community/partner-contributed turn-detection recordings. Human speech only:
  the source corpus also contains TTS-generated samples, but pipecat's set
  keeps only rows with `synthetic == False` (and `language == "eng"`).
- **Reference transcripts are model-generated.** pipecat produces ground truth
  with Gemini (see their README's `stt-benchmark ground-truth` step), not human
  transcription — so WER against this set measures agreement with a strong
  machine baseline, and absolute WER should be read with that in mind.
- **Our 897-clip pool:** the builder reads the full 1,000-row pool from the
  repo's parquet, keeps clips of **2.0–15.0 s** with ≥ 3 words, and dedups by
  transcript — every survivor is included (897), deterministic from the
  frozen source. (The build requires the parquet source, which reads the
  whole pool; the datasets-server REST path caps its scan at a split's first
  100 rows and fails loudly at selection.)
- **Recording level:** each clip is loudness-normalized (RMS target −20 dBFS,
  peak-guarded), same as `stt-v2`. Pass `--normalize`.
- **Durations:** 2.1–15.0 s, median ≈ 10.5 s; ~2.3 hours total. Per-run cost
  is unchanged — runs draw `DATASET_SAMPLE_SIZE` (default 10) clips.
- Items carry pipecat's `sample_id` for provenance back to the source rows.

**What we measure on it.** Same as `stt-v2`: WER, TTFT, audio→final latency,
TTFS, RTF.

To rebuild (downloads the repo parquet, transcodes and uploads the selected
clips, writes the manifest):

```bash
uv run --extra hf-parquet coval-build-dataset --hf pipecat-ai/stt-benchmark-data \
    --config default --split train \
    --dataset-id stt-v3 --num 897 --dur-max 15 --normalize \
    --license "unspecified (no data license published by pipecat-ai)" \
    --source "pipecat-ai/stt-benchmark-data train"
```

Then fill `speech_end_offset_ms` (the TTFS anchor) with
`scripts/precompute_vad_offsets.py --write` against the built 16 kHz WAVs.

**Never overwrite v3.** Future expansions go in `stt-v4`.

License: **unspecified** — the HF dataset card carries no license tag and the
`stt-benchmark` repo has no license file (pipecat's `smart-turn` BSD-2-Clause
covers the model code, not the audio). The audio is mirrored to the public GCS
bucket on the same basis as pipecat's own public redistribution; if pipecat
publishes a data license, record it here and in the manifest.

### `stt-v2.json`

**Retired from the scheduled mix** (replaced by the `stt-v1` + `stt-v3` pair);
retained for reproducibility of runs recorded against it. Never overwrite.

**What we benchmark.** STT providers were scored against a frozen 50-utterance
sample of the `environment_degradation__en__fleurs_clean_en` split of
[bosonai/WildASR](https://huggingface.co/datasets/bosonai/WildASR) — the clean
(undegraded) English baseline of WildASR's environment-degradation suite,
derived from [FLEURS](https://huggingface.co/datasets/google/fleurs). The
WildASR repo is `Apache-2.0`; the underlying FLEURS audio is `CC-BY-4.0` —
both redistributable, so the audio is mirrored to the public GCS bucket and
the OSS reproducibility contract holds.

**Sample composition (fleurs_clean_en, the source pool):**

- **Language:** English; read speech, clean recording conditions.
- **Recording level:** the source audio is published very quiet (~73% of clips
  peak below −26 dBFS, median peak ≈ −41 dBFS), quiet enough that some
  providers' VAD/endpointing fails to detect speech at all. The build
  loudness-normalizes each clip (RMS target −20 dBFS, peak-guarded) so the
  benchmark measures transcription quality on clean speech rather than
  low-level-input robustness. Pass `--normalize` to `coval-build-dataset`.
- **Speaker metadata:** WildASR publishes no per-clip speaker identity for
  this subset (`speaker_id=spk0`, `gender=unknown` on every row), so the
  sample is not speaker-balanced.
- **Our 50-utterance window:** the builder scans the split's first 100 rows
  (its per-split scan cap), keeps clips of **2.0–15.0 s** with ≥ 3 words,
  dedups by transcript (the source repeats some recordings under different
  row indices), and takes the lexicographically-first 50 by transcript —
  fully deterministic from the frozen split.
- **Total audio per benchmark run:** 50 utterances × ~9 s avg ≈ 7.5 minutes.

**What we measure on it.** WER (primary correctness metric), TTFT, audio→final
latency, TTFS, RTF. Transcripts are lowercase with punctuation; the WER
normalization pipeline (lowercase, NFKC, punctuation stripping) applies
identically to reference and hypothesis.

To rebuild (reads the split live via the datasets-server API, downloads and
transcodes the selected clips, uploads, writes the manifest):

```bash
uv run coval-build-dataset --hf bosonai/WildASR \
    --split environment_degradation__en__fleurs_clean_en \
    --dataset-id stt-v2 --dur-max 15 --normalize \
    --license "Apache-2.0 (WildASR; audio derived from FLEURS, CC-BY-4.0)" \
    --source "bosonai/WildASR environment_degradation__en__fleurs_clean_en"
```

Then fill `speech_end_offset_ms` (the TTFS anchor) with
`scripts/precompute_vad_offsets.py --write` against the built 16 kHz WAVs.

**Never overwrite v2.** Future expansions go in `stt-v4` (`stt-v3` is taken by
the pipecat set above).

License: `Apache-2.0` (WildASR); audio derived from FLEURS (`CC-BY-4.0`).

### `stt-v1.json`

**In the scheduled mix** as the easy set alongside `stt-v3` (it was retired in
favor of `stt-v2` from 2026-07-08 to 2026-07-13). Never overwrite.

**What we benchmark.** STT providers are scored against a frozen 50-utterance
sample of [LibriSpeech `test-clean`](https://www.openslr.org/12/) (OpenSLR-12).
LibriSpeech is read English speech from public-domain LibriVox audiobooks,
released under `CC-BY-4.0` — explicitly redistributable, which is why we can
mirror the audio to a public GCS bucket and keep the OSS reproducibility
contract.

**Sample composition (test-clean, the source pool):**

- **Language:** English (US-dominant, with some other English readers).
- **Style:** read speech, studio-quality, near-zero background noise. No
  conversational/spontaneous speech, no overlap, no telephony codecs.
- **Speakers in pool:** 40 total, **20 F / 20 M** (balanced by design).
- **Our 50-utterance window:** filtered to the **2.0–15.0 s** duration range,
  then **round-robin by speaker** in lex order (one utterance per speaker per
  round, cycling until 50 picked). All 40 speakers contribute, with the lex-first
  10 providing 2 utterances each — fully deterministic and gender-balanced. See
  ADR-020 for the full selection rule.
- **Total audio per benchmark run:** ~50 utterances × ~7 s avg ≈ 5–6 minutes.

**What we measure on it.** WER (primary correctness metric), TTFT, audio→final
latency, RTF. Per ADR-020, **absolute WER is artificially low** because most
providers train on LibriSpeech — so the value lives in (a) latency metrics,
which are unaffected, and (b) per-provider regression-over-time, which works
symmetrically.

To rebuild the manifest from scratch (downloads LibriSpeech, transcodes,
uploads, recomputes hashes), run `scripts/build_dataset.py`:

```bash
uv run python scripts/build_dataset.py build \
    --dataset stt-v1 \
    --bucket coval-benchmarks-datasets
```

This script:
1. Downloads the LibriSpeech `test-clean` tar from OpenSLR.
2. Deterministically selects 50 utterances per ADR-020:
   - Filter to utterances with `2.0 <= duration_sec <= 15.0`.
   - Sort by `(speaker_id, chapter_id, utterance_id)` lex.
   - Round-robin by speaker until 50 are picked; re-sort the picks by lex
     so filenames `0001.wav..0050.wav` are in stable order.
3. Transcodes each file to 16 kHz mono PCM_16 WAV.
4. Computes SHA256 for each file.
5. Uploads to `gs://coval-benchmarks-datasets/stt-v1/audio/`.
6. Writes the updated `stt-v1.json` (commit the result).

**Never overwrite v1.**

License: `CC-BY-4.0` (LibriSpeech `test-clean` is explicitly redistributable).

### `tts-v1.json`

**What we benchmark.** TTS providers are scored on a fixed set of **30 short
English transcripts** drawn from synthetic customer-service domains: order
tracking, appointment scheduling, account verification, technical support.
Transcripts are text-only (no reference audio) — so we measure latency, not
intelligibility. License: **Apache-2.0** (same as the rest of the repo); the
manifest is shipped inside the wheel.

**What we measure on it.** TTFA (time to first audio byte, primary), RTF,
end-to-end synthesis latency. There is no reference-audio quality metric —
voices vary by provider, so apples-to-apples audio quality comparison is out
of scope for v1.

Generated from a curated set of 30 short customer-service prompts
(Apache-2.0).

To regenerate from a local CSV (`--in tts-source.csv`, columns
`Testcase ID`, `Transcript`):

```bash
python3 -c "
import csv, json, sys
rows = []
with open(sys.argv[1], encoding='cp1252') as f:
    for row in csv.DictReader(f):
        rows.append({'testcase_id': row['Testcase ID'], 'transcript': row['Transcript']})
manifest = {
    '_license': 'Apache-2.0',
    'id': 'tts-v1',
    'version': '1.0.0',
    'license': 'Apache-2.0',
    'source': 'Coval TTS test cases',
    'items': rows,
}
print(json.dumps(manifest, indent=2, ensure_ascii=False))
" tts-source.csv > runner/src/coval_bench/datasets/manifests/tts-v1.json
```

License: `Apache-2.0`.

### `s2s-v1.json`

**What we benchmark.** Speech-to-speech (S2S) providers are scored on a frozen
50-clip sample of single-turn spoken user *questions* from
[SLURP](https://github.com/pswietojanski/slurp) (Bastianelli et al., EMNLP 2020).
The headline metric is voice-to-voice (V2V) response latency — time from
end-of-user-speech to the model's first returned audio frame.

**Sample composition:**

- **Language:** English; **headset/close-talk** recordings only, `status=correct`.
- **Intents:** **query/question-eliciting only** (`qa_*`, `*_query`, …) so every
  clip demands a spoken reply — V2V latency is undefined for device commands that
  produce no audio.
- **Balance:** **25 F / 25 M**, **25 native / 25 non-native** English speakers,
  spread across **15 SLURP scenarios**.
- **Duration:** 2.0–10.0 s; minimum 3 words.
- **`speech_end_offset_ms` (t0):** end-of-speech anchor from SileroVAD via
  `scripts/precompute_vad_offsets.py` (offline; silero/torch are not runner deps).

**What we measure on it.** V2V latency (first-audio-out − end-of-speech), P50/P95
over rolling windows. No ground-truth answers are needed — latency only.

**License — note the difference from `stt-v1`.** SLURP **text** (transcripts) is
`CC-BY-4.0`; SLURP **audio** ([Zenodo 4274930](https://zenodo.org/record/4274930))
is **`CC-BY-NC-4.0` — NonCommercial, attribution required.** Unlike LibriSpeech's
CC-BY, the audio carries a NonCommercial restriction (a commercial license is
available from Emotech, `info@emotech.co`). Attribution: Bastianelli, Vanzo,
Swietojanski, Rieser, *"SLURP: A Spoken Language Understanding Resource Package,"*
EMNLP 2020.

**How it's built** (selection is deterministic from the immutable SLURP source):

1. Pull SLURP metadata + the Zenodo `slurp_real.tar.gz`; exclude `train_synthetic`.
2. Keep headset recordings with `status=correct` and a non-empty transcript;
   derive `scenario` from the intent; join speaker gender / native-language from
   `metadata.json` (`usrid`).
3. Filter to query/question-eliciting intents.
4. Select 50: balanced 25 F / 25 M and native/non-native, spread across scenarios,
   one recording per prompt; backfill within the same (scenario, gender) stratum
   for clips outside 2–10 s or under 3 words.
5. Transcode to 16 kHz mono PCM_16 WAV; SHA256 each; upload to
   `gs://coval-benchmarks-datasets/s2s-v1/audio/`.
6. Fill `speech_end_offset_ms` with `scripts/precompute_vad_offsets.py` (SileroVAD).

**Never overwrite v1.** Future expansions go in `s2s/v2/`.

License: `CC-BY-NC-4.0` (SLURP audio; NonCommercial, attribution required).

## Schema

```json
{
  "_license": "Apache-2.0",
  "id": "stt-v1",
  "version": "1.0.0",
  "license": "CC-BY-4.0",
  "source": "LibriSpeech test-clean (OpenSLR-12)",
  "items": [
    {
      "path": "audio/0001.wav",
      "sha256": "<64-char hex>",
      "transcript": "...",
      "duration_sec": 4.2,
      "speaker_id": "1234",
      "chapter_id": "5678",
      "utterance_id": "0001"
    }
  ]
}
```

The `_license` field carries the Apache-2.0 SPDX identifier for the manifest
file itself (JSON does not support comments).
