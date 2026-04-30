# Dataset Manifests

These JSON files are packaged inside the `coval-bench` wheel and loaded via
`importlib.resources` at runtime.  They are the **SHA-pinned source of truth**
for each dataset version — the runner validates every downloaded file against
the recorded hash before use.

## How each manifest is produced

### `stt-v1.json`

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

**Phase 2 (current):** Stub with 2 placeholder items.  The real hashes are
not yet populated because the audio files have not been uploaded to GCS.

**Phase 4:** Run `scripts/build_dataset.py` to regenerate:

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

**Never overwrite v1.** Future expansions go in `stt/v2/`.

License: `CC-BY-4.0` (LibriSpeech `test-clean` is explicitly redistributable).

### `tts-v1.json`

**What we benchmark.** TTS providers are scored on a fixed set of **30 short
English transcripts** drawn from real Coval customer-service domains: order
tracking, appointment scheduling, account verification, technical support.
Transcripts are text-only (no reference audio) — so we measure latency, not
intelligibility. License: **Coval-internal — not redistributable**, so this
manifest is shipped inside the wheel but the corpus itself never goes to
public GCS.

**What we measure on it.** TTFA (time to first audio byte, primary), RTF,
end-to-end synthesis latency. There is no reference-audio quality metric —
voices vary by provider, so apples-to-apples audio quality comparison is out
of scope for v1.

Converted from `legacy-benchmarks/tts/Test cases.csv` (cp1252 encoding) by
the `dataset-loader` subagent during Phase 2.  Contains 30 test cases.

To regenerate from the CSV (read-only source):

```bash
python3 -c "
import csv, json
rows = []
with open('legacy-benchmarks/tts/Test cases.csv', encoding='cp1252') as f:
    for row in csv.DictReader(f):
        rows.append({'testcase_id': row['Testcase ID'], 'transcript': row['Transcript']})
manifest = {
    '_license': 'Apache-2.0',
    'id': 'tts-v1',
    'version': '1.0.0',
    'license': 'Coval-internal \u2014 do not redistribute',
    'source': 'Coval TTS test cases (legacy CSV, cp1252)',
    'items': rows,
}
print(json.dumps(manifest, indent=2, ensure_ascii=False))
" > benchmarks/runner/src/coval_bench/datasets/manifests/tts-v1.json
```

License: `Coval-internal — do not redistribute`.

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
