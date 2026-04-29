# Dataset Manifests

These JSON files are packaged inside the `coval-bench` wheel and loaded via
`importlib.resources` at runtime.  They are the **SHA-pinned source of truth**
for each dataset version — the runner validates every downloaded file against
the recorded hash before use.

## How each manifest is produced

### `stt-v1.json`

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
   - Sort by `(speaker_id, chapter_id, utterance_id)`.
   - Take the first 50 in the 2–15 second duration window.
3. Transcodes each file to 16 kHz mono PCM_16 WAV.
4. Computes SHA256 for each file.
5. Uploads to `gs://coval-benchmarks-datasets/stt-v1/audio/`.
6. Writes the updated `stt-v1.json` (commit the result).

**Never overwrite v1.** Future expansions go in `stt/v2/`.

License: `CC-BY-4.0` (LibriSpeech `test-clean` is explicitly redistributable).

### `tts-v1.json`

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
