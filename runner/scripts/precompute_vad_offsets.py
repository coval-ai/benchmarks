"""Offline: compute per-clip speech-end offsets (ms) for the STT dataset.

Standalone dev tool — NOT part of the runner, NOT under src/ (so it is not
subject to the mypy --strict gate). Requires silero-vad + soundfile installed
separately; they are deliberately not runner dependencies:

    uv pip install silero-vad soundfile

Usage:
    python scripts/precompute_vad_offsets.py \
        --manifest src/coval_bench/datasets/manifests/stt-v1.json \
        --audio-dir /path/to/local/16k/wavs \
        [--write] [--min-silence-ms 300] [--speech-pad-ms 100]

For each clip the manifest references, runs SileroVAD and records
speech_end_offset_ms = end of the LAST detected speech segment (already
includes speech-pad as the guard). With --write, patches the field into the
manifest in place; otherwise prints a review table. Audio bytes are untouched,
so every clip's sha256 stays the same.

Validation after running: confirm truncated-clip WER == full-clip WER on all
clips before trusting the offsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLE_RATE = 16000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--min-silence-ms", type=int, default=300)
    parser.add_argument("--speech-pad-ms", type=int, default=100)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    manifest = json.loads(args.manifest.read_text())
    items = manifest["items"] if isinstance(manifest, dict) else manifest

    rows: list[tuple[str, float | None, float]] = []
    for item in items:
        rel = item["path"]
        wav_path = args.audio_dir / Path(rel).name
        if not wav_path.exists():
            rows.append((rel, None, item.get("duration_sec", 0.0)))
            continue

        wav_np, file_sr = sf.read(str(wav_path), dtype="float32")
        if wav_np.ndim > 1:
            wav_np = np.mean(wav_np, axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(wav_np))
        ts = get_speech_timestamps(
            wav,
            model,
            sampling_rate=file_sr,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        duration_ms = float(len(wav)) / file_sr * 1000.0
        if ts:
            raw_end_ms = float(ts[-1]["end"]) / file_sr * 1000.0
            speech_end_ms = round(min(raw_end_ms, duration_ms), 1)  # round after clamp
        else:
            speech_end_ms = None  # no speech detected → don't truncate this clip
        item["speech_end_offset_ms"] = speech_end_ms
        rows.append((rel, speech_end_ms, duration_ms))

    for rel, end_ms, dur_ms in rows:
        trailing = "n/a" if end_ms is None else f"{dur_ms - end_ms:7.1f}"
        end_str = "MISSING/NO-SPEECH" if end_ms is None else f"{end_ms:8.1f}"
        print(f"{rel:48s}  end_ms={end_str}  clip_ms={dur_ms:8.1f}  trailing_ms={trailing}")

    if args.write:
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWrote speech_end_offset_ms into {args.manifest}")
    else:
        print("\n(dry run — pass --write to patch the manifest)")


if __name__ == "__main__":
    main()
