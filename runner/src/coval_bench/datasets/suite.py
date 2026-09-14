from __future__ import annotations

from typing import Final

DEFAULT_STT_DATASET: Final = "stt-v3"
DEFAULT_SAMPLE_SIZE: Final = 10

DEDICATED_STT_SUITE: Final[dict[str, int]] = {
    "stt-v3": 240,
    "stt-wildasr-clean": 96,
    "stt-wildasr-clipping": 24,
    "stt-wildasr-farfield": 24,
    "stt-wildasr-noisegap": 24,
    "stt-wildasr-phonecodec": 24,
    "stt-wildasr-reverb": 24,
    "stt-wildasr-accent": 24,
}

DEDICATED_TTS_SAMPLE_SIZE: Final = 60


def stt_sample_size(source: str, dataset_id: str, override: int | None) -> int:
    if override is not None:
        return override
    if source == "dedicated":
        return DEDICATED_STT_SUITE.get(dataset_id, DEFAULT_SAMPLE_SIZE)
    return DEFAULT_SAMPLE_SIZE


def tts_sample_size(source: str, override: int | None) -> int:
    if override is not None:
        return override
    return DEDICATED_TTS_SAMPLE_SIZE if source == "dedicated" else DEFAULT_SAMPLE_SIZE
