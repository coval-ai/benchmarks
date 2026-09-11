"""Datasets a dedicated execution covers, and how many items each draws.

The dedicated job runs once a day with a single trigger; ``run --kind stt
--source dedicated`` walks this suite sequentially. The shared job still gets
one execution per dataset from its infra triggers, which pass ``DATASET_ID``
and ``DATASET_SAMPLE_SIZE``; both remain overrides for any run.
"""

from __future__ import annotations

from typing import Final

DEFAULT_STT_DATASET: Final = "stt-v3"
DEFAULT_SAMPLE_SIZE: Final = 10

DEDICATED_STT_SUITE: Final[dict[str, int]] = {
    "stt-v3": 30,
    "stt-wildasr-clean": 12,
    "stt-wildasr-clipping": 3,
    "stt-wildasr-farfield": 3,
    "stt-wildasr-noisegap": 3,
    "stt-wildasr-phonecodec": 3,
    "stt-wildasr-reverb": 3,
    "stt-wildasr-accent": 3,
}

# Exceeds the tts-v1 manifest, so the daily dedicated run covers every prompt.
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
