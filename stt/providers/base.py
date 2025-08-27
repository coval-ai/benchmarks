import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass
class TranscriptionResult:
    provider: str

    ttft_seconds: Optional[float] = None  # Time to First Token
    total_time: Optional[float] = None  # Total processing time
    audio_to_final_seconds: Optional[float] = None  # Audio start to final transcript
    rtf_value: Optional[float] = (
        None  # Real-Time Factor (audio_duration / audio_to_final)
    )

    first_token_content: Optional[str] = None  # What triggered TTFT
    complete_transcript: Optional[str] = None  # Final transcript
    partial_transcripts: List[str] = field(default_factory=list)  # All partials

    transcript_length: Optional[int] = None  # Character count
    word_count: Optional[int] = None  # Word count
    wer_percentage: Optional[float] = None  # Word Error Rate as percentage

    error: Optional[str] = None

    audio_start_time: Optional[float] = None

    vad_first_detected: Optional[float] = None  # First VAD event timing
    vad_events_count: Optional[int] = None  # Number of VAD events
    vad_first_event_content: Optional[str] = None  # Content of first VAD event


class STTProvider(ABC):
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
        base_name = self.__class__.__name__.replace("Provider", "").lower()
        if model == "default":
            self.name = base_name
        else:
            self.name = f"{base_name}-{model}"

    @abstractmethod
    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float = None,
    ) -> TranscriptionResult:
        pass

    async def send_audio_chunks(
        self,
        ws,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        close_message: dict,
        result: TranscriptionResult,
        realtime_resolution: float = 0.1,
    ):
        byte_rate = sample_width * sample_rate * channels
        data_copy = audio_data
        first_chunk = True

        try:
            while len(data_copy):
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data_copy = data_copy[:chunk_size], data_copy[chunk_size:]

                if first_chunk:
                    result.audio_start_time = time.time()
                    first_chunk = False

                await ws.send(chunk)

                await asyncio.sleep(realtime_resolution)

            await ws.send(json.dumps(close_message))

        except Exception as e:
            print(f"Send error for {self.name}: {e}")
            raise
