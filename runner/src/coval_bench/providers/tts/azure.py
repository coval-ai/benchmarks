# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Microsoft Azure AI Speech real-time TTS provider.

Drives the synthesis WebSocket protocol directly, not the Speech SDK: the SDK
adds seconds of event-delivery overhead that would inflate TTFA vs. other
providers (same caveat as the Azure STT provider).

Endpoint: wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v1.
Auth: Ocp-Apim-Subscription-Key header. The client sends three text frames —
speech.config and synthesis.context (JSON), then ssml — each a header block and
body separated by a blank line. The server streams audio as binary frames
(2-byte big-endian header length, header, PCM payload) and signals completion
with a turn.end text frame.
"""

from __future__ import annotations

import json
import struct
import time
import uuid
from datetime import UTC, datetime
from xml.sax.saxutils import escape, quoteattr

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_WS_PATH = "/cognitiveservices/websocket/v1"
_SAMPLE_RATE = 24000
# Raw, not riff-*: a RIFF header would corrupt the PCM that finalize_tts_result wraps.
_OUTPUT_FORMAT = "raw-24khz-16bit-mono-pcm"

_SPEECH_CONFIG = {
    "context": {
        "system": {"name": "coval-bench", "version": "1.0"},
        "os": {"platform": "Linux", "name": "Linux", "version": "1.0"},
    }
}

_SYNTHESIS_CONTEXT = {
    "synthesis": {
        "audio": {
            "metadataOptions": {
                "sentenceBoundaryEnabled": False,
                "wordBoundaryEnabled": False,
            },
            "outputFormat": _OUTPUT_FORMAT,
        }
    }
}


class AzureTTSProvider(TTSProvider):
    """Azure AI Speech real-time streaming TTS provider (raw WebSocket)."""

    _VALID_MODELS = frozenset({"neural", "dragon-hd-latest"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if not self._model_supported(model):
            raise ValueError(
                f"Invalid Azure TTS model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        if not voice:
            raise ValueError("AzureTTSProvider requires a voice name (e.g. en-US-AvaNeural)")
        if settings.azure_api_key is None:
            raise ValueError("azure_api_key is required in Settings")
        if settings.azure_region is None:
            raise ValueError("azure_region is required in Settings")
        self._api_key = settings.azure_api_key.get_secret_value()
        self._region = settings.azure_region
        self._model = model
        self._voice = voice

    @property
    def name(self) -> str:
        return f"azure-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _ssml(self, text: str) -> str:
        return (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
            f"<voice name={quoteattr(self._voice)}>{escape(text)}</voice></speak>"
        )

    async def synthesize(self, text: str) -> TTSResult:
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None
        request_id = uuid.uuid4().hex

        try:
            url = f"wss://{self._region}.tts.speech.microsoft.com{_WS_PATH}"
            headers = {
                "Ocp-Apim-Subscription-Key": self._api_key,
                "X-ConnectionId": uuid.uuid4().hex,
            }
            async with ws_client.connect(url, additional_headers=headers) as ws:
                await ws.send(
                    _text_message(
                        "speech.config", request_id, "application/json", json.dumps(_SPEECH_CONFIG)
                    )
                )
                await ws.send(
                    _text_message(
                        "synthesis.context",
                        request_id,
                        "application/json",
                        json.dumps(_SYNTHESIS_CONTEXT),
                    )
                )
                start = time.monotonic()
                await ws.send(
                    _text_message("ssml", request_id, "application/ssml+xml", self._ssml(text))
                )

                async for raw in ws:
                    if isinstance(raw, bytes):
                        path, payload = _parse_binary_message(raw)
                        if path == "audio" and payload:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(payload)
                        continue
                    if _parse_text_path(raw) == "turn.end":
                        break

        except Exception as exc:
            logger.warning("azure_tts_error", provider="azure", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="azure",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=_SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="azure",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=_SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _text_message(path: str, request_id: str, content_type: str, body: str) -> str:
    header = (
        f"Path:{path}\r\nX-RequestId:{request_id}\r\n"
        f"X-Timestamp:{_timestamp()}\r\nContent-Type:{content_type}\r\n"
    )
    return f"{header}\r\n{body}"


def _header_path(header: str) -> str:
    for line in header.split("\r\n"):
        name, _, value = line.partition(":")
        if name.strip().lower() == "path":
            return value.strip().lower()
    return ""


def _parse_text_path(raw: str) -> str:
    head, _, _ = raw.partition("\r\n\r\n")
    return _header_path(head)


def _parse_binary_message(raw: bytes) -> tuple[str, bytes]:
    """Split a server binary frame into (lowercased Path, audio payload)."""
    if len(raw) < 2:
        return "", b""
    (header_len,) = struct.unpack(">H", raw[:2])
    end = 2 + header_len
    if len(raw) < end:
        return "", b""
    header = raw[2:end].decode("utf-8", errors="replace")
    return _header_path(header), raw[end:]
