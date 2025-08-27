import asyncio
import json
import time
import websockets
import logging
from .base import STTProvider, TranscriptionResult


class SpeechmaticsProvider(STTProvider):
    def __init__(self, api_key: str, model: str = "default"):
        super().__init__(api_key, model)
        # Speechmatics models - removed telephony as it's not supported
        valid_models = ["default", "enhanced", "broadcast"]
        if model not in valid_models:
            raise ValueError(
                f"Invalid Speechmatics model: {model}. Valid models: {valid_models}"
            )

    def _build_websocket_url(self) -> str:
        """Build WebSocket URL for Speechmatics Real-Time SaaS."""
        return "wss://eu2.rt.speechmatics.com/v2"

    def _build_start_recognition_config(self, sample_rate: int) -> dict:
        """Build StartRecognition message configuration."""
        transcription_config = {
            "language": "en",
            "enable_partials": True,
        }

        # Add operating point if specified
        if self.model == "enhanced":
            transcription_config["operating_point"] = "enhanced"
        elif self.model == "broadcast":
            transcription_config["domain"] = "broadcast"
        # For "default", use standard operating point (no additional config)

        return {
            "message": "StartRecognition",
            "transcription_config": transcription_config,
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
        }

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float = None,
    ) -> TranscriptionResult:
        """Speechmatics-specific implementation with TTFT using first transcript content method."""
        from wer_calculator import compare_transcription

        result = TranscriptionResult(provider=self.name)
        total_start_time = time.time()

        try:
            # Speechmatics WebSocket URL and headers
            websocket_url = self._build_websocket_url()
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with websockets.connect(websocket_url, extra_headers=headers) as ws:
                # Send StartRecognition configuration
                start_config = self._build_start_recognition_config(sample_rate)
                await ws.send(json.dumps(start_config))

                # Wait for RecognitionStarted confirmation
                await self._wait_for_recognition_started(ws, result)

                # Set audio start time NOW, before creating tasks
                result.audio_start_time = time.time()

                # Create concurrent tasks for sending and receiving
                send_task = asyncio.create_task(
                    self._send_audio_chunks_speechmatics_timed(
                        ws,
                        audio_data,
                        channels,
                        sample_width,
                        sample_rate,
                        result,
                        realtime_resolution,
                    )
                )
                receive_task = asyncio.create_task(self._receive_responses(ws, result))

                # Wait for both tasks to complete
                await asyncio.gather(send_task, receive_task, return_exceptions=True)

        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")

        return result

    async def _wait_for_recognition_started(self, ws, result: TranscriptionResult):
        """Wait for RecognitionStarted message before sending audio."""
        async for message in ws:
            if isinstance(message, bytes):
                continue

            response = json.loads(message)

            if response.get("message") == "RecognitionStarted":
                break
            elif response.get("message") == "Error":
                raise Exception(
                    f"Speechmatics error: {response.get('reason', 'Unknown error')}"
                )

    async def _send_audio_chunks_speechmatics_timed(
        self,
        ws,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        result: TranscriptionResult,
        realtime_resolution: float = 0.1,
    ):
        """Send audio chunks to Speechmatics with pre-set timing baseline."""
        byte_rate = sample_width * sample_rate * channels
        data_copy = audio_data
        seq_no = 0  # Track sequence number for EndOfStream

        try:
            while len(data_copy):
                # Calculate chunk size for realtime_resolution seconds of audio
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data_copy = data_copy[:chunk_size], data_copy[chunk_size:]

                # Send binary audio chunk (AddAudio message)
                await ws.send(chunk)
                seq_no += 1  # Increment sequence number for each audio chunk

                # Wait to simulate real-time
                await asyncio.sleep(realtime_resolution)

            # Send EndOfStream message with last sequence number (as per Speechmatics SDK)
            end_message = {"message": "EndOfStream", "last_seq_no": seq_no}
            await ws.send(json.dumps(end_message))

        except Exception as e:
            print(f"Send error for {self.name}: {e}")
            raise

    async def _receive_responses(self, ws, result: TranscriptionResult):
        """Receive and process Speechmatics responses with TTFT using first transcript content method."""
        final_transcripts = []
        last_final_transcript_time = None

        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue

                response = json.loads(message)
                current_time = time.time()
                message_type = response.get("message")

                # Handle different message types
                if message_type == "AudioAdded":
                    # Confirmation that audio was received - can be used for flow control
                    continue
                elif message_type == "EndOfTranscript":
                    # Final message indicating all processing is complete
                    break
                elif message_type in ["Error", "Warning", "Info"]:
                    if message_type == "Error":
                        # Don't set error for schema validation issues on EndOfStream
                        error_reason = response.get("reason", "Unknown error")
                        if (
                            "EndOfStream" not in error_reason
                            and "schema" not in error_reason.lower()
                        ):
                            result.error = error_reason
                    continue

                # Extract transcript for TTFT measurement
                transcript = self._extract_transcript(response)
                if transcript:
                    # TTFT: Record first actual transcript content timing - Speechmatics uses first content method
                    if (
                        result.ttft_seconds is None
                        and result.audio_start_time is not None
                    ):
                        result.ttft_seconds = current_time - result.audio_start_time
                        result.first_token_content = transcript

                    # Track partial transcripts
                    result.partial_transcripts.append(transcript)

                # For complete transcript, only collect AddTranscript messages (final results)
                if message_type == "AddTranscript":
                    final_transcripts.append(transcript)
                    last_final_transcript_time = (
                        current_time  # Track when last final transcript was received
                    )

        except Exception as e:
            print(f"Receive error for {self.name}: {e}")

        # Calculate audio-to-final timing
        if last_final_transcript_time and result.audio_start_time:
            result.audio_to_final_seconds = (
                last_final_transcript_time - result.audio_start_time
            )

        # Build complete transcript from final results only
        if final_transcripts:
            result.complete_transcript = " ".join(final_transcripts).strip()
            result.transcript_length = len(result.complete_transcript)
            result.word_count = (
                len(result.complete_transcript.split())
                if result.complete_transcript
                else 0
            )

    def _extract_transcript(self, response_json: dict) -> str:
        """Extract transcript from Speechmatics response format."""
        message_type = response_json.get("message")

        # Handle both final and partial transcripts
        if message_type in ["AddTranscript", "AddPartialTranscript"]:
            results = response_json.get("results", [])
            if results:
                # Speechmatics results contain alternatives with content
                transcript_parts = []
                for result in results:
                    alternatives = result.get("alternatives", [])
                    if alternatives:
                        content = alternatives[0].get("content", "")
                        if content:
                            transcript_parts.append(content)

                return " ".join(transcript_parts).strip()

        return ""
