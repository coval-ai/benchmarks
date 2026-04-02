import asyncio
import json
import time
import websockets
import base64
import logging
from .base import STTProvider, TranscriptionResult

class ElevenLabsProvider(STTProvider):
    def __init__(self, api_key: str, model: str = "scribe_v2_realtime"):
        super().__init__(api_key, model)
        # Validate model
        valid_models = ["scribe_v1", "scribe_v2_realtime"]
        if model not in valid_models:
            raise ValueError(f"Invalid ElevenLabs model: {model}. Valid models: {valid_models}")
    
    def _build_websocket_url(self) -> str:
        # ElevenLabs Realtime STT endpoint with query parameters
        # Configuration is done via query params, not initial message
        # Note: model_id is REQUIRED in query params
        params = [
            f"model_id={self.model}"
        ]
        return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{'&'.join(params)}"

    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start_time = time.time()
        
        try:
            websocket_url = self._build_websocket_url()
            # Xi-Api-Key must be passed in headers
            headers = {
                "xi-api-key": self.api_key
            }
            
            async with websockets.connect(websocket_url, extra_headers=headers) as ws:
                # Wait for session_started message
                try:
                    session_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    session_data = json.loads(session_msg)
                    if session_data.get("message_type") == "session_started":
                        print(f"[{self.name}] Session started: {session_data.get('session_id')}")
                        print(f"[{self.name}] Config: {session_data.get('config')}")
                    else:
                        print(f"[{self.name}] Unexpected first message: {session_data}")
                except asyncio.TimeoutError:
                    print(f"[{self.name}] Timeout waiting for session_started")
                except Exception as e:
                    print(f"[{self.name}] Error reading session_started: {e}")
                    raise

                # Create concurrent tasks
                send_task = asyncio.create_task(
                    self._send_elevenlabs_audio(ws, audio_data, sample_rate, 
                                              result, realtime_resolution)
                )
                receive_task = asyncio.create_task(
                    self._receive_responses(ws, result)
                )
                
                # Wait for both tasks
                await asyncio.gather(send_task, receive_task, return_exceptions=True)
                
        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")
        
        result.total_time = time.time() - total_start_time
        
        return result

    async def _send_elevenlabs_audio(self, ws, audio_data: bytes, sample_rate: int, 
                                     result: TranscriptionResult, realtime_resolution: float):
        """
        Custom sender for ElevenLabs: chunks audio, encodes to Base64, wraps in JSON.
        """
        result.audio_start_time = time.time()
        
        # Calculate chunk size (bytes = rate * seconds * channels * width)
        # Assuming 1 channel, 16-bit (2 bytes) PCM
        bytes_per_second = sample_rate * 2 
        chunk_size = int(bytes_per_second * realtime_resolution)
        
        try:
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Encode raw PCM to Base64
                b64_audio = base64.b64encode(chunk).decode("utf-8")
                
                # Use correct API format
                audio_msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": b64_audio
                }
                
                await ws.send(json.dumps(audio_msg))
                await asyncio.sleep(realtime_resolution)
            
            # Send final chunk with commit=True to finalize transcription
            if len(audio_data) > 0:
                final_msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",  # Empty audio to signal end
                    "commit": True  # Commit the final transcription
                }
                await ws.send(json.dumps(final_msg))
            
        except Exception as e:
            print(f"[{self.name}] Send Error: {e}")

    async def _receive_responses(self, ws, result: TranscriptionResult):
        full_transcript_parts = []
        
        try:
            async for message in ws:
                response = json.loads(message)
                current_time = time.time()
                
                msg_type = response.get("message_type")
                if msg_type == "partial_transcript" or msg_type == "committed_transcript":
                    print(response.get("text", "").strip())
                
                # 1. Handle Partial Transcriptions (for TTFT)
                if msg_type == "partial_transcript":
                    transcript = self._extract_transcript(response)
                    if transcript:
                        # Capture TTFT
                        if result.ttft_seconds is None and result.audio_start_time is not None:
                            result.ttft_seconds = current_time - result.audio_start_time
                            result.first_token_content = transcript
                            print(f"[{self.name}] First partial at {result.ttft_seconds:.3f}s")
                        
                        # Store current partial
                        result.partial_transcripts.append(transcript)

                # 2. Handle Committed Transcriptions (final)
                elif msg_type == "committed_transcript":
                    transcript = self._extract_transcript(response)
                    if transcript:
                        full_transcript_parts.append(transcript)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = current_time - result.audio_start_time
                        print(f"[{self.name}] Committed segment: {transcript}")

                # 3. Handle Committed Transcriptions with Timestamps
                elif msg_type == "committed_transcript_with_timestamps":
                    transcript = self._extract_transcript(response)
                    if transcript:
                        full_transcript_parts.append(transcript)
                        if result.audio_start_time is not None:
                            result.audio_to_final_seconds = current_time - result.audio_start_time
                        print(f"[{self.name}] Committed (with timestamps): {transcript}")

                # 4. Handle various error types
                elif msg_type in ["scribe_error", "scribe_auth_error", "scribe_quota_exceeded_error",
                                 "scribe_throttled_error", "scribe_unaccepted_terms_error",
                                 "scribe_rate_limited_error", "scribe_queue_overflow_error",
                                 "scribe_resource_exhausted_error", "scribe_session_time_limit_exceeded_error",
                                 "scribe_input_error", "scribe_chunk_size_exceeded_error",
                                 "scribe_insufficient_audio_activity_error", "scribe_transcriber_error"]:
                    error_msg = response.get("message", response.get("error", "Unknown error"))
                    print(f"[{self.name}] API Error ({msg_type}): {error_msg}")
                    result.error = f"{msg_type}: {error_msg}"
                    break

        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Construct final transcript
        # If we have committed parts, join them. If not, use the last partial.
        if full_transcript_parts:
            result.complete_transcript = " ".join(full_transcript_parts).strip()
        elif result.partial_transcripts:
            # Use the last partial as fallback
            result.complete_transcript = result.partial_transcripts[-1].strip()
            
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())

    def _extract_transcript(self, response_json: dict) -> str:
        """
        Extracts text from ElevenLabs JSON response.
        Structure: { "message_type": "...", "text": "Hello world", ... }
        """
        return response_json.get("text", "").strip()