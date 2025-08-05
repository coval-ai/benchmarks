import asyncio
import json
import time
import websockets
import logging
from .base import STTProvider, TranscriptionResult

class DeepgramProvider(STTProvider):
    def __init__(self, api_key: str, model: str = "default"):
        super().__init__(api_key, model)
        # Validate model
        valid_models = ["default", "nova-2", "nova-3"]
        if model not in valid_models:
            raise ValueError(f"Invalid Deepgram model: {model}. Valid models: {valid_models}")
    
    def _build_websocket_url(self, sample_rate: int, channels: int) -> str:
        """Build WebSocket URL with model parameter if specified."""
        base_url = (f"wss://api.deepgram.com/v1/listen?"
                   f"channels={channels}&sample_rate={sample_rate}&"
                   f"encoding=linear16&interim_results=true&vad_events=true&endpointing=true&smart_format=true")
        
        if self.model != "default":
            base_url += f"&model={self.model}"
        
        return base_url
    
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        """Deepgram-specific implementation with TTFT using first response method."""
        from wer_calculator import compare_transcription
        
        result = TranscriptionResult(provider=self.name, vad_events_count=0)
        total_start_time = time.time()
        
        try:
            # Deepgram WebSocket URL with model and headers
            websocket_url = self._build_websocket_url(sample_rate, channels)
            headers = {"Authorization": f"Token {self.api_key}"}
            close_message = {"type": "CloseStream"}
            
            async with websockets.connect(websocket_url, extra_headers=headers) as ws:
                # Create concurrent tasks for sending and receiving
                send_task = asyncio.create_task(
                    self.send_audio_chunks(ws, audio_data, channels, sample_width, 
                                         sample_rate, close_message, result, realtime_resolution)
                )
                receive_task = asyncio.create_task(
                    self._receive_responses(ws, result)
                )
                
                # Wait for both tasks to complete
                await asyncio.gather(send_task, receive_task, return_exceptions=True)
                
        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")
        
        result.total_time = time.time() - total_start_time
        
        return result
    
    async def _receive_responses(self, ws, result: TranscriptionResult):
        """Receive and process Deepgram responses using first response method for TTFT."""
        final_flag_segments = []
        last_final_transcript_time = None
        
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                
                response = json.loads(message)
                current_time = time.time()
                
                # TTFT: Record first response (any message) timing - Deepgram uses first response method
                if result.ttft_seconds is None and result.audio_start_time is not None:
                    result.ttft_seconds = current_time - result.audio_start_time
                    result.first_token_content = f"Message type: {response.get('type', 'unknown')}"
                
                # Skip non-transcript responses for transcript processing
                if response.get("type") in ["SpeechStarted", "SpeechEnded", "Metadata"]:
                    continue
                
                # Extract transcript for partial tracking
                transcript = self._extract_transcript(response)
                if transcript:
                    # Track partial transcripts
                    result.partial_transcripts.append(transcript)
                    
                    # Check for is_final flag (simplified strategy)
                    channel = response.get("channel", {})
                    is_final_deepgram = channel.get("is_final", False)
                    if not is_final_deepgram:
                        is_final_deepgram = response.get("is_final", False)
                    
                    # Collect final segments and track timing
                    if is_final_deepgram:
                        final_flag_segments.append(transcript)
                        last_final_transcript_time = current_time  # Track when last final transcript was received
                
        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Calculate audio-to-final timing
        if last_final_transcript_time and result.audio_start_time:
            result.audio_to_final_seconds = last_final_transcript_time - result.audio_start_time
        
        # Set the complete transcript using is_final strategy
        if final_flag_segments:
            result.complete_transcript = " ".join(final_flag_segments).strip()
        else:
            # Fallback to longest transcript if no final flags
            if result.partial_transcripts:
                result.complete_transcript = max(result.partial_transcripts, key=len).strip()
        
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split()) if result.complete_transcript else 0
    
    def _extract_transcript(self, response_json: dict) -> str:
        """Extract transcript from Deepgram response format."""
        try:
            channel = response_json.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if alternatives:
                return alternatives[0].get("transcript", "")
            return ""
        except (KeyError, IndexError):
            return ""