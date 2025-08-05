import asyncio
import json
import time
import websockets
import logging
from .base import STTProvider, TranscriptionResult

class AssemblyAIProvider(STTProvider):
    def __init__(self, api_key: str, model: str = "default"):
        super().__init__(api_key, model)
        # Validate model - AssemblyAI v3 streaming doesn't use model parameter in URL
        valid_models = ["universal-streaming"]
        if model not in valid_models:
            raise ValueError(f"Invalid AssemblyAI model: {model}. Valid models: {valid_models}")
    
    def _build_websocket_url(self, sample_rate: int) -> str:
        """Build WebSocket URL for AssemblyAI v3 streaming API."""
        if sample_rate != 16000:
            print(f"Warning: AssemblyAI requires 16kHz audio. Got {sample_rate}Hz - this may cause errors.")
        
        return "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000&format_turns=true"
    
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        """AssemblyAI-specific implementation with TTFT using first transcript content method."""
        from wer_calculator import compare_transcription
        
        result = TranscriptionResult(provider=self.name)
        total_start_time = time.time()
        
        try:
            # AssemblyAI WebSocket URL and headers
            websocket_url = self._build_websocket_url(sample_rate)
            headers = {"Authorization": self.api_key}
            close_message = {"type": "Terminate"}
            
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
        """Receive and process AssemblyAI responses with TTFT using first transcript content method."""
        complete_turns = []
        formatted_turns = []
        last_final_turn_time = None
        
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                
                response = json.loads(message)
                current_time = time.time()
                
                # Handle different message types
                msg_type = response.get('type')
                
                if msg_type == "Begin":
                    session_id = response.get('id')
                    continue
                
                # Extract transcript for TTFT measurement
                transcript = self._extract_transcript(response)
                if transcript:
                    # TTFT: Record first actual transcript content timing - AssemblyAI uses first content method
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = current_time - result.audio_start_time
                        result.first_token_content = transcript
                    
                    # Track partial transcripts
                    result.partial_transcripts.append(transcript)
                
                # Handle AssemblyAI v3 streaming structure
                if msg_type == "Turn":
                    # Check if this is an end of turn (final result)
                    if response.get('end_of_turn', False):
                        complete_turns.append(transcript)
                        last_final_turn_time = current_time  # Track when last final turn was received
                        
                        # Check if this looks like a formatted result (has punctuation, proper numbers)
                        is_formatted = any(char in transcript for char in ['£', '.', '-']) or transcript[0].isupper()
                        if is_formatted:
                            formatted_turns.append(transcript)
                
        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Calculate audio-to-final timing
        if last_final_turn_time and result.audio_start_time:
            result.audio_to_final_seconds = last_final_turn_time - result.audio_start_time
        
        # Build complete transcript - prefer formatted turns
        if formatted_turns:
            result.complete_transcript = " ".join(formatted_turns).strip()
        else:
            result.complete_transcript = " ".join(complete_turns).strip()
        
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split()) if result.complete_transcript else 0
    
    def _extract_transcript(self, response_json: dict) -> str:
        """Extract transcript from AssemblyAI v3 response format."""
        msg_type = response_json.get('type')
        
        # Only extract transcript from Turn messages
        if msg_type == "Turn":
            return response_json.get("transcript", "")
        
        return ""