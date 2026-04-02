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
        complete_turns = []
        formatted_turns = []
        last_final_turn_time = None
        
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                
                response = json.loads(message)
                current_time = time.time()
                
                msg_type = response.get('type')
                
                if msg_type == "Begin":
                    continue
                
                transcript = self._extract_transcript(response)
                if transcript:
                    # TTFT measurement
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = current_time - result.audio_start_time
                        result.first_token_content = transcript[:30] + "..." if len(transcript) > 30 else transcript
                        print(f"[{self.name}] First transcript at {result.ttft_seconds:.3f}s")
                    
                    result.partial_transcripts.append(transcript)
                
                # Handle Turn with end_of_turn
                if msg_type == "Turn":
                    if response.get('end_of_turn', False):
                        # Only record FIRST end_of_turn with non-empty transcript
                        if last_final_turn_time is None and transcript:
                            complete_turns.append(transcript)
                            last_final_turn_time = current_time
                            print(f"[{self.name}] FIRST end_of_turn at {current_time - result.audio_start_time:.3f}s from audio start")
                            
                            is_formatted = any(char in transcript for char in ['£', '.', '-']) or (transcript and transcript[0].isupper())
                            if is_formatted:
                                formatted_turns.append(transcript)
                        elif last_final_turn_time is not None:
                            # Log subsequent end_of_turn events but don't update timing
                            print(f"[{self.name}] Additional end_of_turn at {current_time - result.audio_start_time:.3f}s (ignoring for timing)")
                
        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Build complete transcript
        if formatted_turns:
            result.complete_transcript = " ".join(formatted_turns).strip()
        else:
            result.complete_transcript = " ".join(complete_turns).strip()
        
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
    
    def _extract_transcript(self, response_json: dict) -> str:
        """
        Extract transcript from AssemblyAI v3 response format.
        Checks both the 'transcript' field and 'words' array for content.
        """
        msg_type = response_json.get('type')
        
        # Only extract transcript from Turn messages
        if msg_type != "Turn":
            return ""
        
        # First, try the transcript field
        transcript = response_json.get("transcript", "").strip()
        if transcript:
            return transcript
        
        # If transcript is empty, check the words array
        words = response_json.get("words", [])
        if words:
            # Extract text from words array and join them
            word_texts = []
            for word in words:
                if isinstance(word, dict) and "text" in word:
                    word_text = word["text"].strip()
                    if word_text:
                        word_texts.append(word_text)
            
            if word_texts:
                return " ".join(word_texts)
        
        return ""