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
        valid_models = ["default", "nova-2", "nova-3", "flux-general-en"]
        if model not in valid_models:
            raise ValueError(f"Invalid Deepgram model: {model}. Valid models: {valid_models}")
    
    def _build_websocket_url(self, sample_rate: int, channels: int) -> str:
        if self.model == "flux-general-en":
            base_url = "wss://api.preview.deepgram.com/v2/listen?model=flux-general-en&sample_rate=16000&encoding=linear16"
        else:
            base_url = (f"wss://api.deepgram.com/v1/listen?"
                    f"sample_rate={sample_rate}&"
                    f"encoding=linear16&channels={channels}&interim_results=true&vad_events=true&no_delay=true")
        
            if self.model != "default":
                base_url += f"&model={self.model}"
        
        return base_url
    
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
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
        final_flag_segments = []
        last_final_transcript_time = None
        last_end_of_turn_time = None
        
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                
                response = json.loads(message)
                current_time = time.time()
                
                # Handle Flux EndOfTurn event
                if self.model == "flux-general-en" and response.get("type") == "TurnInfo":
                    event_type = response.get("event")
                    if event_type == "EndOfTurn":
                        last_end_of_turn_time = current_time
                        print(f"[{self.name}] EndOfTurn event at {current_time - result.audio_start_time:.3f}s from audio start")
                        continue
                
                # Skip non-transcript responses
                if response.get("type") in ["SpeechStarted", "SpeechEnded", "Metadata", "Connected"]:
                    continue
                
                # Extract transcript
                transcript = self._extract_transcript(response)
                if transcript:
                    # TTFT measurement
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = current_time - result.audio_start_time
                        result.first_token_content = transcript[:30] + "..." if len(transcript) > 30 else transcript
                        print(f"[{self.name}] First transcript at {result.ttft_seconds:.3f}s")
                    
                    result.partial_transcripts.append(transcript)
                    
                    # Nova models: check for speech_final flag
                    if self.model in ["nova-2", "nova-3"]:
                        speech_final = response.get("speech_final", False)
                        
                        if speech_final:
                            final_flag_segments.append(transcript)
                            last_final_transcript_time = current_time
                            print(f"[{self.name}] speech_final at {current_time - result.audio_start_time:.3f}s from audio start")
                    
                    # Flux: collect all transcripts (EndOfTurn handled separately above)
                    elif self.model == "flux-general-en":
                        final_flag_segments = transcript
                
        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Build complete transcript
        if final_flag_segments:
            if self.model != "flux-general-en":
                result.complete_transcript = " ".join(final_flag_segments).strip()
            else:
                result.complete_transcript = transcript
        else:
            if result.partial_transcripts:
                result.complete_transcript = max(result.partial_transcripts, key=len).strip()
        
        if result.complete_transcript:
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split())
    
    def _extract_transcript(self, response_json: dict) -> str:
        if(self.model == "flux-general-en"):
            transcript = response_json.get("transcript", "").strip()
            if transcript:
                # words = response_json.get("words", [])
                # for w in words:
                #     print(f"Word: {w['word']}, Confidence: {w['confidence']}")
                # print(response_json)
                return(transcript)
        else:
            try:
                channel = response_json.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if alternatives:
                    alternative = alternatives[0]
                    # print(alternative)
                    # First, check the words array for any words
                    words = alternative.get("words", [])
                    if words:
                        # Extract text from words array and join them
                        word_texts = []
                        for word in words:
                            #print(f"Word: {word['word']}, Confidence: {word['confidence']}")
                            if isinstance(word, dict):
                                # Try punctuated_word first (if smart_format is enabled), then fallback to word
                                word_text = word.get("punctuated_word", "").strip()
                                if not word_text:
                                    word_text = word.get("word", "").strip()
                                if word_text:
                                    word_texts.append(word_text)
                        
                        if word_texts:
                            return " ".join(word_texts)
                    
                    # Fallback to transcript field if words array is empty
                    transcript = alternative.get("transcript", "").strip()
                    if transcript:
                        return transcript
                
                return ""
            except (KeyError, IndexError, TypeError):
                return ""