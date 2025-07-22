import asyncio
import json
import time
import websockets
import traceback
import wave
import pandas as pd
import logging
import os
import jiwer
from jiwer import transforms
import io
import librosa
import soundfile as sf
from datasets import load_dataset
import re

import concurrent.futures
import csv
from datetime import datetime
from sqlalchemy import create_engine
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from huggingface_hub import login
login(token="")

from wer_calculator import compare_transcription

# Create WER normalization transforms - combine into a single callable
WER_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.RemovePunctuation(),
    transforms.ToLowerCase(),
    transforms.RemoveMultipleSpaces(),
    transforms.Strip(),
    transforms.ExpandCommonEnglishContractions(),
    transforms.RemoveEmptyStrings(),
    transforms.ReduceToListOfListOfWords()
])

# Ground truth for WER calculation (will be updated dynamically)
GROUND_TRUTH = "For orders over £500, shipping is free when you use promo code SHIP123 or call our order desk at 02079460371."

# Set Google credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './google-credentials.json'

# Import Google Cloud Speech client
try:
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: google-cloud-speech not installed. GoogleProvider will not work.")
    print("Install with: pip install google-cloud-speech")

@dataclass
class TranscriptionResult:
    provider: str
    
    # Core timing measurements
    ttft_seconds: Optional[float] = None  # Time to First Token
    total_time: Optional[float] = None    # Total processing time
    audio_to_final_seconds: Optional[float] = None  # Audio start to final transcript
    rtf_value: Optional[float] = None     # Real-Time Factor (audio_duration / audio_to_final)
    
    # Content tracking
    first_token_content: Optional[str] = None        # What triggered TTFT
    complete_transcript: Optional[str] = None        # Final transcript
    partial_transcripts: List[str] = field(default_factory=list)  # All partials
    
    # Transcript metrics
    transcript_length: Optional[int] = None  # Character count
    word_count: Optional[int] = None         # Word count
    wer_percentage: Optional[float] = None   # Word Error Rate as percentage
    
    # Error handling
    error: Optional[str] = None
    
    # Internal timing (for calculations)
    audio_start_time: Optional[float] = None
    
    # Deepgram-specific VAD data (bonus metrics)
    vad_first_detected: Optional[float] = None       # First VAD event timing
    vad_events_count: Optional[int] = None           # Number of VAD events
    vad_first_event_content: Optional[str] = None    # Content of first VAD event

class STTProvider(ABC):
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
        # Create provider name including model
        base_name = self.__class__.__name__.replace('Provider', '').lower()
        if model == "default":
            self.name = base_name
        else:
            self.name = f"{base_name}-{model}"
    
    @abstractmethod
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        """
        Measure TTFT for this provider.
        Each provider implements their own connection and response handling logic.
        """
        pass
    
    async def send_audio_chunks(self, ws, audio_data: bytes, channels: int, 
                              sample_width: int, sample_rate: int, close_message: dict,
                              result: TranscriptionResult, realtime_resolution: float = 0.1):
        """
        Send audio chunks with consistent real-time timing simulation.
        This ensures identical timing across all providers for fair benchmarking.
        """
        byte_rate = sample_width * sample_rate * channels
        data_copy = audio_data
        first_chunk = True
        
        try:
            while len(data_copy):
                # Calculate chunk size for realtime_resolution seconds of audio
                chunk_size = int(byte_rate * realtime_resolution)
                chunk, data_copy = data_copy[:chunk_size], data_copy[chunk_size:]
                
                # Start timing when we send the first audio chunk
                if first_chunk:
                    result.audio_start_time = time.time()
                    first_chunk = False
                
                # Send chunk
                await ws.send(chunk)
                
                # Wait to simulate real-time
                await asyncio.sleep(realtime_resolution)
            
            # Send provider-specific close message
            await ws.send(json.dumps(close_message))
            
        except Exception as e:
            print(f"Send error for {self.name}: {e}")
            raise

# Deepgram Provider Implementation with TTFT using first response method
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
        
        # Calculate WER and RTF if transcript is available
        if result.complete_transcript:
            try:
                wer_analysis = compare_transcription(GROUND_TRUTH, result.complete_transcript)
                custom_wer = wer_analysis['wer']
                
                if custom_wer is not None:
                    result.wer_percentage = custom_wer * 100
                    
                    # Log detailed error analysis for debugging when there are differences
                    if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                        logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
                else:
                    print(f"Error calculating WER for {self.name}: {e}")
            except Exception as e:
                print(f"Error calculating WER for {self.name}: {e}")
                result.wer_percentage = None
        
        # Calculate RTF if audio-to-final timing is available
        if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
            result.rtf_value = audio_duration / result.audio_to_final_seconds
        
        # Print transcript to console with WER, Audio-to-Final timing, and RTF
        if result.complete_transcript:
            wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
            audio_to_final_text = f" (Audio→Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
            rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
            
            print(f"\n📝 {self.name.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
            print(f"   {result.complete_transcript}")
        else:
            print(f"\n📝 {self.name.upper()} TRANSCRIPT: [EMPTY]")
        
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

# AssemblyAI Provider Implementation with TTFT using first transcript content method
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
        
        # Calculate WER and RTF if transcript is available
        if result.complete_transcript:
            try:
                wer_analysis = compare_transcription(GROUND_TRUTH, result.complete_transcript)
                custom_wer = wer_analysis['wer']
                
                if custom_wer is not None:
                    result.wer_percentage = custom_wer * 100
                    
                    # Log detailed error analysis for debugging when there are differences
                    if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                        logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
                else:
                    print(f"Error calculating WER for {self.name}: {e}")
            except Exception as e:
                print(f"Error calculating WER for {self.name}: {e}")
                result.wer_percentage = None
        
        # Calculate RTF if audio-to-final timing is available
        if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
            result.rtf_value = audio_duration / result.audio_to_final_seconds
        
        # Print transcript to console with WER, Audio-to-Final timing, and RTF
        if result.complete_transcript:
            wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
            audio_to_final_text = f" (Audio→Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
            rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
            
            print(f"\n📝 {self.name.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
            print(f"   {result.complete_transcript}")
        else:
            print(f"\n📝 {self.name.upper()} TRANSCRIPT: [EMPTY]")
        
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

# Speechmatics Provider Implementation with TTFT using first transcript content method
class SpeechmaticsProvider(STTProvider):
    def __init__(self, api_key: str, model: str = "default"):
        super().__init__(api_key, model)
        # Speechmatics models - removed telephony as it's not supported
        valid_models = ["default", "enhanced", "broadcast"]
        if model not in valid_models:
            raise ValueError(f"Invalid Speechmatics model: {model}. Valid models: {valid_models}")
    
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
                "sample_rate": sample_rate
            }
        }
    
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        """Speechmatics-specific implementation with TTFT using first transcript content method."""
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
                    self._send_audio_chunks_speechmatics_timed(ws, audio_data, channels, sample_width, 
                                                             sample_rate, result, realtime_resolution)
                )
                receive_task = asyncio.create_task(
                    self._receive_responses(ws, result)
                )
                
                # Wait for both tasks to complete
                await asyncio.gather(send_task, receive_task, return_exceptions=True)
                
        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")
        
        # Calculate WER and RTF if transcript is available
        if result.complete_transcript:
            try:
                wer_analysis = compare_transcription(GROUND_TRUTH, result.complete_transcript)
                custom_wer = wer_analysis['wer']
                
                if custom_wer is not None:
                    result.wer_percentage = custom_wer * 100
                    
                    # Log detailed error analysis for debugging when there are differences
                    if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                        logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
                else:
                    print(f"Error calculating WER for {self.name}: {e}")
            except Exception as e:
                print(f"Error calculating WER for {self.name}: {e}")
                result.wer_percentage = None
        
        # Calculate RTF if audio-to-final timing is available
        if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
            result.rtf_value = audio_duration / result.audio_to_final_seconds
        
        # Print transcript to console with WER, Audio-to-Final timing, and RTF
        if result.complete_transcript:
            wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
            audio_to_final_text = f" (Audio→Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
            rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
            
            print(f"\n📝 {self.name.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
            print(f"   {result.complete_transcript}")
        else:
            print(f"\n📝 {self.name.upper()} TRANSCRIPT: [EMPTY]")
        
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
                raise Exception(f"Speechmatics error: {response.get('reason', 'Unknown error')}")
    
    async def _send_audio_chunks_speechmatics_timed(self, ws, audio_data: bytes, channels: int, 
                                                  sample_width: int, sample_rate: int,
                                                  result: TranscriptionResult, realtime_resolution: float = 0.1):
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
            end_message = {
                "message": "EndOfStream",
                "last_seq_no": seq_no
            }
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
                        error_reason = response.get('reason', 'Unknown error')
                        if 'EndOfStream' not in error_reason and 'schema' not in error_reason.lower():
                            result.error = error_reason
                    continue
                
                # Extract transcript for TTFT measurement
                transcript = self._extract_transcript(response)
                if transcript:
                    # TTFT: Record first actual transcript content timing - Speechmatics uses first content method
                    if result.ttft_seconds is None and result.audio_start_time is not None:
                        result.ttft_seconds = current_time - result.audio_start_time
                        result.first_token_content = transcript
                    
                    # Track partial transcripts
                    result.partial_transcripts.append(transcript)
                
                # For complete transcript, only collect AddTranscript messages (final results)
                if message_type == "AddTranscript":
                    final_transcripts.append(transcript)
                    last_final_transcript_time = current_time  # Track when last final transcript was received
                
        except Exception as e:
            print(f"Receive error for {self.name}: {e}")
        
        # Calculate audio-to-final timing
        if last_final_transcript_time and result.audio_start_time:
            result.audio_to_final_seconds = last_final_transcript_time - result.audio_start_time
        
        # Build complete transcript from final results only
        if final_transcripts:
            result.complete_transcript = " ".join(final_transcripts).strip()
            result.transcript_length = len(result.complete_transcript)
            result.word_count = len(result.complete_transcript.split()) if result.complete_transcript else 0
    
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


# Updated Google Speech-to-Text Provider Implementation with v2 API
class GoogleProvider(STTProvider):
    def __init__(self, api_key: str = "service_account", model: str = "default"):
        super().__init__(api_key, model)
        
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-cloud-speech package is required for GoogleProvider")
        
        # Validate model for v2 API
        valid_models = ["default", "short", "long", "telephony", "chirp_2"]
        if model not in valid_models:
            raise ValueError(f"Invalid Google model: {model}. Valid models: {valid_models}")
        
        # Initialize Google Speech v2 client with us-central1 endpoint
        self.client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint="us-central1-speech.googleapis.com",
            )
        )
    
    def _get_model_name(self) -> str:
        """Get the Google v2 model name for API calls."""
        if self.model == "default":
            return "chirp_2"
        # For v2 API, model names are used directly
        return self.model
    
    def _get_recognizer_name(self) -> str:
        """Build the recognizer resource name for v2 API."""
        return f"projects/stt-benchmarking-464409/locations/us-central1/recognizers/_"
    
    async def measure_ttft(self, audio_data: bytes, channels: int, 
                          sample_width: int, sample_rate: int,
                          realtime_resolution: float = 0.1, audio_duration: float = None) -> TranscriptionResult:
        """Google v2 API-specific implementation with TTFT using first transcript content method."""
        result = TranscriptionResult(provider=self.name)
        total_start_time = time.time()
        
        try:
            # Configure the streaming recognition request for v2 API
            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=channels,
                ),
                language_codes=["en-US"],
                model=self._get_model_name(),
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                ),
            )
            
            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )
            
            # Run Google's v2 streaming recognition in executor to handle sync API
            def run_streaming_recognition():
                final_transcripts = []
                last_final_result_time = None
                
                try:
                    # Create all requests upfront to avoid generator issues
                    requests = []
                    
                    # First request with config
                    config_request = cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._get_recognizer_name(),
                        streaming_config=streaming_config
                    )
                    requests.append(config_request)
                    
                    # Prepare audio chunks
                    data_copy = audio_data
                    byte_rate = sample_width * sample_rate * channels
                    first_chunk = True
                    
                    while len(data_copy):
                        chunk_size = int(byte_rate * realtime_resolution)
                        chunk, data_copy = data_copy[:chunk_size], data_copy[chunk_size:]
                        
                        # Record start time when preparing first audio chunk
                        if first_chunk and chunk:
                            result.audio_start_time = time.time()
                            first_chunk = False
                        
                        # Create audio request
                        audio_request = cloud_speech.StreamingRecognizeRequest(audio=chunk)
                        requests.append(audio_request)
                    
                    # Use iterator over the pre-built requests list
                    def request_iterator():
                        for req in requests:
                            yield req
                            if req.audio:  # If this is an audio chunk, simulate real-time
                                time.sleep(realtime_resolution)
                    
                    # Call streaming_recognize with the iterator
                    responses = self.client.streaming_recognize(requests=request_iterator())
                    
                    # Process responses with TTFT timing
                    for response in responses:
                        current_time = time.time()
                        
                        transcript = self._extract_transcript(response)
                        if transcript:
                            # TTFT: Record first actual transcript content timing
                            if result.ttft_seconds is None and result.audio_start_time is not None:
                                result.ttft_seconds = current_time - result.audio_start_time
                                result.first_token_content = transcript
                            
                            # Track partial transcripts
                            result.partial_transcripts.append(transcript)
                        
                        # For complete transcript, only collect final results (is_final=True)
                        if hasattr(response, 'results') and response.results:
                            for result_item in response.results:
                                if getattr(result_item, 'is_final', False):
                                    final_transcripts.append(transcript)
                                    last_final_result_time = current_time
                                    break
                    
                    # Calculate audio-to-final timing
                    if last_final_result_time and result.audio_start_time:
                        result.audio_to_final_seconds = last_final_result_time - result.audio_start_time
                    
                    # Build complete transcript from final results only
                    if final_transcripts:
                        result.complete_transcript = " ".join(final_transcripts).strip()
                        result.transcript_length = len(result.complete_transcript)
                        result.word_count = len(result.complete_transcript.split()) if result.complete_transcript else 0
                        
                except Exception as e:
                    print(f"Inner streaming error: {e}")
                    raise
            
            # Run in thread pool to integrate with async code
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, run_streaming_recognition)
                        
        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")
            traceback.print_exc()
        
        # Calculate WER and RTF if transcript is available
        if result.complete_transcript:
            try:
                wer_analysis = compare_transcription(GROUND_TRUTH, result.complete_transcript)
                custom_wer = wer_analysis['wer']
                
                if custom_wer is not None:
                    result.wer_percentage = custom_wer * 100
                    
                    # Log detailed error analysis for debugging when there are differences
                    if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                        logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
                else:
                    print(f"Error calculating WER for {self.name}: {e}")
            except Exception as e:
                print(f"Error calculating WER for {self.name}: {e}")
                result.wer_percentage = None
        
        # Calculate RTF if audio-to-final timing is available
        if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
            result.rtf_value = audio_duration / result.audio_to_final_seconds
        
        # Print transcript to console with WER, Audio-to-Final timing, and RTF
        if result.complete_transcript:
            wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
            audio_to_final_text = f" (Audio→Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
            rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
            
            print(f"\n📝 {self.name.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
            print(f"   {result.complete_transcript}")
        else:
            print(f"\n📝 {self.name.upper()} TRANSCRIPT: [EMPTY]")
        
        return result
    
    def _extract_transcript(self, response) -> str:
        """Extract transcript from Google Speech v2 response."""
        try:
            if not response.results:
                return ""
            
            result = response.results[0]
            if not result.alternatives:
                return ""
            
            return result.alternatives[0].transcript
            
        except (AttributeError, IndexError):
            return ""

# Common Voice Integration Functions
def load_common_voice_sample(min_duration: float = 2.0, max_duration: float = 15.0, max_retries: int = 10) -> tuple[bytes, int, int, int, str, float, str]:
    """
    Load a random sample from Common Voice English test set.
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds, ground_truth)
    """
    print(f"🔍 Loading Common Voice dataset (English test split)...")
    
    # Load dataset in streaming mode (no download)
    try:
        dataset = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="test", streaming=True, trust_remote_code=True)
        print("✅ Dataset connection established")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise
    
    # Try to find a suitable sample
    for attempt in range(max_retries):
        try:
            # Get a random sample
            sample = next(iter(dataset.shuffle(seed=None).take(1)))
            
            # Get audio data and info
            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            transcript = sample['sentence']
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            # Check if duration is in our range
            if min_duration <= duration <= max_duration:
                # Convert to WAV format with consistent 16kHz sample rate
                audio_data = convert_to_wav_bytes(audio_array, sample_rate, target_sample_rate=16000)
                
                # Clean transcript for ground truth
                ground_truth = clean_transcript(transcript)
                
                # Generate filename
                filename = f"common_voice_sample_{attempt+1}.wav"
                
                print(f"✅ Selected sample: {duration:.2f}s, {len(ground_truth.split())} words")
                print(f"📝 Ground truth: \"{ground_truth}\"")
                
                # Return in same format as original load_wav_file - force 16kHz sample rate
                return audio_data, 1, 2, 16000, filename, duration, ground_truth
            else:
                print(f"⏭️  Sample {attempt+1}: {duration:.2f}s (outside {min_duration}-{max_duration}s range)")
                
        except Exception as e:
            print(f"⚠️  Attempt {attempt+1} failed: {e}")
            continue
    
    raise Exception(f"Failed to find suitable sample after {max_retries} attempts")

def convert_to_wav_bytes(audio_array, original_sample_rate: int, target_sample_rate: int = 16000) -> bytes:
    """
    Convert audio array to WAV bytes format expected by STT providers.
    Always resamples to target_sample_rate (default 16kHz) for consistency.
    """
    print(f"🎵 Audio conversion: {original_sample_rate}Hz → {target_sample_rate}Hz")
    
    # Always resample to ensure consistent sample rate across all providers
    if original_sample_rate != target_sample_rate:
        print(f"   Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz...")
        audio_array = librosa.resample(audio_array, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    else:
        print(f"   No resampling needed (already {target_sample_rate}Hz)")
    
    # Convert to bytes using soundfile
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_array, target_sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        # Skip WAV header (44 bytes) to get raw PCM data
        wav_bytes = wav_buffer.read()
        pcm_data = wav_bytes[44:]  # Skip WAV header
        
        print(f"   Converted to {len(pcm_data)} bytes of PCM data")
        return pcm_data

def clean_transcript(transcript: str) -> str:
    """
    Clean and normalize transcript text for WER calculation.
    """
    # Convert to lowercase
    transcript = transcript.lower()
    
    # Remove extra whitespace
    transcript = ' '.join(transcript.split())
    
    # REMOVED: All regex operations to eliminate errors
    # Basic string cleaning only
    
    return transcript.strip()

# Utility Functions
def save_results_to_csv(results: List[TranscriptionResult], timestamp: str, audio_filename: str = "test.wav"):
    """Save enhanced benchmark results to CSV file in long format with TTFT, WER and RTF metrics."""
    filename = "all_benchmarks.csv"
    
    # Prepare CSV headers for long format
    headers = [
        'provider',
        'model', 
        'voice',
        'benchmark',
        'metric_type',
        'metric_value',
        'metric_units',
        'audio_filename',
        'timestamp',
        'status'
    ]
    
    # Prepare data rows in long format
    rows = []
    readable_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for result in results:
        if isinstance(result, Exception):
            continue  # Skip exception results
        
        # Parse provider and model from provider name
        if '-' in result.provider:
            provider, model = result.provider.split('-', 1)
        else:
            provider = result.provider
            model = "default"
        
        # Determine status
        status = "failed" if result.error else "success"
        
        # Base row template
        base_row = [
            provider,
            f"{provider} {model}",
            "N/A",  # voice
            "STT",  # benchmark
            None,   # metric_type (to be filled)
            None,   # metric_value (to be filled)
            "s",    # metric_units (default for timing)
            audio_filename,
            readable_timestamp,
            status
        ]
        
        # Add TTFT metric
        if result.ttft_seconds is not None:
            ttft_row = base_row.copy()
            ttft_row[4] = "TTFT"
            ttft_row[5] = result.ttft_seconds
            rows.append(ttft_row)
        
        # Add WER metric
        if result.wer_percentage is not None:
            wer_row = base_row.copy()
            wer_row[4] = "WER"
            wer_row[5] = result.wer_percentage
            wer_row[6] = "%"  # percentage units
            rows.append(wer_row)
        
        # Add RTF metric
        if result.rtf_value is not None:
            rtf_row = base_row.copy()
            rtf_row[4] = "RTF"
            rtf_row[5] = result.rtf_value
            rtf_row[6] = None  # NULL units
            rows.append(rtf_row)
    
    # Write to CSV
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"\n📄 Results saved to: {filename}")
        print(f"📊 Contains {len(rows)} metric rows (TTFT, WER & RTF) from {len([r for r in results if not isinstance(r, Exception)])} providers")
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

def load_wav_file(file_path: str) -> tuple[bytes, int, int, int, str, float]:
    """
    Load WAV file and return audio data with parameters, filename, and duration.
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds)
    """
    import os
    filename = os.path.basename(file_path)
    
    with wave.open(file_path, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        audio_data = wav_file.readframes(frames)
        
        # Calculate duration in seconds
        duration_seconds = frames / sample_rate
        
        print(f"Loaded audio: {channels} channels, {sample_rate}Hz, {sample_width} bytes/sample, {len(audio_data)} bytes, {duration_seconds:.2f}s duration")
        return audio_data, channels, sample_width, sample_rate, filename, duration_seconds

def display_results_table(results: List[TranscriptionResult]):
    """Display comprehensive results table with TTFT timing metric and WER."""
    print("\n" + "="*140)
    print("STT PROVIDER/MODEL ENHANCED BENCHMARK RESULTS")
    print("="*140)
    print(f"{'Provider/Model':<20} {'TTFT (s)':<8} {'WER (%)':<8} {'Total (s)':<9} {'VAD (s)':<8} {'First Token':<30} {'Error':<15}")
    print("-"*140)
    
    successful_ttft = []
    successful_wer = []
    failed_results = []
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception: {result}")
            continue
        
        provider_name = result.provider
        
        # Format timing values
        ttft = f"{result.ttft_seconds:.3f}" if result.ttft_seconds else "N/A"
        wer = f"{result.wer_percentage:.1f}" if result.wer_percentage is not None else "N/A"
        total = f"{result.total_time:.3f}" if result.total_time else "N/A"
        vad = f"{result.vad_first_detected:.3f}" if result.vad_first_detected else "N/A"
        
        # Format content strings
        first_token = (result.first_token_content[:27] + "...") if result.first_token_content and len(result.first_token_content) > 30 else (result.first_token_content or "")
        error = result.error[:12] + "..." if result.error and len(result.error) > 15 else (result.error or "")
        
        print(f"{provider_name:<20} {ttft:<8} {wer:<8} {total:<9} {vad:<8} {first_token:<30} {error:<15}")
        
        # Collect successful results for analysis
        if not result.error:
            if result.ttft_seconds:
                successful_ttft.append((provider_name, result.ttft_seconds))
            if result.wer_percentage is not None:
                successful_wer.append((provider_name, result.wer_percentage))
        else:
            failed_results.append(provider_name)
    
    return successful_ttft, successful_wer, failed_results

def display_analysis(successful_ttft: List[tuple], successful_wer: List[tuple], failed_results: List[str]):
    """Display detailed analysis of the TTFT results and WER."""
    print("\n" + "="*140)
    print("DETAILED ANALYSIS")
    print("="*140)
    
    # TTFT Analysis
    if successful_ttft:
        successful_ttft.sort(key=lambda x: x[1])
        fastest_ttft = successful_ttft[0]
        print(f"🏆 Fastest TTFT (Time to First Token): {fastest_ttft[0]} ({fastest_ttft[1]:.3f}s)")
        
        if len(successful_ttft) > 1:
            avg_ttft = sum(result[1] for result in successful_ttft) / len(successful_ttft)
            print(f"📊 Average TTFT: {avg_ttft:.3f}s")
            
            print("\n🎯 TTFT Rankings (Time to First Token):")
            for i, (name, time_val) in enumerate(successful_ttft[:5], 1):
                print(f"   {i}. {name}: {time_val:.3f}s")
    
    # WER Analysis
    if successful_wer:
        successful_wer.sort(key=lambda x: x[1])  # Sort by WER (lower is better)
        best_wer = successful_wer[0]
        print(f"\n🎯 Best WER (Accuracy): {best_wer[0]} ({best_wer[1]:.1f}%)")
        
        if len(successful_wer) > 1:
            avg_wer = sum(result[1] for result in successful_wer) / len(successful_wer)
            print(f"📊 Average WER: {avg_wer:.1f}%")
            
            print("\n🎯 WER Rankings (Accuracy - Lower is Better):")
            for i, (name, wer_val) in enumerate(successful_wer[:5], 1):
                print(f"   {i}. {name}: {wer_val:.1f}%")
    
    # Provider Comparisons
    print("\n📈 Model Comparisons within Providers:")
    
    # Deepgram comparisons
    deepgram_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('deepgram')]
    deepgram_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('deepgram')]
    
    if deepgram_ttft:
        deepgram_ttft.sort(key=lambda x: x[1])
        print(f"   Deepgram TTFT fastest: {deepgram_ttft[0][0]} ({deepgram_ttft[0][1]:.3f}s)")
    if deepgram_wer:
        deepgram_wer.sort(key=lambda x: x[1])
        print(f"   Deepgram WER best: {deepgram_wer[0][0]} ({deepgram_wer[0][1]:.1f}%)")
    
    # Google comparisons
    google_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('google')]
    google_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('google')]
    
    if google_ttft:
        google_ttft.sort(key=lambda x: x[1])
        print(f"   Google TTFT fastest: {google_ttft[0][0]} ({google_ttft[0][1]:.3f}s)")
    if google_wer:
        google_wer.sort(key=lambda x: x[1])
        print(f"   Google WER best: {google_wer[0][0]} ({google_wer[0][1]:.1f}%)")
    
    # Speechmatics comparisons
    speechmatics_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('speechmatics')]
    speechmatics_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('speechmatics')]
    
    if speechmatics_ttft:
        speechmatics_ttft.sort(key=lambda x: x[1])
        print(f"   Speechmatics TTFT fastest: {speechmatics_ttft[0][0]} ({speechmatics_ttft[0][1]:.3f}s)")
    if speechmatics_wer:
        speechmatics_wer.sort(key=lambda x: x[1])
        print(f"   Speechmatics WER best: {speechmatics_wer[0][0]} ({speechmatics_wer[0][1]:.1f}%)")
    
    # Failed results
    if failed_results:
        print(f"\n❌ Failed: {', '.join(failed_results)}")

# Main benchmark execution
async def main():
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize providers with different models
    providers = [
        # Deepgram models (only nova variants)
        DeepgramProvider(api_key="", model="nova-2"),
        DeepgramProvider(api_key="", model="nova-3"),
        
        # AssemblyAI models (v3 streaming API)
        AssemblyAIProvider(api_key="", model="universal-streaming"),
        
        # Speechmatics models (reduced to avoid quota issues)
        SpeechmaticsProvider(api_key="", model="default"),
        SpeechmaticsProvider(api_key="", model="enhanced"),
        
        # Google Speech-to-Text models (only include if available)
    ]
    
    # Add Google providers only if the library is available
    if GOOGLE_AVAILABLE:
        providers.extend([
            GoogleProvider(model="short"),
            GoogleProvider(model="long"), 
            GoogleProvider(model="telephony"),
            GoogleProvider(model="chirp_2"),
        ])
    
    # Load random audio sample from Common Voice instead of fixed file
    try:
        audio_data, channels, sample_width, sample_rate, audio_filename, audio_duration, ground_truth = load_common_voice_sample()
        
        # Update global ground truth for this run
        global GROUND_TRUTH
        GROUND_TRUTH = ground_truth
        
    except Exception as e:
        print(f"❌ Failed to load Common Voice sample: {e}")
        print("📁 Falling back to local test.wav file")
        audio_data, channels, sample_width, sample_rate, audio_filename, audio_duration = load_wav_file("test.wav")
        ground_truth = GROUND_TRUTH  # Use original hardcoded value
    
    print(f"\nStarting Enhanced Benchmark with {len(providers)} provider/model combinations...")
    print("TTFT Timing System:")
    print("  🎯 TTFT (Time to First Token):")
    print("  🎯 WER (Word Error Rate): Accuracy measurement vs ground truth")
    print("\nModels being tested:")
    for provider in providers:
        print(f"  - {provider.name}")
    
    print(f"\n🎯 Ground Truth: \"{ground_truth}\"")
    print(f"📏 Audio Length: {audio_duration:.2f}s")
    print(f"📏 Ground Truth Length: {len(ground_truth.split())} words, {len(ground_truth)} characters")
    print(f"🎵 Audio Source: {audio_filename}")
    
    # Run all providers concurrently
    tasks = [
        provider.measure_ttft(audio_data, channels, sample_width, sample_rate, 0.1, audio_duration)
        for provider in providers
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Display results and analysis
    successful_ttft, successful_wer, failed_results = display_results_table(results)
    display_analysis(successful_ttft, successful_wer, failed_results)
    
    print(f"\n✅ Completed enhanced benchmark for {len(providers)} provider/model combinations")
    
    # Save results to CSV
    save_results_to_csv(results, timestamp, audio_filename)

    try:
        df = pd.read_csv("all_benchmarks.csv")
        engine = create_engine("postgresql://neondb_owner:npg_DI4ulRe5rUVN@ep-plain-king-a4t6r8tz-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require")
        df.to_sql('all_benchmarks', engine, if_exists='append', index=False)
        print("Data uploaded to database.")
    except Exception as e:
        logging.error(f"Error writing results to database: {e}")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   • Total providers tested: {len(providers)}")
    print(f"   • Successful TTFT measurements: {len(successful_ttft)}")
    print(f"   • Successful WER measurements: {len(successful_wer)}")
    print(f"   • Failed providers: {len(failed_results)}")
    print(f"   • Sample source: Common Voice EN test set")
    
    if successful_ttft:
        ttft_winner = min(successful_ttft, key=lambda x: x[1])
        print(f"   • TTFT winner: {ttft_winner[0]} ({ttft_winner[1]:.3f}s)")
    
    if successful_wer:
        wer_winner = min(successful_wer, key=lambda x: x[1])
        print(f"   • Accuracy winner: {wer_winner[0]} ({wer_winner[1]:.1f}% WER)")

if __name__ == "__main__":
    # Run enhanced benchmark
    asyncio.run(main())