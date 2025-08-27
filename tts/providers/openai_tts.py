import os
from openai import AsyncOpenAI
import time
import base64
import websockets
import json
import asyncio
import wave

from secretmanager import get_secret, get_api_key, load_all_secrets
secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark

class OpenAI_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.client = self.setup()
    
    def setup(self):
        api_key = get_api_key('OPENAI_API_KEY', secrets)
        if not api_key:
            raise ValueError("OpenAI API Key not found in .env file")
        return AsyncOpenAI(api_key=api_key)
    
    def is_audio_chunk(self, chunk):
        """Standardized audio detection"""
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        return False
    
    async def calculateTTFA(self, text):
        valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer"]

        if self.voice not in valid_voices:
            print(f"Voice {self.voice} cannot be found. Voice alloy used instead") 
            self.voice = "alloy"
        
        audio_chunks = []
        ttfa = None
        chunk_count = 1
        if self.model in ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]:
            # Use OpenAI's streaming API for true streaming
            start_time = time.time()
            
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="wav"
            ) as response:
                # Stream chunks as they arrive from the server
                async for chunk in response.iter_bytes():
                    if ttfa is None:
                        ttfa = (time.time() - start_time) * 1000
                        print(f"OpenAI ({self.model}) TTFA: {ttfa:.2f} ms")
                    
                    if self.is_audio_chunk(chunk):
                        audio_chunks.append(chunk)
                        # print(f"{chunk_count} {time.time() - start_time:.5f} {len(chunk)}")
                        # chunk_count += 1
        else:
            raise ValueError(f"Unsupported OpenAI model provided: {self.model}. Supported models: gpt-4o-mini-tts, tts-1, tts-1-hd")
        
        filename = None
        if audio_chunks:
            filename = f"openai_{self.model}_{int(time.time())}.wav"
            
            if self.model in ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]:
                with open(filename, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
            else:
                audio_data = b''.join(audio_chunks)
                with wave.open(filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(audio_data)
    
        return ttfa, filename if audio_chunks else None