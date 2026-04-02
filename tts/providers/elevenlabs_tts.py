import time
import websockets
import json
import asyncio
import os
import wave
import base64
from io import BytesIO

from elevenlabs import ElevenLabs

from secretmanager import get_secret, get_api_key, load_all_secrets
secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark

class ElevenLabs_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_api_key('ELEVENLABS_API_KEY', secrets)
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in .env file")

    def is_audio_chunk(self, message):
        """Standardized audio detection"""
        if isinstance(message, bytes) and len(message) > 0:
            return True

        if isinstance(message, str):
            try:
                data = json.loads(message)
                return data.get("audio") and data["audio"] is not None
            except (json.JSONDecodeError, TypeError):
                return False

        return False

    def extract_audio_data(self, message):
        """Extract audio data from message"""
        if isinstance(message, bytes):
            return message

        try:
            data = json.loads(message)
            if data.get("audio"):
                return base64.b64decode(data["audio"])
        except (json.JSONDecodeError, TypeError, Exception):
            pass

        return b''

    def is_final_message(self, message):
        """Check if message indicates completion"""
        if isinstance(message, str):
            try:
                data = json.loads(message)
                return data.get("isFinal") or data.get("done") or data.get("finished")
            except:
                pass
        return False

    async def calculateTTFA(self, text):
        
        ttfa = None

        elevenlabs = ElevenLabs(api_key=self.api_key)

        start_time = time.time()

        response = elevenlabs.text_to_speech.convert(
        voice_id=self.voice, 
        output_format="pcm_24000",
        text=text,
        model_id=self.model,
    )

        audio_stream = BytesIO()

        for chunk in response:
            if chunk:
                if self.is_audio_chunk(chunk) and ttfa is None:
                    ttfa = (time.time() - start_time) * 1000
                audio_stream.write(chunk)

        filename = None
        audio_data = audio_stream.getvalue()
        if audio_data:
            filename = f"elevenlabs_{self.model}_{int(time.time())}.wav"
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)
        else:
            print("ElevenLabs: No audio chunks received")
        
        return ttfa, filename