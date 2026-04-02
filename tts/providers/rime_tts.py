import os
import time
import asyncio
import aiohttp

from secretmanager import get_secret, get_api_key, load_all_secrets
secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark

class Rime_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_api_key('RIME_API_KEY', secrets)
        if not self.api_key:
            raise ValueError("RIME_API_KEY not found in .env file")

    def is_audio_chunk(self, chunk):
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        return False

    async def calculateTTFA(self, text):
        audio_chunks = []
        ttfa = None

        valid_models = ["arcana", "mistv2"]
        if self.model not in valid_models:
            raise ValueError(f"Unsupported Rime model: {self.model}. Valid models: {valid_models}")

        url = "https://users.rime.ai/v1/rime-tts"
        sampling_rate = 24000

        payload = {
            "speaker": self.voice or "abbie",
            "text": text,
            "modelId": self.model,
            "repetition_penalty": 1.5,
            "temperature": 0.5,
            "top_p": 1,
            "samplingRate": sampling_rate,
            "max_tokens": 1200
        }

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()

                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        print(f"Rime HTTP error: {response.status}")
                        error_text = await response.text()
                        print(f"Error details: {error_text}")
                        return None, None

                    async for chunk in response.content.iter_any():
                        if self.is_audio_chunk(chunk):
                            if ttfa is None:
                                ttfa = (time.time() - start_time) * 1000
                                print(f"Rime ({self.model}) TTFA: {ttfa:.2f} ms")

                            audio_chunks.append(chunk)

            except Exception as e:
                print(f"Rime HTTP streaming error: {e}")
                return None, None

        # Save audio file
        filename = None
        if audio_chunks:
            filename = f"rime_{self.model}_{int(time.time())}.wav"
            import wave
            audio_data = b''.join(audio_chunks)

            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)

        return ttfa, filename
