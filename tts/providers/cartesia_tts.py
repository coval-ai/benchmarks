import time
import wave
import os
from cartesia import AsyncCartesia

from secretmanager import get_secret, get_api_key

secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark

class Cartesia_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        if secrets:
            self.api_key = get_api_key('CARTESIA_API_KEY', secrets)
        else:
            self.api_key = os.getenv('CARTESIA_API_KEY')
        
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not found")

    def is_audio_chunk(self, chunk):
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        if hasattr(chunk, 'audio') and chunk.audio:
            return True
        if hasattr(chunk, 'data') and chunk.data:
            return True
        return False

    def extract_audio_data(self, chunk):
        if hasattr(chunk, "audio"):
            return chunk.audio
        elif hasattr(chunk, "data"):
            return chunk.data
        else:
            return chunk

    async def calculateTTFA(self, text):
        cartesia = AsyncCartesia(api_key=self.api_key)
        output_format = {"sample_rate": 24000, "container": "raw", "encoding": "pcm_s16le"} 
        
        # Setup WebSocket connection
        ws = await cartesia.tts.websocket()
        await ws.connect()
        
        start_time = time.time()
        
        gen = await ws.send(
            model_id=self.model,
            language="en",
            voice={"id": self.voice},
            output_format=output_format,
            transcript=text,
        )
        
        audio_chunks = []
        ttfa = None
        
        async for chunk in gen:
            if self.is_audio_chunk(chunk) and ttfa is None:
                ttfa = (time.time() - start_time) * 1000
            
            if self.is_audio_chunk(chunk):
                audio_chunks.append(self.extract_audio_data(chunk))

        filename = None
        if audio_chunks:
            filename = f"cartesia_{self.model}_{int(time.time())}.wav"
            audio_data = b''.join(audio_chunks)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)
        
        return ttfa, filename