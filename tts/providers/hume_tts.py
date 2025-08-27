import os
import time
import asyncio
import base64

from secretmanager import get_secret, get_api_key, load_all_secrets

secrets = get_secret("prod/benchmarking")

from hume import HumeClient, AsyncHumeClient
from hume.tts import (
    PostedUtterance,
    FormatWav,
    PostedUtteranceVoiceWithName,
    PostedUtteranceVoiceWithId,
)

try:
    from hume.types import UserInput
except ImportError:
    UserInput = str

try:
    from hume.models.social.subscribe import SubscribeEvent
except ImportError:

    class SubscribeEvent:
        def __init__(self, **kwargs):
            pass


from .base import TTS_Benchmark


class Hume_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_api_key("HUME_API_KEY", secrets)
        if not self.api_key:
            raise ValueError("HUME_API_KEY not found in .env file")

    def is_audio_chunk(self, chunk):
        """Standardized audio detection"""
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        return False

    async def calculateTTFA(self, text):
        audio_chunks = []
        ttfa = None

        if self.model == "octave-tts":
            client = HumeClient(api_key=self.api_key)

            # CRITICAL FIX: Start timing immediately before API call to match WebSocket providers
            start_time = time.time()

            response = client.tts.synthesize_file_streaming(
                utterances=[
                    PostedUtterance(
                        text=text,
                        voice=PostedUtteranceVoiceWithId(
                            id="176a55b1-4468-4736-8878-db82729667c1",
                            provider="HUME_AI",
                        ),
                    )
                ],
                format=FormatWav(),
                num_generations=1,
                instant_mode=True,
            )

            for chunk_count, chunk in enumerate(response):
                if self.is_audio_chunk(chunk) and ttfa is None:
                    ttfa = (time.time() - start_time) * 1000
                    print(f"Hume (octave-tts) TTFA: {ttfa:.2f} ms")

                if self.is_audio_chunk(chunk):
                    audio_chunks.append(chunk)
                    # print(f"Received chunk {chunk_count} at {time.time() - start_time:.5f}s, size={len(chunk)} bytes")

            filename = None
            if audio_chunks:
                filename = f"hume_{self.model}_{int(time.time())}.wav"
                with open(filename, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)

        elif self.model == "emphatic-voice-interface":
            client = AsyncHumeClient(api_key=self.api_key)

            audio_chunks = []
            conversation_complete = asyncio.Event()
            ttfa = None
            start_time = None

            async def on_message(message: SubscribeEvent):
                nonlocal audio_chunks, ttfa, start_time

                if message.type == "audio_output":
                    message_bytes = base64.b64decode(message.data.encode("utf-8"))
                    if ttfa is None and start_time is not None:
                        ttfa = (time.time() - start_time) * 1000
                        print(f"Hume (emphatic-voice) TTFA: {ttfa:.2f} ms")
                    audio_chunks.append(message_bytes)
                elif message.type == "assistant_end":
                    conversation_complete.set()
                elif message.type == "error":
                    print(f"Error ({message.code}): {message.message}")
                    conversation_complete.set()

            async def on_error(error):
                print(f"Error: {error}")
                conversation_complete.set()

            try:
                async with client.empathic_voice.chat.connect_with_callbacks(
                    on_message=on_message, on_error=on_error
                ) as socket:
                    # STANDARDIZED: Start timing immediately before sending request
                    start_time = time.time()

                    user_input_message = UserInput(
                        text=f"Speak this text exactly as provided: {text}"
                    )
                    await socket.send_user_input(user_input_message)
                    await conversation_complete.wait()

            except Exception as e:
                print(f"An error occurred: {e}")

            filename = None
            if audio_chunks:
                filename = f"hume_{self.model}_{int(time.time())}.wav"
                audio_data = b"".join(audio_chunks)

                with open(filename, "wb") as f:
                    f.write(audio_data)

        return ttfa, filename if audio_chunks else None
