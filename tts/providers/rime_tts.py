import requests
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
        self.api_key = get_api_key("RIME_API_KEY", secrets)
        if not self.api_key:
            raise ValueError("RIME_API_KEY not found in .env file")

    def is_audio_chunk(self, chunk):
        """Standardized audio detection"""
        if isinstance(chunk, bytes) and len(chunk) > 0:
            return True
        return False

    async def calculateTTFA(self, text):
        audio_chunks = []
        ttfa = None

        if self.model in ["arcana", "mistv2", "mist"]:
            # Use HTTP streaming for all models
            url = "https://users.rime.ai/v1/rime-tts"

            # Setup voice validation (exclude from timing)
            try:
                voice_url = "https://users.rime.ai/data/voices/voice_details.json"
                voice_response = requests.get(voice_url)
                voice_response.raise_for_status()
                speakers = voice_response.json()

                # Validate voice for the specified model
                valid_voice = False
                for speaker in speakers:
                    if (
                        speaker["name"] == self.voice
                        and speaker["model_id"] == self.model
                    ):
                        valid_voice = True
                        break

                if not valid_voice:
                    # Find first available voice for the model
                    for speaker in speakers:
                        if speaker["model_id"] == self.model:
                            self.voice = speaker["name"]
                            print(
                                f"Voice not found for {self.model}, using {self.voice} instead"
                            )
                            break
                    else:
                        print(f"No voices found for {self.model} model")
                        return None, None

            except Exception as e:
                print(f"Rime voice validation error: {e}")
                return None, None

            # Set sampling rate based on model
            sampling_rate = 24000 if self.model == "arcana" else 22050

            payload = {
                "speaker": self.voice,
                "text": text,
                "modelId": self.model,
                "repetition_penalty": 1.5,
                "temperature": 0.5,
                "top_p": 1,
                "samplingRate": sampling_rate,
                "max_tokens": 1200,
            }

            headers = {
                "Accept": "audio/wav",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Use aiohttp for async HTTP streaming
            async with aiohttp.ClientSession() as session:
                try:
                    # STANDARDIZED: Start timing immediately before sending request
                    start_time = time.time()

                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        if response.status != 200:
                            print(f"Rime HTTP error: {response.status}")
                            error_text = await response.text()
                            print(f"Error details: {error_text}")
                            return None, None

                        chunk_count = 0
                        async for chunk in response.content.iter_chunked(1024):
                            if self.is_audio_chunk(chunk):
                                if ttfa is None:
                                    ttfa = (time.time() - start_time) * 1000
                                    print(f"Rime ({self.model}) TTFA: {ttfa:.2f} ms")

                                audio_chunks.append(chunk)
                                # print(f"Received chunk {chunk_count} at {time.time() - start_time:.5f}s, size={len(chunk)} bytes")
                                # chunk_count += 1

                except Exception as e:
                    print(f"Rime HTTP streaming error: {e}")
                    return None, None

        else:
            # Keep existing WebSocket implementation for mist/mistv2 models
            try:
                voice_url = "https://users.rime.ai/data/voices/voice_details.json"
                voice_response = requests.get(voice_url)
                voice_response.raise_for_status()
                speakers = voice_response.json()
            except Exception as e:
                print(f"Rime voice details error: {e}")
                return None, None

            valid_models = ["mistv2", "mist"]

            if self.model in valid_models:
                for speaker in speakers:
                    if (
                        speaker["name"] == self.voice
                        and speaker["model_id"] == self.model
                    ):
                        break
                else:
                    self.voice = "cove" if self.model == "mist" else "breeze"
            else:
                self.model = "mist"
                self.voice = "cove"

            import websockets

            uri = f"wss://users-ws.rime.ai/ws?speaker={self.voice}&modelId={self.model}&audioFormat=pcm&samplingRate=22050&reduceLatency=true"
            auth_headers = {"Authorization": f"Bearer {self.api_key}"}

            try:
                async with websockets.connect(
                    uri, extra_headers=auth_headers
                ) as websocket:
                    # STANDARDIZED: Start timing immediately before sending request
                    start_time = time.time()

                    await websocket.send(text)
                    await websocket.send("<EOS>")

                    chunk_count = 0
                    while True:
                        try:
                            audio = await asyncio.wait_for(
                                websocket.recv(), timeout=5.0
                            )
                            if self.is_audio_chunk(audio):
                                if ttfa is None:
                                    ttfa = (time.time() - start_time) * 1000
                                    print(f"Rime ({self.model}) TTFA: {ttfa:.2f} ms")
                                audio_chunks.append(audio)
                                print(
                                    f"Received chunk {chunk_count} at {time.time() - start_time:.2f}s, size={len(audio)} bytes"
                                )
                                chunk_count += 1
                        except asyncio.TimeoutError:
                            break
                        except websockets.exceptions.ConnectionClosed:
                            break

            except Exception as e:
                print(f"Rime WebSocket error: {e}")
                return None, None

        # Save audio file
        filename = None
        if audio_chunks:
            filename = f"rime_{self.model}_{int(time.time())}.wav"

            if self.model == "arcana":
                # HTTP streaming returns WAV format directly
                with open(filename, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
            else:
                # WebSocket returns PCM data, need to wrap in WAV
                import wave

                audio_data = b"".join(audio_chunks)

                with wave.open(filename, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(22050)
                    wav_file.writeframes(audio_data)

        return ttfa, filename
