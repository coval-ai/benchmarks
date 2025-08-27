import time
import websockets
import json
import asyncio
import os
import base64

from secretmanager import get_secret, get_api_key, load_all_secrets

secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark


class ElevenLabs_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_api_key("ELEVENLABS_API_KEY", secrets)
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
            except:
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
        except:
            pass

        return b""

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
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream-input?model_id={self.model}"

        audio_chunks = []
        ttfa = None

        try:
            async with websockets.connect(uri) as ws:
                # Send authentication and setup (exclude from timing)
                await ws.send(json.dumps({"text": " ", "xi_api_key": self.api_key}))

                # STANDARDIZED: Start timing immediately before sending request
                start_time = time.time()

                # Send actual text request
                await ws.send(json.dumps({"text": text + " "}))
                await ws.send(json.dumps({"text": ""}))

                timeout_duration = 15
                end_time = start_time + timeout_duration
                chunk_count = 0
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=3.0)

                        if self.is_audio_chunk(message) and ttfa is None:
                            ttfa = (time.time() - start_time) * 1000
                            print(f"ElevenLabs TTFA: {ttfa:.2f} ms")

                        if self.is_audio_chunk(message):
                            audio_data = self.extract_audio_data(message)
                            if audio_data:
                                audio_chunks.append(audio_data)
                                # print(f"Received chunk {chunk_count} at {time.time() - start_time:.2f}s, size={len(audio_data)} bytes")
                                # chunk_count += 1

                        if self.is_final_message(message):
                            break

                    except asyncio.TimeoutError:
                        if audio_chunks:
                            break
                        else:
                            print("ElevenLabs: Timeout with no audio data")
                            continue
                    except websockets.exceptions.ConnectionClosed:
                        print("ElevenLabs: WebSocket connection closed")
                        break

        except Exception as e:
            print(f"ElevenLabs WebSocket error: {e}")
            import traceback

            traceback.print_exc()
            return None, None

        filename = None
        if audio_chunks:
            filename = f"elevenlabs_{self.model}_{int(time.time())}.mp3"
            with open(filename, "wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)
        else:
            print("ElevenLabs: No audio chunks received")

        return ttfa, filename
