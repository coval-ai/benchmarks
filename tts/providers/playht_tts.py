import os
import aiohttp
import websockets
import time
import json
import asyncio

from secretmanager import get_secret, get_api_key, load_all_secrets

secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark


class Playht_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = get_api_key("PLAYHT_API_KEY", secrets)
        self.user_id = get_api_key("PLAYHT_USER_ID", secrets)
        if not self.api_key or not self.user_id:
            raise ValueError("PLAYHT_API_KEY and PLAYHT_USER_ID is required.")

    def is_audio_chunk(self, message):
        """Standardized audio detection"""
        if isinstance(message, bytes) and len(message) > 0:
            return True
        return False

    def is_completion_message(self, message):
        """Check if message indicates completion"""
        if not isinstance(message, str):
            return False

        try:
            data = json.loads(message)
            return data.get("done") or data.get("finished")
        except:
            return False

    async def calculateTTFA(self, text):
        audio_chunks = []
        ttfa = None

        # Setup authentication and WebSocket URL (exclude from timing)
        url = "https://api.play.ht/api/v4/websocket-auth"
        headers = {
            "accept": "audio/wav",
            "content-type": "application/json",
            "Authorization": self.api_key,
            "X-User-Id": self.user_id,
        }

        # Use aiohttp for proper async resource management
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json={}) as response:
                    response.raise_for_status()
                    auth_data = await response.json()
            except Exception as e:
                print(f"PlayHT auth error: {e}")
                return None, None

        valid_models = [
            "Play3.0-mini",
            "PlayDialog",
            "PlayDialogArabic",
            "PlayDialogMultilingual",
        ]

        if self.model not in valid_models:
            print(f"{self.model} could not be found. PlayDialog used instead.")
            self.model = "PlayDialog"

        self.voice = "s3://voice-cloning-zero-shot/b27bc13e-996f-4841-b584-4d35801aea98/original/manifest.json"
        uri = auth_data["websocket_urls"][self.model]

        try:
            async with websockets.connect(uri) as ws:
                # STANDARDIZED: Start timing immediately before sending request
                start_time = time.time()

                tts_command = {"text": text, "voice": self.voice}
                await ws.send(json.dumps(tts_command))

                timeout_duration = 10
                end_time = start_time + timeout_duration
                no_data_count = 0
                max_no_data = 5
                chunk_count = 0
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)

                        if self.is_audio_chunk(message):
                            if ttfa is None:
                                ttfa = (time.time() - start_time) * 1000
                                print(f"PlayHT TTFA: {ttfa:.2f} ms")
                            audio_chunks.append(message)
                            # print(f"Received chunk {chunk_count} at {time.time() - start_time:.2f}s, size={len(message)} bytes")
                            # chunk_count += 1
                            no_data_count = 0
                        else:
                            no_data_count += 1

                            if self.is_completion_message(message):
                                print("PlayHT: Received completion signal")
                                break

                            if no_data_count >= max_no_data:
                                print(
                                    "PlayHT: Too many non-audio messages, assuming complete"
                                )
                                break

                    except asyncio.TimeoutError:
                        print("PlayHT: Timeout waiting for message, assuming complete")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        print("PlayHT: WebSocket connection closed")
                        break

        except Exception as e:
            print(f"PlayHT WebSocket error: {e}")
            return None, None

        filename = None
        if audio_chunks:
            filename = f"playht_{self.model}_{int(time.time())}.wav"
            with open(filename, "wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)

        return ttfa, filename
