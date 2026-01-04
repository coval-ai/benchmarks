import time
import wave
import asyncio
import os

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import SpeakV1SocketClientResponse, SpeakV1TextMessage, SpeakV1ControlMessage

from secretmanager import get_secret, get_api_key
secrets = get_secret("prod/benchmarking")

from .base import TTS_Benchmark

class Deepgram_Benchmark(TTS_Benchmark):
    def __init__(self, config):
        super().__init__(config)
        if secrets:
            self.api_key = get_api_key('DEEPGRAM_API_KEY', secrets)
        else:
            self.api_key = os.getenv('DEEPGRAM_API_KEY')
        
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not found")

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
        deepgram = DeepgramClient(api_key=self.api_key)
        
        with deepgram.speak.v1.connect(
            model=self.model,
            encoding="linear16",
            sample_rate=24000
        ) as dg_connection:
        
            audio_chunks = []
            ttfa = None
            start_time = None
            
            # Event handlers
            def on_open(open_result, **kwargs):
                #print(f"[EVENT] OPEN: {open_result}")
                print(f"Connection open")
            
            def on_message(message, **kwargs):
                # Handles both binary audio data and message objects
                nonlocal ttfa, audio_chunks, start_time
                
                # Handle binary audio data
                if isinstance(message, bytes) and len(message) > 0:
                    
                    # Stop time at first audio chunk
                    if ttfa is None and start_time is not None:
                        ttfa = (time.time() - start_time) * 1000
                        #print(f"Deepgram TTFA: {ttfa:.2f} ms")
                    
                    audio_chunks.append(message)
                else:
                    msg_type = getattr(message, "type", "Unknown")
                    #print(f"[EVENT] {msg_type}: {message}")
                    
                    # Check if this is the flushed event
                    if hasattr(message, "type") and message.type == "Flushed":
                        #print(f"Audio generation complete")
                        dg_connection.send_control(SpeakV1ControlMessage(type="Close"))
            
            def on_error(error, **kwargs):
                print(f"[EVENT] ERROR: {error}")
            
            def on_close(close_result, **kwargs):
                print(f"Connection closed")

            dg_connection.on(EventType.OPEN, on_open)
            dg_connection.on(EventType.MESSAGE, on_message)
            dg_connection.on(EventType.ERROR, on_error)
            dg_connection.on(EventType.CLOSE, on_close)

            # Send text 
            start_time = time.time()
            dg_connection.send_text(SpeakV1TextMessage(type = "Speak", text=text))
            
            # Flush
            dg_connection.send_control(SpeakV1ControlMessage(type="Flush"))
    
            dg_connection.start_listening()

            # Brief moment for cleanup
            await asyncio.sleep(0.1)
                    
        # Save audio file
        filename = None
        if audio_chunks:
            filename = f"deepgram_{self.model}_{int(time.time())}.wav"
            audio_data = b''.join(audio_chunks)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(audio_data)
        
        return ttfa, filename