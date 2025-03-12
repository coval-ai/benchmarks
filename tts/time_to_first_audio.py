  async def _run_cartesia_tts(self, input_str: str, recording_sid: str):
        assert os.environ.get("CARTESIA_API_KEY") is not None, "CARTESIA_API_KEY is not set"
        cartesia = AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

        voice_id = "bf0a246a-8642-498a-9950-80c35e9276b5" # Sophie
        output_format = {"sample_rate": 44100, "container": "raw", "encoding": "pcm_f32le"}
        
        ws = await cartesia.tts.websocket()
        await ws.connect()
        
        ttfas = []
        for _ in range(5):
            gen = await ws.send(
                model_id=self.model,
                language="en",
                voice_id=voice_id,
                output_format=output_format,
                transcript=input_str,
            )
            start_time = time.time()
            ttfa = None
            i = 0
            async for chunk in gen:
                if i == 0:
                    ttfa = time.time() - start_time
                    ttfas.append(ttfa * 1000)
                i += 1
        
        ttfa = np.median(ttfas)
        try:
            s3_key = stream_to_s3(
                stream=gen,
                recording_sid=recording_sid
            )
        except Exception as e:
            s3_key = None
            print(f"ERROR STREAMING TO S3: {e}")
        
        await ws.close()
        return ttfa, s3_key

    async def _run_elevenlabs_tts(self, input_str: str, recording_sid: str):
        n_trials = 5
        ttfas = []
        audio_chunks = []  # Store audio chunks for the final trial
        
        for trial in range(n_trials):
            uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model}"
            ws = await websockets.connect(uri)
            
            # Send dummy request to warm up.
            # This gives elevenlabs the best TTFA.
            key = self.elevenlabs_api_key
            await ws.send(json.dumps({"text": " ", "xi_api_key": key}))

            # Start benchmarking
            start = time.perf_counter()
            first, ttfa = False, None
            await ws.send(json.dumps({"text": input_str + " "}))
            await ws.send(json.dumps({"text": ""}))
            
            # Clear audio chunks for each new trial
            if trial == n_trials - 1:  # Only collect audio in the last trial
                audio_chunks = []
            while True:
                message = await ws.recv()
                if not first:
                    first = True
                    ttfa = (time.perf_counter() - start) * 1000
                    ttfas.append(ttfa)
                
                message_json = json.loads(message)
                
                # Collect audio data in the last trial
                if trial == n_trials - 1 and "audio" in message_json:
                    # ElevenLabs sends base64 encoded audio
                    if ("audio" in message_json and message_json["audio"]):
                        try:
                            audio_data = base64.b64decode(message_json["audio"])
                            audio_chunks.append(audio_data)
                        except Exception as e:
                            print(f"Error decoding audio: {e}")
                
                if "isFinal" in message_json and message_json["isFinal"]:
                    break

            await ws.close()
        
        ttfa = np.median(ttfas)
        print(f"TTFA (median): {ttfa:0.2f} ms")
        
        print(f"Audio chunks: {len(audio_chunks)}")
        def audio_generator():
            for chunk in audio_chunks:
                yield chunk
        
        s3_key = ""
        if len(audio_chunks) > 0:
            try:
                s3_key = stream_to_s3(
                    stream=audio_generator(),
                    recording_sid=recording_sid
                )
            except Exception as e:
                s3_key = None
                print(f"Error streaming to S3: {e}")
        
        return ttfa, s3_key
