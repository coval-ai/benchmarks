import asyncio
import time
import concurrent.futures
import traceback
import logging
from .base import STTProvider, TranscriptionResult

# Check if Google Cloud Speech is available
try:
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class GoogleProvider(STTProvider):
    def __init__(self, api_key: str = "service_account", model: str = "default"):
        super().__init__(api_key, model)

        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "google-cloud-speech package is required for GoogleProvider"
            )

        # Validate model for v2 API
        valid_models = ["default", "short", "long", "telephony", "chirp_2"]
        if model not in valid_models:
            raise ValueError(
                f"Invalid Google model: {model}. Valid models: {valid_models}"
            )

        # Initialize Google Speech v2 client with us-central1 endpoint
        self.client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint="us-central1-speech.googleapis.com",
            )
        )

    def _get_model_name(self) -> str:
        """Get the Google v2 model name for API calls."""
        if self.model == "default":
            return "chirp_2"
        # For v2 API, model names are used directly
        return self.model

    def _get_recognizer_name(self) -> str:
        """Build the recognizer resource name for v2 API."""
        return f"projects/stt-benchmarking-464409/locations/us-central1/recognizers/_"

    async def measure_ttft(
        self,
        audio_data: bytes,
        channels: int,
        sample_width: int,
        sample_rate: int,
        realtime_resolution: float = 0.1,
        audio_duration: float = None,
    ) -> TranscriptionResult:
        """Google v2 API-specific implementation with TTFT using first transcript content method."""
        from wer_calculator import compare_transcription

        result = TranscriptionResult(provider=self.name)
        total_start_time = time.time()

        try:
            # Configure the streaming recognition request for v2 API
            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=channels,
                ),
                language_codes=["en-US"],
                model=self._get_model_name(),
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=True,
                ),
            )

            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=True,
                ),
            )

            # Run Google's v2 streaming recognition in executor to handle sync API
            def run_streaming_recognition():
                final_transcripts = []
                last_final_result_time = None

                try:
                    # Create all requests upfront to avoid generator issues
                    requests = []

                    # First request with config
                    config_request = cloud_speech.StreamingRecognizeRequest(
                        recognizer=self._get_recognizer_name(),
                        streaming_config=streaming_config,
                    )
                    requests.append(config_request)

                    # Prepare audio chunks
                    data_copy = audio_data
                    byte_rate = sample_width * sample_rate * channels
                    first_chunk = True

                    while len(data_copy):
                        chunk_size = int(byte_rate * realtime_resolution)
                        chunk, data_copy = (
                            data_copy[:chunk_size],
                            data_copy[chunk_size:],
                        )

                        # Record start time when preparing first audio chunk
                        if first_chunk and chunk:
                            result.audio_start_time = time.time()
                            first_chunk = False

                        # Create audio request
                        audio_request = cloud_speech.StreamingRecognizeRequest(
                            audio=chunk
                        )
                        requests.append(audio_request)

                    # Use iterator over the pre-built requests list
                    def request_iterator():
                        for req in requests:
                            yield req
                            if (
                                req.audio
                            ):  # If this is an audio chunk, simulate real-time
                                time.sleep(realtime_resolution)

                    # Call streaming_recognize with the iterator
                    responses = self.client.streaming_recognize(
                        requests=request_iterator()
                    )

                    # Process responses with TTFT timing
                    for response in responses:
                        current_time = time.time()

                        transcript = self._extract_transcript(response)
                        if transcript:
                            # TTFT: Record first actual transcript content timing
                            if (
                                result.ttft_seconds is None
                                and result.audio_start_time is not None
                            ):
                                result.ttft_seconds = (
                                    current_time - result.audio_start_time
                                )
                                result.first_token_content = transcript

                            # Track partial transcripts
                            result.partial_transcripts.append(transcript)

                        # For complete transcript, only collect final results (is_final=True)
                        if hasattr(response, "results") and response.results:
                            for result_item in response.results:
                                if getattr(result_item, "is_final", False):
                                    final_transcripts.append(transcript)
                                    last_final_result_time = current_time
                                    break

                    # Calculate audio-to-final timing
                    if last_final_result_time and result.audio_start_time:
                        result.audio_to_final_seconds = (
                            last_final_result_time - result.audio_start_time
                        )

                    # Build complete transcript from final results only
                    if final_transcripts:
                        result.complete_transcript = " ".join(final_transcripts).strip()
                        result.transcript_length = len(result.complete_transcript)
                        result.word_count = (
                            len(result.complete_transcript.split())
                            if result.complete_transcript
                            else 0
                        )

                except Exception as e:
                    print(f"Inner streaming error: {e}")
                    raise

            # Run in thread pool to integrate with async code
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, run_streaming_recognition)

        except Exception as e:
            result.error = str(e)
            print(f"Error with {self.name}: {e}")
            traceback.print_exc()

        return result

    def _extract_transcript(self, response) -> str:
        """Extract transcript from Google Speech v2 response."""
        try:
            if not response.results:
                return ""

            result = response.results[0]
            if not result.alternatives:
                return ""

            return result.alternatives[0].transcript

        except (AttributeError, IndexError):
            return ""
