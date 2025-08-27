import asyncio
import time
import wave
import pandas as pd
import logging
import os
import jiwer
from jiwer import transforms
import io
import librosa
import soundfile as sf
from datasets import load_dataset
import re
import csv
from datetime import datetime
from sqlalchemy import create_engine
from typing import List
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretmanager import get_secret, get_api_key, get_google_credentials

from providers import (
    DeepgramProvider, 
    AssemblyAIProvider, 
    SpeechmaticsProvider,
    GoogleProvider,
    GOOGLE_AVAILABLE,
    TranscriptionResult
)

from wer_calculator import compare_transcription

print("Loading API keys from AWS Secrets Manager...")
secrets = get_secret("prod/benchmarking")

hf_token = get_api_key('HUGGING_FACE_TOKEN', secrets)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
else:
    print("HUGGING_FACE_TOKEN not found.")

WER_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.RemovePunctuation(),
    transforms.ToLowerCase(),
    transforms.RemoveMultipleSpaces(),
    transforms.Strip(),
    transforms.ExpandCommonEnglishContractions(),
    transforms.RemoveEmptyStrings(),
    transforms.ReduceToListOfListOfWords()
])

GROUND_TRUTH = "For orders over £500, shipping is free when you use promo code SHIP123 or call our order desk at 02079460371."

google_creds_path = get_google_credentials(secrets)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path

def create_providers():
    providers = []
    
    try:
        deepgram_key = get_api_key('DEEPGRAM_API_KEY', secrets)
        if deepgram_key:
            providers.extend([
                DeepgramProvider(api_key=deepgram_key, model="nova-2"),
                DeepgramProvider(api_key=deepgram_key, model="nova-3"),
            ])
        else:
            print("DEEPGRAM_API_KEY not found - skipping Deepgram providers")
    except Exception as e:
        print(f"Error creating Deepgram providers: {e}")
    
    try:
        assemblyai_key = get_api_key('ASSEMBLYAI_API_KEY', secrets)
        if assemblyai_key:
            providers.append(
                AssemblyAIProvider(api_key=assemblyai_key, model="universal-streaming")
            )
        else:
            print("ASSEMBLYAI_API_KEY not found - skipping AssemblyAI providers")
    except Exception as e:
        print(f"Error creating AssemblyAI providers: {e}")
    
    try:
        speechmatics_key = get_api_key('SPEECHMATICS_API_KEY', secrets)
        if speechmatics_key:
            providers.extend([
                SpeechmaticsProvider(api_key=speechmatics_key, model="default"),
                SpeechmaticsProvider(api_key=speechmatics_key, model="enhanced"),
            ])
        else:
            print("SPEECHMATICS_API_KEY not found - skipping Speechmatics providers")
    except Exception as e:
        print(f"Error creating Speechmatics providers: {e}")
    
    if GOOGLE_AVAILABLE:
        try:
            providers.extend([
                GoogleProvider(model="short"),
                GoogleProvider(model="long"), 
                GoogleProvider(model="telephony"),
                GoogleProvider(model="chirp_2"),
            ])
        except Exception as e:
            print(f"Error creating Google providers: {e}")
    else:
        print("Google Cloud Speech not available - skipping Google providers")
    
    return providers

def process_transcription_result(result: TranscriptionResult, ground_truth: str, audio_duration: float) -> TranscriptionResult:
    
    if result.complete_transcript:
        try:
            wer_analysis = compare_transcription(ground_truth, result.complete_transcript)
            custom_wer = wer_analysis['wer']
            
            if custom_wer is not None:
                result.wer_percentage = custom_wer * 100
                
                if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                    logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
            else:
                print(f"Error calculating WER for {result.provider}")
        except Exception as e:
            print(f"Error calculating WER for {result.provider}: {e}")
            result.wer_percentage = None
    
    if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
        result.rtf_value = audio_duration / result.audio_to_final_seconds
    
    if result.complete_transcript:
        wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
        audio_to_final_text = f" (Audio→Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
        rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
        
        print(f"\n{result.provider.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
        print(f"   {result.complete_transcript}")
    else:
        print(f"\n{result.provider.upper()} TRANSCRIPT: [EMPTY]")
    
    return result

def load_common_voice_sample(min_duration: float = 2.0, max_duration: float = 15.0, max_retries: int = 10) -> tuple[bytes, int, int, int, str, float, str]:
    print(f"Loading Common Voice dataset (English test split)...")
    
    # Load dataset in streaming mode (no download)
    try:
        dataset = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="test", streaming=True)
        print("Dataset connection established")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise
    
    # Try to find a suitable sample
    for attempt in range(max_retries):
        try:
            # Get a random sample
            sample = next(iter(dataset.shuffle(seed=None).take(1)))
            
            # Get audio data and info
            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            transcript = sample['sentence']
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            # Check if duration is in our range
            if min_duration <= duration <= max_duration:
                # Convert to WAV format with consistent 16kHz sample rate
                audio_data = convert_to_wav_bytes(audio_array, sample_rate, target_sample_rate=16000)
                
                # Clean transcript for ground truth
                ground_truth = clean_transcript(transcript)
                
                # Generate filename
                filename = f"common_voice_sample_{attempt+1}.wav"
                
                # Return in same format as original load_wav_file - force 16kHz sample rate
                return audio_data, 1, 2, 16000, filename, duration, ground_truth
            else:
                print(f"Sample {attempt+1}: {duration:.2f}s (outside {min_duration}-{max_duration}s range)")
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            continue
    
    raise Exception(f"Failed to find suitable sample after {max_retries} attempts")

def convert_to_wav_bytes(audio_array, original_sample_rate: int, target_sample_rate: int = 16000) -> bytes:
    print(f"Audio conversion: {original_sample_rate}Hz → {target_sample_rate}Hz")
    
    if original_sample_rate != target_sample_rate:
        print(f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz...")
        audio_array = librosa.resample(audio_array, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    else:
        print(f"No resampling needed (already {target_sample_rate}Hz)")
    
    # Convert to bytes using soundfile
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_array, target_sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        wav_bytes = wav_buffer.read()
        pcm_data = wav_bytes[44:]  # Skip WAV header
        
        print(f"Converted to {len(pcm_data)} bytes of PCM data")
        return pcm_data

def clean_transcript(transcript: str) -> str:
    transcript = transcript.lower()
    
    transcript = ' '.join(transcript.split())
    
    
    return transcript.strip()

def save_results_to_csv(results: List[TranscriptionResult], timestamp: str, audio_filename: str = "test.wav"):
    filename = "all_benchmarks.csv"
    
    headers = [
        'provider',
        'model', 
        'voice',
        'benchmark',
        'metric_type',
        'metric_value',
        'metric_units',
        'audio_filename',
        'timestamp',
        'status'
    ]
    
    rows = []
    readable_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for result in results:
        if isinstance(result, Exception):
            continue  # Skip exception results
        
        if '-' in result.provider:
            provider, model = result.provider.split('-', 1)
        else:
            provider = result.provider
            model = "default"
        
        status = "failed" if result.error else "success"
        
        base_row = [
            provider,
            model,
            "N/A",  # voice
            "STT",  # benchmark
            None,   # metric_type (to be filled)
            None,   # metric_value (to be filled)
            "s",    # metric_units (default for timing)
            audio_filename,
            readable_timestamp,
            status
        ]
        
        if result.ttft_seconds is not None:
            ttft_row = base_row.copy()
            ttft_row[4] = "TTFT"
            ttft_row[5] = result.ttft_seconds
            rows.append(ttft_row)
        
        if result.wer_percentage is not None:
            wer_row = base_row.copy()
            wer_row[4] = "WER"
            wer_row[5] = result.wer_percentage
            wer_row[6] = "%"  # percentage units
            rows.append(wer_row)
        
        if result.rtf_value is not None:
            rtf_row = base_row.copy()
            rtf_row[4] = "RTF"
            rtf_row[5] = result.rtf_value
            rtf_row[6] = None  # NULL units
            rows.append(rtf_row)
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"\nResults saved to: {filename}")
        print(f"Contains {len(rows)} metric rows (TTFT, WER & RTF) from {len([r for r in results if not isinstance(r, Exception)])} providers")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")

def load_wav_file(file_path: str) -> tuple[bytes, int, int, int, str, float]:
    import os
    filename = os.path.basename(file_path)
    
    with wave.open(file_path, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        audio_data = wav_file.readframes(frames)
        
        duration_seconds = frames / sample_rate
        
        print(f"Loaded audio: {channels} channels, {sample_rate}Hz, {sample_width} bytes/sample, {len(audio_data)} bytes, {duration_seconds:.2f}s duration")
        return audio_data, channels, sample_width, sample_rate, filename, duration_seconds

def display_results_table(results: List[TranscriptionResult]):
    print("\n" + "="*140)
    print("STT MODEL BENCHMARK RESULTS")
    print("="*140)
    print(f"{'Provider/Model':<20} {'TTFT (s)':<8} {'WER (%)':<8} {'Total (s)':<9} {'VAD (s)':<8} {'First Token':<30} {'Error':<15}")
    print("-"*140)
    
    successful_ttft = []
    successful_wer = []
    failed_results = []
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception: {result}")
            continue
        
        provider_name = result.provider
        
        ttft = f"{result.ttft_seconds:.3f}" if result.ttft_seconds else "N/A"
        wer = f"{result.wer_percentage:.1f}" if result.wer_percentage is not None else "N/A"
        total = f"{result.total_time:.3f}" if result.total_time else "N/A"
        vad = f"{result.vad_first_detected:.3f}" if result.vad_first_detected else "N/A"
        
        first_token = (result.first_token_content[:27] + "...") if result.first_token_content and len(result.first_token_content) > 30 else (result.first_token_content or "")
        error = result.error[:12] + "..." if result.error and len(result.error) > 15 else (result.error or "")
        
        print(f"{provider_name:<20} {ttft:<8} {wer:<8} {total:<9} {vad:<8} {first_token:<30} {error:<15}")
        
        if not result.error:
            if result.ttft_seconds:
                successful_ttft.append((provider_name, result.ttft_seconds))
            if result.wer_percentage is not None:
                successful_wer.append((provider_name, result.wer_percentage))
        else:
            failed_results.append(provider_name)
    
    return successful_ttft, successful_wer, failed_results

def display_analysis(successful_ttft: List[tuple], successful_wer: List[tuple], failed_results: List[str]):
    if successful_ttft:
        successful_ttft.sort(key=lambda x: x[1])
        fastest_ttft = successful_ttft[0]
        print(f"\nFastest TTFT (Time to First Token): {fastest_ttft[0]} ({fastest_ttft[1]:.3f}s)")
        
        if len(successful_ttft) > 1:
            avg_ttft = sum(result[1] for result in successful_ttft) / len(successful_ttft)
            print(f"Average TTFT: {avg_ttft:.3f}s")
            
            print("\nTTFT Rankings (Time to First Token):")
            for i, (name, time_val) in enumerate(successful_ttft[:5], 1):
                print(f"   {i}. {name}: {time_val:.3f}s")
    
    if successful_wer:
        successful_wer.sort(key=lambda x: x[1])  # Sort by WER (lower is better)
        best_wer = successful_wer[0]
        print(f"\nBest WER (Accuracy): {best_wer[0]} ({best_wer[1]:.1f}%)")
        
        if len(successful_wer) > 1:
            avg_wer = sum(result[1] for result in successful_wer) / len(successful_wer)
            print(f"Average WER: {avg_wer:.1f}%")
            
            print("\n WER Rankings (Accuracy - Lower is Better):")
            for i, (name, wer_val) in enumerate(successful_wer[:5], 1):
                print(f"   {i}. {name}: {wer_val:.1f}%")
    
    print("\nModel Comparisons within Providers:")
    
    deepgram_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('deepgram')]
    deepgram_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('deepgram')]
    
    if deepgram_ttft:
        deepgram_ttft.sort(key=lambda x: x[1])
        print(f"Deepgram TTFT fastest: {deepgram_ttft[0][0]} ({deepgram_ttft[0][1]:.3f}s)")
    if deepgram_wer:
        deepgram_wer.sort(key=lambda x: x[1])
        print(f"Deepgram WER best: {deepgram_wer[0][0]} ({deepgram_wer[0][1]:.1f}%)")
    
    google_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('google')]
    google_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('google')]
    
    if google_ttft:
        google_ttft.sort(key=lambda x: x[1])
        print(f"Google TTFT fastest: {google_ttft[0][0]} ({google_ttft[0][1]:.3f}s)")
    if google_wer:
        google_wer.sort(key=lambda x: x[1])
        print(f"Google WER best: {google_wer[0][0]} ({google_wer[0][1]:.1f}%)")
    
    speechmatics_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('speechmatics')]
    speechmatics_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('speechmatics')]
    
    if speechmatics_ttft:
        speechmatics_ttft.sort(key=lambda x: x[1])
        print(f"Speechmatics TTFT fastest: {speechmatics_ttft[0][0]} ({speechmatics_ttft[0][1]:.3f}s)")
    if speechmatics_wer:
        speechmatics_wer.sort(key=lambda x: x[1])
        print(f"Speechmatics WER best: {speechmatics_wer[0][0]} ({speechmatics_wer[0][1]:.1f}%)")
    
    if failed_results:
        print(f"\n Failed: {', '.join(failed_results)}")

async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    providers = create_providers()
    
    if not providers:
        print("No providers could be created. Check your API keys and dependencies.")
        return
    
    try:
        audio_data, channels, sample_width, sample_rate, audio_filename, audio_duration, ground_truth = load_common_voice_sample()
        
        global GROUND_TRUTH
        GROUND_TRUTH = ground_truth
        
    except Exception as e:
        print(f"Failed to load Common Voice sample: {e}")
        print("Falling back to local test.wav file")
        audio_data, channels, sample_width, sample_rate, audio_filename, audio_duration = load_wav_file("test.wav")
        ground_truth = GROUND_TRUTH  # Use original hardcoded value
    
    print(f"Running benchmarks for {len(providers)} provider/model combinations...")
    print("\nModels being tested:")
    for provider in providers:
        print(f"  - {provider.name}")
    
    print(f"\nGround Truth: \"{ground_truth}\"")
    print(f"Audio Length: {audio_duration:.2f}s")
    print(f"Ground Truth Length: {len(ground_truth.split())} words, {len(ground_truth)} characters")
    print(f"Audio Source: {audio_filename}")
    
    tasks = [
        provider.measure_ttft(audio_data, channels, sample_width, sample_rate, 0.1, audio_duration)
        for provider in providers
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(result)
        else:
            processed_results.append(process_transcription_result(result, ground_truth, audio_duration))
    
    successful_ttft, successful_wer, failed_results = display_results_table(processed_results)
    display_analysis(successful_ttft, successful_wer, failed_results)
    
    print(f"\nCompleted benchmarks for {len(providers)} provider/model combinations")
    
    save_results_to_csv(processed_results, timestamp, audio_filename)

    try:
        df = pd.read_csv("all_benchmarks.csv")
        engine = create_engine(get_api_key("DATABASE_URL", secrets))
        df.to_sql('all_benchmarks', engine, if_exists='append', index=False)
        print("Data uploaded to database.")
    except Exception as e:
        logging.error(f"Error writing results to database: {e}")
    
    print(f"\nSummary:")
    print(f"Total providers tested: {len(providers)}")
    print(f"Successful TTFT measurements: {len(successful_ttft)}")
    print(f"Successful WER measurements: {len(successful_wer)}")
    print(f"Failed providers: {len(failed_results)}")
    print(f"Sample source: Common Voice EN test set")
    
    if successful_ttft:
        ttft_winner = min(successful_ttft, key=lambda x: x[1])
        print(f"\nTTFT winner: {ttft_winner[0]} ({ttft_winner[1]:.3f}s)")
    
    if successful_wer:
        wer_winner = min(successful_wer, key=lambda x: x[1])
        print(f"Accuracy winner: {wer_winner[0]} ({wer_winner[1]:.1f}% WER)")

if __name__ == "__main__":
    asyncio.run(main())