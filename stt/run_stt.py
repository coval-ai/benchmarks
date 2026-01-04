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
from typing import List, Optional, Tuple
import sys
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import secrets management
from secretmanager import get_secret, get_api_key, get_google_credentials

# Import providers
from providers import (
    DeepgramProvider, 
    ElevenLabsProvider,
    AssemblyAIProvider, 
    SpeechmaticsProvider,
    GoogleProvider,
    GOOGLE_AVAILABLE,
    TranscriptionResult
)

# Import WER calculator
from wer_calculator import compare_transcription

# Load secrets at startup
print("Loading API keys from AWS Secrets Manager...")
secrets = get_secret("prod/benchmarking")

hf_token = get_api_key('HUGGING_FACE_TOKEN', secrets)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
else:
    print("HUGGING_FACE_TOKEN not found.")

# Create WER normalization transforms - combine into a single callable
WER_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.RemovePunctuation(),
    transforms.ToLowerCase(),
    transforms.RemoveMultipleSpaces(),
    transforms.Strip(),
    transforms.ExpandCommonEnglishContractions(),
    transforms.RemoveEmptyStrings(),
    transforms.ReduceToListOfListOfWords()
])

# Ground truth for WER calculation (will be updated dynamically)
GROUND_TRUTH = "For orders over £500, shipping is free when you use promo code SHIP123 or call our order desk at 02079460371."

# Set Google credentials
google_creds_path = get_google_credentials(secrets)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path

def create_providers():
    """Create and return list of configured providers."""
    providers = []
    
    try:
        # Deepgram models (only nova variants)
        deepgram_key = get_api_key('DEEPGRAM_API_KEY', secrets)
        if deepgram_key:
            providers.extend([
                DeepgramProvider(api_key=deepgram_key, model="nova-2"),
                DeepgramProvider(api_key=deepgram_key, model="nova-3"),
                DeepgramProvider(api_key=deepgram_key, model="flux-general-en"),
            ])
        else:
            print("DEEPGRAM_API_KEY not found - skipping Deepgram providers")
    except Exception as e:
        print(f"Error creating Deepgram providers: {e}")

    # try:
    #     # ElevenLabs models
    #     elevenlabs_key = get_api_key('ELEVENLABS_API_KEY', secrets)
    #     if elevenlabs_key:
    #         providers.append(
    #             ElevenLabsProvider(api_key=elevenlabs_key, model="scribe_v2_realtime")
    #         )
    #     else:
    #         print("ASSEMBLYAI_API_KEY not found - skipping AssemblyAI providers")
    # except Exception as e:
    #     print(f"Error creating Deepgram providers: {e}")
    
    try:
        # AssemblyAI models (v3 streaming API)
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
        # Speechmatics models (reduced to avoid quota issues)
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
    
    # Add Google providers only if the library is available
    # if GOOGLE_AVAILABLE:
    #     try:
    #         providers.extend([
    #             GoogleProvider(model="short"),
    #             GoogleProvider(model="long"), 
    #             GoogleProvider(model="telephony"),
    #             GoogleProvider(model="chirp_2"),
    #         ])
    #     except Exception as e:
    #         print(f"Error creating Google providers: {e}")
    # else:
    #     print("Google Cloud Speech not available - skipping Google providers")
    
    return providers

def process_transcription_result(result: TranscriptionResult, ground_truth: str, audio_duration: float) -> TranscriptionResult:
    """Process transcription result to calculate WER and RTF metrics."""
    
    # Calculate WER if transcript is available
    if result.complete_transcript:
        try:
            wer_analysis = compare_transcription(ground_truth, result.complete_transcript)
            custom_wer = wer_analysis['wer']
            
            if custom_wer is not None:
                result.wer_percentage = custom_wer * 100
                
                # Log detailed error analysis for debugging when there are differences
                if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                    logging.info(f"Custom WER errors: {wer_analysis['incorrect_words']}")
            else:
                print(f"Error calculating WER for {result.provider}")
        except Exception as e:
            print(f"Error calculating WER for {result.provider}: {e}")
            result.wer_percentage = None
    
    # Calculate RTF if audio-to-final timing is available
    if result.audio_to_final_seconds is not None and audio_duration is not None and audio_duration > 0:
        result.rtf_value = audio_duration / result.audio_to_final_seconds
    
    # Print transcript to console with WER, Audio-to-Final timing, and RTF
    if result.complete_transcript:
        wer_text = f" (WER: {result.wer_percentage:.1f}%)" if result.wer_percentage is not None else ""
        audio_to_final_text = f" (Audio -> Final: {result.audio_to_final_seconds:.2f}s)" if result.audio_to_final_seconds is not None else ""
        rtf_text = f" (RTF: {result.rtf_value:.2f}x)" if result.rtf_value is not None else ""
        
        print(f"\n{result.provider.upper()} TRANSCRIPT{wer_text}{audio_to_final_text}{rtf_text}:")
        print(f"   {result.complete_transcript}")
    else:
        print(f"\n{result.provider.upper()} TRANSCRIPT: [EMPTY]")
    
    return result

# LOCAL DATA LOADING FUNCTIONS (NEW)
def load_local_sample(min_duration: float = 2.0, max_duration: float = 15.0, max_retries: int = 10) -> Tuple[bytes, int, int, int, str, float, str]:
    """
    Load a random sample from local en_test folder and test.tsv file.
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds, ground_truth)
    """
    print("Loading from local en_test folder and test.tsv...")
    
    # Check if local files exist
    tsv_path = "test.tsv"
    audio_dir = "en_test"
    
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"test.tsv not found at {tsv_path}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"en_test directory not found at {audio_dir}")
    
    # Load TSV file
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(df)} entries from test.tsv")
    except Exception as e:
        raise Exception(f"Failed to load test.tsv: {e}")
    
    # Required columns
    required_columns = ['path', 'sentence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in test.tsv: {missing_columns}")
    
    # Try to find a suitable sample
    for attempt in range(max_retries):
        try:
            # Select a random row
            sample_row = df.sample(n=1).iloc[0]
            
            # Get audio file path and sentence
            audio_filename = sample_row['path']
            audio_path = os.path.join(audio_dir, audio_filename)
            sentence = sample_row['sentence']
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"Attempt {attempt+1}: Audio file not found: {audio_path}")
                continue
            
            # Load and check audio duration
            try:
                audio_data, channels, sample_width, sample_rate, _, duration = load_audio_file(audio_path)
                
                # Check if duration is in our range
                if min_duration <= duration <= max_duration:
                    # Clean transcript for ground truth
                    ground_truth = clean_transcript(sentence)
                    
                    print(f"Selected local sample: {audio_filename} ({duration:.2f}s)")
                    return audio_data, channels, sample_width, sample_rate, audio_filename, duration, ground_truth
                else:
                    print(f"Sample {attempt+1}: {audio_filename} - {duration:.2f}s (outside {min_duration}-{max_duration}s range)")
            
            except Exception as e:
                print(f"Attempt {attempt+1}: Failed to load {audio_filename}: {e}")
                continue
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            continue
    
    raise Exception(f"Failed to find suitable local sample after {max_retries} attempts")

# COMMON VOICE FUNCTIONS (EXISTING)
def load_common_voice_sample(min_duration: float = 2.0, max_duration: float = 15.0, max_retries: int = 10) -> Tuple[bytes, int, int, int, str, float, str]:
    """
    Load a random sample from Common Voice English test set.
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds, ground_truth)
    """
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

def load_audio_sample(min_duration: float = 2.0, max_duration: float = 15.0, max_retries: int = 10) -> Tuple[bytes, int, int, int, str, float, str]:
    """
    Load audio sample with fallback strategy:
    1. Try local en_test folder first
    2. Fall back to Common Voice API
    3. Fall back to hardcoded test.wav
    
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds, ground_truth)
    """
    print("Loading audio sample with fallback strategy...")
    
    # First try: Local files
    try:
        print("Attempting to load from local en_test folder...")
        return load_local_sample(min_duration, max_duration, max_retries)
    except Exception as e:
        print(f"Local loading failed: {e}")
    
    # Second try: Common Voice API
    try:
        print("Attempting to load from Common Voice API...")
        return load_common_voice_sample(min_duration, max_duration, max_retries)
    except Exception as e:
        print(f"Common Voice API failed: {e}")
    
    # Final fallback: test.wav with hardcoded ground truth
    try:
        print("Falling back to local test.wav file...")
        if os.path.exists("test.wav"):
            audio_data, channels, sample_width, sample_rate, filename, duration = load_audio_file("test.wav")
            ground_truth = GROUND_TRUTH  # Use hardcoded value
            print(f"Loaded fallback test.wav ({duration:.2f}s)")
            return audio_data, channels, sample_width, sample_rate, filename, duration, ground_truth
        else:
            raise FileNotFoundError("test.wav not found")
    except Exception as e:
        raise Exception(f"All audio loading methods failed. Final error: {e}")

def convert_to_wav_bytes(audio_array, original_sample_rate: int, target_sample_rate: int = 16000) -> bytes:
    """
    Convert audio array to WAV bytes format expected by STT providers.
    Always resamples to target_sample_rate (default 16kHz) for consistency.
    """
    print(f"Audio conversion: {original_sample_rate}Hz â†’ {target_sample_rate}Hz")
    
    # Always resample to ensure consistent sample rate across all providers
    if original_sample_rate != target_sample_rate:
        print(f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz...")
        audio_array = librosa.resample(audio_array, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    else:
        print(f"No resampling needed (already {target_sample_rate}Hz)")
    
    # Convert to bytes using soundfile
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_array, target_sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        # Skip WAV header (44 bytes) to get raw PCM data
        wav_bytes = wav_buffer.read()
        pcm_data = wav_bytes[44:]  # Skip WAV header
        
        print(f"Converted to {len(pcm_data)} bytes of PCM data")
        return pcm_data

def clean_transcript(transcript: str) -> str:
    """
    Clean and normalize transcript text for WER calculation.
    """
    # Convert to lowercase
    transcript = transcript.lower()
    
    # Remove extra whitespace
    transcript = ' '.join(transcript.split())
    
    # REMOVED: All regex operations to eliminate errors
    # Basic string cleaning only
    
    return transcript.strip()

# Utility Functions
def save_results_to_csv(results: List[TranscriptionResult], timestamp: str, audio_filename: str = "test.wav", test_type: str = "unknown"):
    """Save benchmark results to CSV file in long format with TTFT, WER, RTF metrics and transcript column."""
    filename = "all_benchmarks.csv"
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)
    
    # Prepare CSV headers for long format (added transcript column)
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
        'status',
        'transcript'  # New column for first transcript (only for TTFT rows)
    ]
    
    # Prepare data rows in long format
    rows = []
    readable_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine if this is local test.wav (for NTTFT vs TTFT)
    is_local_test = audio_filename == "test.wav"
    
    for result in results:
        if isinstance(result, Exception):
            continue  # Skip exception results
        
        # Parse provider and model from provider name
        if '-' in result.provider:
            provider, model = result.provider.split('-', 1)
        else:
            provider = result.provider
            model = "default"
        
        # Determine status
        status = "failed" if result.error else "success"
        
        # Get first transcript content (for TTFT rows only)
        first_transcript = getattr(result, 'first_token_content', None) or ""
        
        # Base row template
        base_row = [
            provider,
            model,
            "N/A",  # voice
            "STT",  # benchmark
            None,   # metric_type (to be filled)
            None,   # metric_value (to be filled)
            None,   # metric_units (to be filled)
            audio_filename,
            readable_timestamp,
            status,
            ""      # transcript (only filled for TTFT rows)
        ]
        
        # Add TTFT metric (NTTFT for local test.wav, TTFT for Common Voice)
        if result.ttft_seconds is not None:
            ttft_row = base_row.copy()
            ttft_row[4] = "NTTFT" if is_local_test else "TTFT"
            ttft_row[5] = result.ttft_seconds
            ttft_row[6] = "s"  # seconds units
            ttft_row[10] = first_transcript  # Add first transcript to transcript column
            rows.append(ttft_row)
        
        # Add WER metric (no transcript)
        if result.wer_percentage is not None:
            wer_row = base_row.copy()
            wer_row[4] = "WER"
            wer_row[5] = result.wer_percentage
            wer_row[6] = "%"  # percentage units
            # transcript column remains empty for WER rows
            rows.append(wer_row)
        
        # Add RTF metric (no transcript)
        if result.rtf_value is not None:
            rtf_row = base_row.copy()
            rtf_row[4] = "RTF"
            rtf_row[5] = result.rtf_value
            rtf_row[6] = None  # NULL units
            # transcript column remains empty for RTF rows
            rows.append(rtf_row)
        
        # NEW: Audio-to-Final metric (raw value from audio start)
        if result.audio_to_final_seconds is not None:
            audio_final_row = base_row.copy()
            audio_final_row[4] = "AudioToFinal"
            audio_final_row[5] = result.audio_to_final_seconds
            audio_final_row[6] = "s"
            rows.append(audio_final_row)
        
    # Write to CSV (append mode to combine both tests in same file)
    try:
        with open(filename, 'a' if file_exists else 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:  # Only write headers if file doesn't exist
                writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"\n{'Appended to' if file_exists else 'Created'}: {filename}")
        print(f"Added {len(rows)} metric rows ({test_type}) from {len([r for r in results if not isinstance(r, Exception)])} providers")
        
    except Exception as e:
        print(f"Error saving CSV: {e}")

def load_audio_file(file_path: str, target_sample_rate: int = 16000) -> Tuple[bytes, int, int, int, str, float]:
    """
    Load audio file (WAV, MP3, etc.) and return audio data with parameters, filename, and duration.
    Automatically converts to target sample rate and format expected by STT providers.
    Returns: (audio_data, channels, sample_width, sample_rate, filename, duration_seconds)
    """
    import os
    filename = os.path.basename(file_path)
    
    try:
        # Use librosa to load any audio format (MP3, WAV, etc.)
        audio_array, original_sample_rate = librosa.load(file_path, sr=None, mono=True)
        
        # Calculate original duration
        duration_seconds = len(audio_array) / original_sample_rate
        
        print(f"Loaded {filename}: {original_sample_rate}Hz, {len(audio_array)} samples, {duration_seconds:.2f}s duration")
        
        # Convert to target sample rate if needed
        if original_sample_rate != target_sample_rate:
            print(f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz...")
            audio_array = librosa.resample(audio_array, orig_sr=original_sample_rate, target_sr=target_sample_rate)
        
        # Convert to PCM bytes format expected by STT providers
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_array, target_sample_rate, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            
            # Skip WAV header (44 bytes) to get raw PCM data
            wav_bytes = wav_buffer.read()
            pcm_data = wav_bytes[44:]  # Skip WAV header
            
            # Return format consistent with original function
            # (audio_data, channels, sample_width, sample_rate, filename, duration_seconds)
            return pcm_data, 1, 2, target_sample_rate, filename, duration_seconds
            
    except Exception as e:
        raise Exception(f"Failed to load audio file {filename}: {e}")

def load_wav_file(file_path: str) -> Tuple[bytes, int, int, int, str, float]:
    """
    Legacy function for loading WAV files. Now redirects to load_audio_file for compatibility.
    """
    return load_audio_file(file_path)

def display_results_table(results: List[TranscriptionResult]):
    print("\n" + "="*150)
    print("STT MODEL BENCHMARK RESULTS")
    print("="*150)
    print(f"{'Provider/Model':<25} {'TTFT (s)':<10} {'WER (%)':<10} {'First Token':<30} {'Error':<15}")
    print("-"*150)
    
    successful_ttft = []
    successful_wer = []
    failed_results = []
    
    for result in results:
        if isinstance(result, Exception):
            continue
        
        provider_name = result.provider
        
        ttft = f"{result.ttft_seconds:.3f}" if result.ttft_seconds else "N/A"
        wer = f"{result.wer_percentage:.1f}" if result.wer_percentage is not None else "N/A"
        first_token = (result.first_token_content[:27] + "...") if result.first_token_content and len(result.first_token_content) > 30 else (result.first_token_content or "")
        error = result.error[:12] + "..." if result.error and len(result.error) > 15 else (result.error or "")
        
        print(f"{provider_name:<25} {ttft:<10} {wer:<10} {first_token:<30} {error:<15}")
        
        if not result.error:
            if result.ttft_seconds:
                successful_ttft.append((provider_name, result.ttft_seconds))
            if result.wer_percentage is not None:
                successful_wer.append((provider_name, result.wer_percentage))
        else:
            failed_results.append(provider_name)
    
    return successful_ttft, successful_wer, failed_results

def display_analysis(successful_ttft: List[tuple], successful_wer: List[tuple], failed_results: List[str]):
    """Display detailed analysis of the TTFT results and WER."""
    # TTFT Analysis
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
    
    # WER Analysis
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
    
    # Provider Comparisons
    print("\nModel Comparisons within Providers:")
    
    # Deepgram comparisons
    deepgram_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('deepgram')]
    deepgram_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('deepgram')]
    
    if deepgram_ttft:
        deepgram_ttft.sort(key=lambda x: x[1])
        print(f"Deepgram TTFT fastest: {deepgram_ttft[0][0]} ({deepgram_ttft[0][1]:.3f}s)")
    if deepgram_wer:
        deepgram_wer.sort(key=lambda x: x[1])
        print(f"Deepgram WER best: {deepgram_wer[0][0]} ({deepgram_wer[0][1]:.1f}%)")
    
    # Google comparisons
    google_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('google')]
    google_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('google')]
    
    if google_ttft:
        google_ttft.sort(key=lambda x: x[1])
        print(f"Google TTFT fastest: {google_ttft[0][0]} ({google_ttft[0][1]:.3f}s)")
    if google_wer:
        google_wer.sort(key=lambda x: x[1])
        print(f"Google WER best: {google_wer[0][0]} ({google_wer[0][1]:.1f}%)")
    
    # Speechmatics comparisons
    speechmatics_ttft = [(name, time_val) for name, time_val in successful_ttft if name.startswith('speechmatics')]
    speechmatics_wer = [(name, wer_val) for name, wer_val in successful_wer if name.startswith('speechmatics')]
    
    if speechmatics_ttft:
        speechmatics_ttft.sort(key=lambda x: x[1])
        print(f"Speechmatics TTFT fastest: {speechmatics_ttft[0][0]} ({speechmatics_ttft[0][1]:.3f}s)")
    if speechmatics_wer:
        speechmatics_wer.sort(key=lambda x: x[1])
        print(f"Speechmatics WER best: {speechmatics_wer[0][0]} ({speechmatics_wer[0][1]:.1f}%)")
    
    # Failed results
    if failed_results:
        print(f"\n Failed: {', '.join(failed_results)}")

# Main benchmark execution
async def main():
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create providers
    providers = create_providers()
    
    if not providers:
        print("No providers could be created. Check your API keys and dependencies.")
        return
    
    print(f"Running dual benchmark suite for {len(providers)} provider/model combinations...")
    print("\nModels being tested:")
    for provider in providers:
        print(f"  - {provider.name}")
    
    # =====================================
    # TEST 1: Common Voice (TTFT + WER)
    # =====================================
    print("\n" + "="*60)
    print("TEST 1: COMMON VOICE - ACCURACY & PERFORMANCE BENCHMARK")
    print("="*60)
    
    try:
        print("Loading Common Voice sample from local files...")
        audio_data_cv, channels_cv, sample_width_cv, sample_rate_cv, filename_cv, duration_cv, ground_truth_cv = load_local_sample()
        
        print(f"Common Voice Sample Loaded:")
        print(f"   File: {filename_cv}")
        print(f"   Duration: {duration_cv:.2f}s")
        print(f"   Ground Truth: \"{ground_truth_cv}\"")
        print(f"   Ground Truth Length: {len(ground_truth_cv.split())} words, {len(ground_truth_cv)} characters")
        print(f"   Source: Local en_test folder")
        
        # Run all providers on Common Voice sample
        print(f"\nRunning Common Voice benchmark...")
        tasks_cv = [
            provider.measure_ttft(audio_data_cv, channels_cv, sample_width_cv, sample_rate_cv, 0.1, duration_cv)
            for provider in providers
        ]
        
        results_cv = await asyncio.gather(*tasks_cv, return_exceptions=True)
        
        # Process Common Voice results (with WER calculation)
        processed_results_cv = []
        for result in results_cv:
            if isinstance(result, Exception):
                processed_results_cv.append(result)
            else:
                processed_results_cv.append(process_transcription_result(result, ground_truth_cv, duration_cv))
        
        # Display Common Voice results
        print("\nCOMMON VOICE RESULTS (TTFT + WER):")
        successful_ttft_cv, successful_wer_cv, failed_results_cv = display_results_table(processed_results_cv)
        
        # Save Common Voice results to CSV
        save_results_to_csv(processed_results_cv, timestamp, filename_cv, "Common Voice TTFT+WER")
        
    except Exception as e:
        print(f"Common Voice test failed: {e}")
        processed_results_cv = []
        successful_ttft_cv, successful_wer_cv, failed_results_cv = [], [], []
    
    # =====================================
    # DELAY BETWEEN TESTS
    # =====================================
    print("\nWaiting 5 seconds between tests to reset API quotas...")
    await asyncio.sleep(5)
    
    # =====================================
    # TEST 2: Local test.wav (TTFT only)
    # =====================================
    print("\n" + "="*60)
    print("TEST 2: LOCAL test.wav - PERFORMANCE BENCHMARK (TTFT ONLY)")
    print("="*60)
    
    try:
        print("Loading local test.wav for performance testing...")
        if os.path.exists("test.wav"):
            audio_data_local, channels_local, sample_width_local, sample_rate_local, filename_local, duration_local = load_audio_file("test.wav")
            
            print(f"Local Test File Loaded:")
            print(f"   File: {filename_local}")
            print(f"   Duration: {duration_local:.2f}s")
            print(f"   Purpose: TTFT performance measurement only")
            
            # Run all providers on local test.wav (TTFT only - no WER)
            print(f"\nRunning local test.wav benchmark...")
            tasks_local = [
                provider.measure_ttft(audio_data_local, channels_local, sample_width_local, sample_rate_local, 0.1, duration_local)
                for provider in providers
            ]
            
            results_local = await asyncio.gather(*tasks_local, return_exceptions=True)
            
            # Process local results (NO WER or RTF calculation - only NTTFT)
            processed_results_local = []
            for result in results_local:
                if isinstance(result, Exception):
                    processed_results_local.append(result)
                else:
                    # Change TTFT to NTTFT for local test.wav results
                    if hasattr(result, 'ttft_seconds') and result.ttft_seconds is not None:
                        # Create a copy to avoid modifying the original result object
                        result_copy = result
                        # We'll handle the NTTFT vs TTFT distinction in save_results_to_csv
                    processed_results_local.append(result)
            
            # Display local results (TTFT only)
            print("\nLOCAL test.wav RESULTS (TTFT ONLY):")
            successful_ttft_local, _, failed_results_local = display_results_table(processed_results_local)
            
            # Save local results to CSV (append to same file with consistent timestamp for this run)
            save_results_to_csv(processed_results_local, timestamp, filename_local, "Local test.wav TTFT")
            
        else:
            print("test.wav not found - skipping local benchmark")
            processed_results_local = []
            successful_ttft_local, failed_results_local = [], []
            
    except Exception as e:
        print(f"Local test.wav benchmark failed: {e}")
        processed_results_local = []
        successful_ttft_local, failed_results_local = [], []
    
    # =====================================
    # COMBINED ANALYSIS
    # =====================================
    print("\n" + "="*60)
    print("COMBINED ANALYSIS")
    print("="*60)
    
    # Analyze Common Voice results
    if successful_ttft_cv or successful_wer_cv:
        print("\nCOMMON VOICE ANALYSIS:")
        display_analysis(successful_ttft_cv, successful_wer_cv, failed_results_cv)
    
    # Analyze local results  
    if successful_ttft_local:
        print("\nLOCAL test.wav ANALYSIS:")
        display_analysis(successful_ttft_local, [], failed_results_local)
    
    # Upload to database (single CSV file now contains both result sets)
    try:
        csv_file = "all_benchmarks.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            engine = create_engine(get_api_key("DATABASE_URL", secrets))
            df.to_sql('all_benchmarks', engine, if_exists='append', index=False)
            print("All benchmark data uploaded to database")
            
            # Delete CSV file after successful upload to prevent duplicates
            os.remove(csv_file)
            print(f"Deleted {csv_file} to prevent duplicate uploads")
            
    except Exception as e:
        logging.error(f"Error writing results to database: {e}")
    #     # Print first few rows of data for debugging
    #     if 'df' in locals():
    #         print("Sample data causing issues:")
    #         print(df[['provider', 'metric_type', 'metric_value', 'transcript']].head(10))
    #     print(f"CSV file kept for debugging: all_benchmarks.csv")
    
    # =====================================
    # SUMMARY
    # =====================================
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Total providers tested: {len(providers)}")
    
    # Common Voice summary
    cv_success = len([r for r in processed_results_cv if not isinstance(r, Exception) and not r.error]) if processed_results_cv else 0
    print(f"\nCommon Voice Test (Accuracy + Performance):")
    print(f"   Successful TTFT measurements: {len(successful_ttft_cv)}")
    print(f"   Successful WER measurements: {len(successful_wer_cv)}")
    print(f"   Sample source: Common Voice EN test set")
    
    # Local test summary
    local_success = len([r for r in processed_results_local if not isinstance(r, Exception) and not r.error]) if processed_results_local else 0
    print(f"\nLocal test.wav (Performance Only):")
    print(f"   Successful TTFT measurements: {len(successful_ttft_local)}")
    print(f"   Sample source: Local test.wav file")
    
    # Winners
    if successful_ttft_cv:
        ttft_winner_cv = min(successful_ttft_cv, key=lambda x: x[1])
        print(f"\nCommon Voice TTFT winner: {ttft_winner_cv[0]} ({ttft_winner_cv[1]:.3f}s)")
    
    if successful_wer_cv:
        wer_winner = min(successful_wer_cv, key=lambda x: x[1])
        print(f"Accuracy winner: {wer_winner[0]} ({wer_winner[1]:.1f}% WER)")
        
    if successful_ttft_local:
        ttft_winner_local = min(successful_ttft_local, key=lambda x: x[1])
        print(f"Local test.wav TTFT winner: {ttft_winner_local[0]} ({ttft_winner_local[1]:.3f}s)")

if __name__ == "__main__":
    # Run dual benchmark
    asyncio.run(main())