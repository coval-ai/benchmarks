import asyncio
import csv
import datetime
import logging
import time
import pandas as pd
import os
import random
from sqlalchemy import create_engine
from dotenv import load_dotenv

from secretmanager import get_secret, get_api_key
from wer_calculator import compare_transcription

# Load secrets via AWS
print("Loading API keys from AWS Secrets Manager")
secrets = get_secret("prod/benchmarking", "us-east-2")

# Load secrets via .env
if not secrets:
    print("Loading API keys from .env")
    load_dotenv()

from openai import OpenAI # Whisper for transcription

from tts.providers.openai_tts import OpenAI_Benchmark
from tts.providers.cartesia_tts import Cartesia_Benchmark
from tts.providers.elevenlabs_tts import ElevenLabs_Benchmark
from tts.providers.hume_tts import Hume_Benchmark
from tts.providers.rime_tts import Rime_Benchmark
from tts.providers.deepgram_tts import Deepgram_Benchmark

TTS_PROVIDERS = {
    "OpenAI": OpenAI_Benchmark,
    "Cartesia": Cartesia_Benchmark,
    "ElevenLabs": ElevenLabs_Benchmark,
    "Deepgram": Deepgram_Benchmark,
    "Hume": Hume_Benchmark,
    "Rime": Rime_Benchmark
}

CONFIGURATIONS = {
    # "OpenAI": {
    #     "voice": "alloy",  # Options: "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer"
    #     "models": [
    #         "gpt-4o-mini-tts",
    #         "tts-1",
    #         "tts-1-hd",
    #         "gpt-realtime-2025-08-28"
    #     ]
    # },

    "ElevenLabs": {
        "voice": "IKne3meq5aSn9XLyUdCD",  # Use voice ID or name from your ElevenLabs account
        "models": [
            "eleven_flash_v2_5",
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5"
        ]
    },

    # "Cartesia": {
    #     "voice": "f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Voice ID from Cartesia
    #     "models": [
    #         "sonic-3",
    #         "sonic-2",
    #         "sonic-turbo",
    #         "sonic"
    #     ]
    # },

    # "Deepgram": {
    #     "voice": "aura-2-thalia-en",  # Options: aura-asteria-en, aura-luna-en, aura-stella-en etc.
    #     "models": [
    #         "aura-2-thalia-en"
    #     ]
    # },

    # "Hume": {
    #     "voice": "male_01",  # Check Hume documentation for available voices
    #     "models": [
    #         "octave-tts",
    #         "octave-2"
    #     ]
    # },

    # "Rime": {
    #     "voice": "luna",  
    #     "models": [
    #         "arcana",
    #         "mistv2"
    #     ]
    # }
}

async def TTFA_Benchmark(tts_provider, input_str):
    ttfa = await tts_provider.calculateTTFA(input_str)
    return ttfa

def load_test_cases(path):
    try:
        df = pd.read_csv(path, encoding='cp1252')
        test_cases = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Testcase ID']) and pd.notna(row['Transcript']):
                test_cases.append({
                    'testcase_id': row['Testcase ID'],
                    'transcript': row['Transcript'].strip(),
                })
        
        return test_cases
    except Exception as e:
        logging.error(f"Error loading test cases: {e}")
        return []
    
async def run_test(testcase, provider_name, model, voice, timestamp):
    
    print(f"Testing {testcase['testcase_id']} with {provider_name} - {model}.")
    
    ttfa_result = {
        'provider': provider_name,
        'model': model,
        'voice': voice,
        'benchmark': 'TTS',
        'metric_type': 'TTFA',
        'metric_value': None,
        'metric_units': 'ms',
        'audio_filename': None,
        'timestamp': timestamp,
        'status': 'failed'
    }
    
    wer_result = {
        'provider': provider_name,
        'model': model,
        'voice': voice,
        'benchmark': 'TTS',
        'metric_type': 'WER',
        'metric_value': None,
        'metric_units': '%',
        'audio_filename': None,
        'timestamp': timestamp,
        'status': 'failed'
    }
    
    try:
        # Step 1: Generate audio and calculate TTFA
        try:
            provider = TTS_PROVIDERS[provider_name]
            config = {'model': model, 'voice': voice}
            client = provider(config) 
            
            ttfa, audio_filename = await client.calculateTTFA(testcase['transcript'])
            
        except Exception as e:
            logging.error(f"Error generating audio with {provider_name}: {e}")
            ttfa = None
            audio_filename = None
        
        if ttfa is None or audio_filename is None:
            ttfa_result['status'] = 'tts_failed'
            wer_result['status'] = 'tts_failed'
            return [ttfa_result, wer_result]
        
        # Update TTFA result
        ttfa_result['metric_value'] = round(ttfa, 2)
        ttfa_result['audio_filename'] = audio_filename
        ttfa_result['status'] = 'success'
        
        # Step 2: Transcribe audio
        try:
            if not audio_filename or not os.path.exists(audio_filename):
                logging.error(f"Audio file not found: {audio_filename}")
                wer_result['status'] = 'audio_file_not_found'
                return [ttfa_result, wer_result]

            if secrets:    
                api_key = get_api_key('OPENAI_API_KEY', secrets)
            else:
                api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            client = OpenAI(api_key=api_key)
            
            # Transcribe audio file
            with open(audio_filename, "rb") as audio_file:
                openai_transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
            openai_hypothesis = openai_transcript.strip()

        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            wer_result['status'] = 'stt_failed'
            return [ttfa_result, wer_result]
        
        if openai_hypothesis is None:
            wer_result['status'] = 'stt_failed'
            return [ttfa_result, wer_result]
        
        # Step 3: Calculate WER
        try:
            wer_analysis = compare_transcription(testcase['transcript'], openai_hypothesis)
            custom_wer = wer_analysis['wer']
            
            if custom_wer is not None:
                wer_result['metric_value'] = round(custom_wer * 100, 2)
                wer_result['audio_filename'] = audio_filename
                wer_result['status'] = 'success'
                
                # Log detailed error analysis for debugging when there are differences
                if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                    logging.info(f"Custom WER errors for {testcase['testcase_id']}: {wer_analysis['incorrect_words']}")
                    logging.info(f"Original normalized: '{wer_analysis['normalized_original_text']}'")
                    logging.info(f"Transcription normalized: '{wer_analysis['normalized_transcription']}'")
            else:
                wer_result['status'] = 'wer_failed'
        except Exception as e:
            logging.error(f"Error calculating custom WER: {e}")
            wer_result['status'] = 'custom_calculation_failed'
        
        # Comment out to view the transcription 
        # print(f"Test case (normalized): {wer_analysis['normalized_original_text']}")
        # print(f"Transcribed (normalized): {wer_analysis['normalized_transcription']}")

        print(f"TTFA: {ttfa_result['metric_value']} ms, WER: {wer_result['metric_value']}%")
        
        # Optionally clean up audio file (comment out if you want to keep them)
        if os.path.exists(audio_filename):
            os.remove(audio_filename)
        
    except Exception as e:
        logging.error(f"Error in test {testcase['testcase_id']} with {provider_name}: {e}")
        ttfa_result['status'] = f'error: {str(e)}'
        wer_result['status'] = f'error: {str(e)}'
    
    return [ttfa_result, wer_result]

async def tts_benchmarks(test_cases):
    results = []
    
    # Calculate total tests: test_cases * providers * models
    total_tests = len(test_cases) * sum(len(provider_config['models']) for provider_config in CONFIGURATIONS.values())
    current_test = 0

    for testcase in test_cases:
        
        # Generate timestamp once per test case
        test_case_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for provider_name, provider_config in CONFIGURATIONS.items():
            for model in provider_config['models']:
                current_test += 1
                print(f"[{current_test}/{total_tests}] ", end="")
                
                test_results = await run_test(testcase, provider_name, model, provider_config['voice'], test_case_timestamp)
                results.extend(test_results)  # Add both TTFA and WER results
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
    
    # Write results to CSV
    output_file = "results.csv"
    try:
        fieldnames = [
            'provider', 'model', 'voice',
            'benchmark', 'metric_type', 'metric_value', 'metric_units',
            'audio_filename', 'timestamp', 'status'
        ]
        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([r for r in results if r is not None])

        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error writing results to CSV: {e}")
    
    # Save to database
    # try:
    #     df = pd.DataFrame(results)
    #     df.to_csv(output_file, index=False)
    #     engine = create_engine(get_api_key("DATABASE_URL", secrets))
    #     df.to_sql('all_benchmarks', engine, if_exists='append', index=False)
    #     print("Data uploaded to database.")
    # except Exception as e:
    #     logging.error(f"Error writing results to database: {e}")
    
    # Print summary
    successful_ttfa_tests = [r for r in results if r['status'] == 'success' and r['metric_type'] == 'TTFA']
    successful_wer_tests = [r for r in results if r['status'] == 'success' and r['metric_type'] == 'WER']
    
    print(f"\nTotal result rows: {len(results)}")
    print(f"Successful TTFA tests: {len(successful_ttfa_tests)}")
    print(f"Successful WER tests: {len(successful_wer_tests)}")
    print(f"Failed tests: {len(results) - len(successful_ttfa_tests) - len(successful_wer_tests)}")

    if successful_ttfa_tests:
        avg_ttfa = sum(r['metric_value'] for r in successful_ttfa_tests) / len(successful_ttfa_tests)
        print(f"\nAverage TTFA: {avg_ttfa:,.2f} ms")
        
        print(f"\nTTFA by Provider and Model:")
        for provider_name, provider_config in CONFIGURATIONS.items():
            for model in provider_config['models']:
                provider_model_results = [r for r in successful_ttfa_tests if r['provider'] == provider_name and r['model'] == model]
                if provider_model_results:
                    avg_ttfa = sum(r['metric_value'] for r in provider_model_results) / len(provider_model_results)
                    print(f"  {provider_name} - {model}: {avg_ttfa:,.2f} ms")
    
    if successful_wer_tests:
        avg_wer = sum(r['metric_value'] for r in successful_wer_tests) / len(successful_wer_tests)
        print(f"\nAverage WER: {avg_wer:,.2f}%")
        
        print(f"\nWER by Provider and Model:")
        for provider_name, provider_config in CONFIGURATIONS.items():
            for model in provider_config['models']:
                provider_model_results = [r for r in successful_wer_tests if r['provider'] == provider_name and r['model'] == model]
                if provider_model_results:
                    avg_wer = sum(r['metric_value'] for r in provider_model_results) / len(provider_model_results)
                    print(f"  {provider_name} - {model}: {avg_wer:,.2f}%")

database_path = "tts/Test cases.csv"  # Update this path as needed
test_cases = load_test_cases(database_path)

if not test_cases:
    print("No test cases found. Please check your file.")
else:
    print(f"Loaded {len(test_cases)} test cases")
    
    random_test_case = random.choice(test_cases)
    test_cases = [random_test_case]
    
    print(f"Running test case: {random_test_case['testcase_id']}")
    print(f"Text: {random_test_case['transcript']}")
    print("-" * 50)

asyncio.run(tts_benchmarks(test_cases))
print("\nBenchmark completed")