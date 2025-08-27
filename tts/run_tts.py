import asyncio
import argparse
import csv
import datetime
import logging
import time
import pandas as pd
import os
import random
from sqlalchemy import create_engine
from dotenv import load_dotenv

import boto3
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretmanager import get_secret, get_api_key
print("Loading API keys from AWS Secrets Manager...")
secrets = get_secret("prod/benchmarking")

load_dotenv()
from openai import OpenAI
from deepgram import DeepgramClient, ClientOptionsFromEnv, PrerecordedOptions
import assemblyai as aai
from rev_ai import apiclient

from providers.openai_tts import OpenAI_Benchmark
from providers.cartesia_tts import Cartesia_Benchmark
from providers.elevenlabs_tts import ElevenLabs_Benchmark
from providers.hume_tts import Hume_Benchmark
from providers.playht_tts import Playht_Benchmark
from providers.rime_tts import Rime_Benchmark

from wer_calculator import compare_transcription

async def TTFA_Benchmark(tts_provider, input_str):
    ttfa = await tts_provider.calculateTTFA(input_str)
    return ttfa

TTS_PROVIDERS = {
    "OpenAI": OpenAI_Benchmark,
    "Cartesia": Cartesia_Benchmark,
    "ElevenLabs": ElevenLabs_Benchmark,
    "Hume": Hume_Benchmark,
    "PlayHT": Playht_Benchmark,
    "Rime": Rime_Benchmark
}

CONFIGURATIONS = {
    "OpenAI": {
        "voice": "alloy",  # Options: "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer"
        "models": [
            "gpt-4o-mini-tts",
            "tts-1",
            "tts-1-hd"
        ]
    },
    "ElevenLabs": {
        "voice": "IKne3meq5aSn9XLyUdCD",  # Use voice ID or name from your ElevenLabs account
        "models": [
            "eleven_flash_v2_5",
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5"
        ]
    },
    "Cartesia": {
        "voice": "bf0a246a-8642-498a-9950-80c35e9276b5",  # Voice ID from Cartesia
        "models": [
            "sonic-2",
            "sonic-turbo",
            "sonic"
        ]
    },
    "Hume": {
        "voice": "male_01",  # Check Hume documentation for available voices
        "models": [
            "octave-tts"
        ]
    },
    # "PlayHT": {
    #     "voice": "s3://voice-cloning-zero-shot/b27bc13e-996f-4841-b584-4d35801aea98/original/manifest.json",  # Default voice or another voice ID
    #     "models": [
    #         "Play3.0-mini",
    #         "PlayDialog",
    #         "PlayDialogMultilingual"
    #     ]
    # },
    "Rime": {
        "voice": "cove",  # For "mist" model: "cove", for "mistv2" model: "breeze"
        "models": [
            "arcana",
            "mistv2"
        ]
    }
}

def load_test_cases(path):
    try:
        df = pd.read_csv(path, encoding='cp1252')
        test_cases = []
        
        for _, row in df.iterrows():
            if pd.notna(row['Testcase ID']) and pd.notna(row['Transcript']):
                test_cases.append({
                    'testcase_id': row['Testcase ID'],
                    'transcript': row['Transcript'].strip(),
                    'recording_conditions': row.get('Recording conditions', ''),
                    'mic': row.get('Mic', '')
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
        
        ttfa_result['metric_value'] = round(ttfa, 2)
        ttfa_result['audio_filename'] = audio_filename
        ttfa_result['status'] = 'success'
        
        try:
            if not audio_filename or not os.path.exists(audio_filename):
                logging.error(f"Audio file not found: {audio_filename}")
                wer_result['status'] = 'audio_file_not_found'
                return [ttfa_result, wer_result]
                
            api_key = get_api_key('OPENAI_API_KEY', secrets)
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            client = OpenAI(api_key=api_key)
            transcript_start = time.time()
            
            with open(audio_filename, "rb") as audio_file:
                openai_transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
            openai_transcript_time = (time.time() - transcript_start) * 1000
            openai_hypothesis = openai_transcript.strip()

        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            wer_result['status'] = 'stt_failed'
            return [ttfa_result, wer_result]
        
        if openai_hypothesis is None:
            wer_result['status'] = 'stt_failed'
            return [ttfa_result, wer_result]
        
        try:
            wer_analysis = compare_transcription(testcase['transcript'], openai_hypothesis)
            custom_wer = wer_analysis['wer']
            
            if custom_wer is not None:
                wer_result['metric_value'] = round(custom_wer * 100, 2)
                wer_result['audio_filename'] = audio_filename
                wer_result['status'] = 'success'
                
                if custom_wer > 0 and len(wer_analysis['incorrect_words']) > 0:
                    logging.info(f"Custom WER errors for {testcase['testcase_id']}: {wer_analysis['incorrect_words']}")
                    logging.info(f"Original normalized: '{wer_analysis['normalized_original_text']}'")
                    logging.info(f"Transcription normalized: '{wer_analysis['normalized_transcription']}'")
            else:
                wer_result['status'] = 'wer_failed'
        except Exception as e:
            logging.error(f"Error calculating custom WER: {e}")
            wer_result['status'] = 'custom_calculation_failed'
        
        print(f"TTFA: {ttfa_result['metric_value']} ms, WER: {wer_result['metric_value']}%")
        
        
    except Exception as e:
        logging.error(f"Error in test {testcase['testcase_id']} with {provider_name}: {e}")
        ttfa_result['status'] = f'error: {str(e)}'
        wer_result['status'] = f'error: {str(e)}'
    
    return [ttfa_result, wer_result]

async def tts_benchmarks(test_cases):
    results = []
    
    total_tests = len(test_cases) * sum(len(provider_config['models']) for provider_config in CONFIGURATIONS.values())
    current_test = 0

    print(f"Total tests to run: {total_tests}")
    print("-" * 50)

    for testcase in test_cases:
        print(f"\nProcessing test case: {testcase['testcase_id']}")
        print(f"Text: {testcase['transcript'][:50]}...")
        
        test_case_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for provider_name, provider_config in CONFIGURATIONS.items():
            for model in provider_config['models']:
                current_test += 1
                print(f"[{current_test}/{total_tests}] ", end="")
                
                test_results = await run_test(testcase, provider_name, model, provider_config['voice'], test_case_timestamp)
                results.extend(test_results)  # Add both TTFA and WER results
                
                await asyncio.sleep(0.5)
    
    output_file = "all_benchmarks.csv"
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
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        engine = create_engine(get_api_key("DATABASE_URL", secrets))
        df.to_sql('all_benchmarks', engine, if_exists='append', index=False)
        print("Data uploaded to database.")
    except Exception as e:
        logging.error(f"Error writing results to database: {e}")
    
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

database_path = "Test cases.csv"  # Update this path as needed
test_cases = load_test_cases(database_path)

if not test_cases:
    print("No test cases found. Please check your Excel file.")
else:
    print(f"Loaded {len(test_cases)} test cases")
    
    random_test_case = random.choice(test_cases)
    test_cases = [random_test_case]
    
    print(f"Running random test case: {random_test_case['testcase_id']}")
    print(f"Text: {random_test_case['transcript']}")

asyncio.run(tts_benchmarks(test_cases))
print("\nBenchmark completed!")