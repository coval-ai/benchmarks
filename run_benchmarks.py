#!/usr/bin/env python3
"""
Master script to run TTS and STT benchmarks
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting benchmark suite...")
    
    # Run TTS benchmark first
    print("\n🎵 Running TTS Benchmark...")
    os.chdir('tts')
    subprocess.run([sys.executable, 'run_tts.py'])
    os.chdir('..')
    
    # Run STT benchmark second
    print("\n🎤 Running STT Benchmark...")
    os.chdir('stt')
    subprocess.run([sys.executable, 'run_stt.py'])
    os.chdir('..')
    
    print("\n✅ Benchmark suite completed!")

if __name__ == "__main__":
    main()