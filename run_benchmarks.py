import subprocess
import sys
import os

def main():
    print("Starting benchmark suite...")
    
    # Run TTS benchmark first
    print("\nRunning TTS Benchmark...")
    
    subprocess.run([sys.executable, 'run_tts.py'])
   
    
    # Run STT benchmark second
    print("\nRunning STT Benchmark...")
    os.chdir('stt')
    subprocess.run([sys.executable, 'run_stt.py'])
    os.chdir('..')
    
    print("\nBenchmark suite completed!")

if __name__ == "__main__":
    main()