import subprocess
import sys


def main():
    print("Starting benchmark suite...")

    # Run TTS benchmark first
    print("\nRunning TTS Benchmark...")
    subprocess.run([sys.executable, 'run_tts.py'], check=True)

    # Run STT benchmark second
    print("\nRunning STT Benchmark...")
    subprocess.run([sys.executable, 'run_stt.py'], cwd='stt', check=True)

    print("\nBenchmark suite completed!")

if __name__ == "__main__":
    main()
