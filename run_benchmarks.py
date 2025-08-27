#!/usr/bin/env python3

import subprocess
import sys
import os


def main():
    print("Starting benchmark suite...")

    print("Running TTS Benchmark...")
    os.chdir("tts")
    subprocess.run([sys.executable, "run_tts.py"])
    os.chdir("..")

    print("Running STT Benchmark...")
    os.chdir("stt")
    subprocess.run([sys.executable, "run_stt.py"])
    os.chdir("..")

    print("Benchmark suite completed!")


if __name__ == "__main__":
    main()
