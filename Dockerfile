FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (including boto3 for AWS integration)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the providers directory first (for better layer caching)
COPY tts/providers/ ./providers/

# Copy main scripts and helper modules
COPY tts/runner.py tts/stt_script.py tts/wer_calculator.py tts/aws_secrets.py ./

# Copy any additional Python modules
COPY *.py ./

# Create directories for audio files and outputs
RUN mkdir -p /app/audio_files /app/outputs /app/test_cases /tmp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-east-1

# Default command (can be overridden via EventBridge)
CMD ["python", "runner.py"]