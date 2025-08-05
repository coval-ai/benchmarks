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
RUN pip install --no-cache-dir -r requirements.txt boto3

# Copy shared modules to root
COPY wer_calculator.py secretmanager.py ./

# Copy the entire tts directory
COPY tts/ ./tts/

# Copy the entire stt directory  
COPY stt/ ./stt/

# Copy master script
COPY run_benchmarks.py ./

# Copy environment file (if present)
COPY .env* ./

# Create directories for audio files and outputs
RUN mkdir -p /app/audio_files /app/outputs /app/test_cases

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-east-2

# Default command - run the master script
CMD ["python", "run_benchmarks.py"]