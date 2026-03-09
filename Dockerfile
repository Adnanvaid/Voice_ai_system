# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# Expose port 
EXPOSE 8000

# Run the main script
CMD ["python", "voice_ai_system.py"]