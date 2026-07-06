FROM python:3.10-slim

# Install system dependencies (FFmpeg is required by Whisper and Pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables (Hugging Face user id is 1000 by default, run in user space)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy files to user home directory and adjust permission
COPY --chown=user . $HOME/app

# Expose port (Hugging Face Spaces runs on 7860 by default)
EXPOSE 7860

# Run uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
