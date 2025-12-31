FROM python:3.14-slim

# Install build dependencies for Pillow (needed for ARM builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir requests Pillow

# Create directories
RUN mkdir -p /images /data

# Copy application files
COPY app.py /app/app.py
COPY VERSION /app/VERSION
RUN chmod +x /app/app.py

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-u", "app.py"]