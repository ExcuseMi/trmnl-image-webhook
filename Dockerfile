FROM python:3.14-slim

# Install dependencies
RUN pip install --no-cache-dir requests Pillow

# Create directories
RUN mkdir -p /images /data

# Copy application
COPY app.py /app/app.py
RUN chmod +x /app/app.py

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-u", "app.py"]