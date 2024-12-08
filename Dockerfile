# Use an official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire application into the container
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs \
    && chown -R nobody:nogroup /app

# Switch to non-root user
USER nobody

# Set the entrypoint command
CMD ["python", "-u", "/app/app.py"]