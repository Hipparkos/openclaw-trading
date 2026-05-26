# Use an official lightweight Python image
FROM python:3.12-slim

# Prevent Python from writing pyc files and enforce unbuffered stdout for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -r appuser

# Set the working directory
WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Ensure necessary directories exist and are owned by the non-root user
RUN mkdir -p data/logs data/database \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Run the application
CMD ["python", "src/main.py"]