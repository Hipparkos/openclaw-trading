FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PYPPETEER_ARGS="--no-sandbox --disable-setuid-sandbox"

RUN useradd -m -r appuser
WORKDIR /app
COPY requirements.txt .

# This line installs the system-level Chromium browser alongside your Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ python3-dev chromium \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN mkdir -p data/logs data/database \
    && chown -R appuser:appuser /app

USER appuser
CMD ["python", "src/main.py"]