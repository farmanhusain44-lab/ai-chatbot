FROM python:3.12-slim

# System deps: ffmpeg for video ops, gcc for any wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Runtime env
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

CMD gunicorn server:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2
