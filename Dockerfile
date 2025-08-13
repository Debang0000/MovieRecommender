# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional):
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app source
COPY run.py ./
COPY web ./web
COPY movie_recommender ./movie_recommender

# Expose port
EXPOSE 5000

# Runtime config via envs
ENV APP_HOST=0.0.0.0 \
    APP_PORT=5000 \
    APP_DEBUG=false

CMD ["python", "run.py"] 