# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools just for building wheels; keep libgomp1 (OpenMP runtime)
# We will purge build-essential after pip install to keep image small

# Copy minimal files first to leverage Docker layer caching
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libgomp1 \
	&& pip install --no-cache-dir -r requirements.txt \
	&& apt-get purge -y --auto-remove build-essential \
	&& rm -rf /var/lib/apt/lists/*

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