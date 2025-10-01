#!/usr/bin/env bash
set -euo pipefail

echo "Starting Flask backend with Gunicorn..."

# Number of Gunicorn workers (1 worker for low memory environments)
WORKERS=${WEB_CONCURRENCY:-1}

# Number of threads per worker
THREADS=${GUNICORN_THREADS:-2}

# Port Render expects
PORT=${PORT:-5174}

exec gunicorn server:app \
    --bind 0.0.0.0:$PORT \
    --workers $WORKERS \
    --threads $THREADS \
    --timeout 120 \
    --worker-class gthread
