#!/bin/sh
set -eu

if [ ! -f /app/models/production.joblib ]; then
  python /app/scripts/run_full_pipeline.py
fi

exec uvicorn api.main:app --host 0.0.0.0 --port 8000