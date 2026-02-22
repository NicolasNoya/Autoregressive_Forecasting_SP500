# ─────────────────────────────────────────────────────────────────────────────
# Codabench Docker image — S&P 500 Autoregressive Forecasting Challenge
#
# Codabench mounts the following directories at runtime:
#   /app/input_data        ← input splits (train/, test/, private_test/)
#   /app/ingested_program  ← participant's submission.py
#   /app/ingestion_program ← ingestion.py (organiser code)
#   /app/scoring_program   ← scoring.py   (organiser code)
#   /app/output            ← ingestion writes predictions here
#   /app/input/ref         ← reference labels (scoring)
#   /app/input/res         ← predictions to score (scoring)
#
# Build from the PROJECT ROOT, not from tools/:
#   docker build -t sp500-challenge -f tools/Dockerfile .
#
# The build context must be the project root so that requirements.txt is
# accessible via COPY.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps: gcc is needed to compile some numpy/pandas C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-install Python dependencies.
# The 'docker' package is a local dev tool — skip it inside the container.
# torch is installed from the official CPU wheel index to avoid pulling the
# full CUDA build (~2 GB saved).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && grep -v "^docker" /tmp/requirements.txt \
    | grep -v "^#" \
    | grep -v "^$" \
    | grep -v "torch" \
    | pip install --no-cache-dir -r /dev/stdin \
 && pip install --no-cache-dir \
        torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
