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
# torch is installed explicitly with the CPU-only wheel to keep the image small.
# The 'docker' package is a local dev tool and is excluded from the container.
RUN pip install --no-cache-dir --upgrade pip

# Install torch CPU wheel first (separate layer so it's cached independently)
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime deps (excluding torch and docker which are handled separately)
RUN pip install --no-cache-dir numpy pandas scikit-learn
