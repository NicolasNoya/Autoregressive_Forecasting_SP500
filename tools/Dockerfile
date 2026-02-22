# Step 1: Start from an official PyTorch + GPU base image
# (Ubuntu 22.04, Python 3.11, PyTorch, CUDA 12.6, torch 2.8)
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 2: Install system-level dependencies (if any)
# e.g., git, wget, or common libraries for OpenCV like libgl1
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -U pip

# Step 3: Copy and pre-install all Python dependencies
# This 'requirements.txt' file should list pandas, scikit-learn, timm, etc.
# Place it in the same directory as this Dockerfile.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install opencv-python-headless
