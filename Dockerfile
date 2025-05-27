FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt

# Copy the model and code
COPY model /models
COPY run_sdxl.py /app/run_sdxl.py
WORKDIR /app

CMD ["python3.10", "run_sdxl.py"]
