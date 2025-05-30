FROM python:3.10-slim

WORKDIR /app

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision \
    diffusers==0.21.4 \
    transformers==4.33.2 \
    accelerate==0.23.0 \
    safetensors==0.4.0 \
    Pillow==10.2.0

COPY model ./model
COPY run_sdxl.py .

ENTRYPOINT ["python", "/app/run_sdxl.py"]
