# Use an official Python 3.10 image.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Install system-level dependencies (git is required to clone repositories, and others may be needed for packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# We install the PyTorch packages from the official PyTorch wheel repository (adjust the URL if you need a different CUDA version).
RUN pip install --upgrade pip && \
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install diffusers==0.25.0 && \
    pip install accelerate==0.31.0 && \
    pip install "transformers>=4.25.1" && \
    pip install ftfy && \
    pip install Jinja2 && \
    pip install datasets && \
    pip install wandb && \
    pip install onnxruntime-gpu==1.19.2 && \
    pip install omegaconf && \
    pip install einops && \
    pip install torchmetrics && \
    pip install clean-fid && \
    pip install scikit-image && \
    pip install opencv-python && \
    pip install fvcore && \
    pip install cloudpickle && \
    pip install pycocotools && \
    pip install av && \
    pip install scipy && \
    pip install peft && \
    pip install huggingface-hub==0.24.6

COPY . .

CMD ["bash", "-c", "CUDA_VISIBLE_DEVICES=0 python inference.py --config_p './configs/VITONHD.yaml' --pretrained_unet_path './checkpoints/VITONHD/model/pytorch_model.bin' --save_name VITONHD"]