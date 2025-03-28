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
    pip install transformers==4.41.0 && \
    pip install sentence-transformers==3.4.1 && \
    pip install ftfy==6.1.1 && \
    pip install Jinja2==3.1.3 && \
    pip install datasets==2.18.0 && \
    pip install wandb==0.16.6 && \
    pip install onnxruntime-gpu==1.19.2 && \
    pip install omegaconf==2.3.0 && \
    pip install einops==0.7.0 && \
    pip install torchmetrics==1.3.2 && \
    pip install clean-fid==0.1.35 && \
    pip install scikit-image==0.22.0 && \
    pip install opencv-python==4.9.0.80 && \
    pip install fvcore==0.1.5.post20221221 && \
    pip install cloudpickle==3.0.0 && \
    pip install pycocotools==2.0.7 && \
    pip install av==12.0.0 && \
    pip install scipy==1.13.0 && \
    pip install peft==0.13.2 && \
    pip install huggingface-hub==0.24.6 && \
    pip install numpy==1.26.4

COPY . .

CMD ["bash", "-c", "CUDA_VISIBLE_DEVICES=0 python inference.py --config_p './configs/VITONHD.yaml' --pretrained_unet_path './checkpoints/VITONHD/model/pytorch_model.bin' --init_model_path './pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1' --init_cloth_encoder_path './pretrained_models/stable-diffusion-xl-base-1.0' --init_vae_path './pretrained_models/sdxl-vae-fp16-fix' --save_name VITONHD"]