# Use a modern Runpod PyTorch base image with CUDA support
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install latest diffusers from GitHub for cutting-edge features
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Copy handler script
COPY handler.py .

# Set environment variables for optimization and faster downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0,8.6,8.9,9.0"

# Pre-download the base model components and the GGUF transformer during build
RUN python -c "\
import torch; \
from huggingface_hub import hf_hub_download; \
from diffusers import QwenImageEditPlusPipeline; \
print('Pre-downloading GGUF weights...'); \
hf_hub_download(repo_id='Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF', filename='v23/v23/Qwen-Rapid-NSFW-v23_Q5_K.gguf'); \
print('Pre-downloading base pipeline components...'); \
QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2511', torch_dtype=torch.bfloat16); \
print('Pre-download complete!')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# Set entrypoint to run the handler
CMD ["python", "handler.py"]
