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
RUN pip install -r requirements.txt

# Install latest diffusers from GitHub for cutting-edge features
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Copy handler script
COPY handler.py .

# (Optional) Pre-download the model to reduce cold start latency
# Uncomment the following lines to include the model in the image
#RUN python -c "import torch; from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.float16)"

# Set environment variables for optimization
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0,8.6,8.9,9.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# Set entrypoint to run the handler
CMD ["python", "handler.py"]
