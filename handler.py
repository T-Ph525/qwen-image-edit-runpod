import runpod
from diffusers import QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel, GGUFQuantizationConfig
from diffusers.models.attention_processor import QwenDoubleStreamAttnProcessorFA3
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from huggingface_hub import hf_hub_download
import boto3
import os
from urllib.parse import urlparse

# Hugging Face Repository Information
REPO_ID = "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF"
FILENAME = "v23/Qwen-Rapid-NSFW-v23_Q5_K.gguf"

# S3 Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

def get_s3_client():
    """Initialize S3 client with optional custom endpoint."""
    if S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY:
        return boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION
        )
    return None

def download_from_s3(s3_url):
    """Download file from S3-compatible storage."""
    try:
        s3_client = get_s3_client()
        if not s3_client:
            print(f"S3 credentials not configured, treating as regular URL: {s3_url}")
            return None
        
        parsed_url = urlparse(s3_url)
        bucket = parsed_url.netloc.split('.')[0]
        key = parsed_url.path.lstrip('/')
        
        print(f"Downloading from S3: s3://{bucket}/{key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"S3 download failed: {e}")
        return None

def load_transformers():
    """
    Load the Qwen Image Edit pipeline using only GGUF transformer weights.
    """
    print("Downloading GGUF weights...")
    gguf_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    print("Loading GGUF quantized transformer...")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16
    )

    print("Creating pipeline with GGUF transformer...")
    pipe = QwenImageEditPlusPipeline(
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # Apply FlashAttention-3 processor if supported
    try:
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    except Exception as e:
        print(f"FA3 Processor warning: {e}. Falling back to default attention.")

    # Enable memory efficiency
    pipe.enable_attention_slicing()

    print("Pipeline loaded and optimized successfully.")
    return pipe

# Load model on startup
pipe = load_transformers()

def handler(event):
    """
    Runpod serverless handler function. Receives job input and returns output.
    Supports both regular image URLs and S3-compatible storage URLs.
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        output_s3_path = input_data.get("output_s3_path")  # Optional S3 output path
        # Default steps for Rapid/Lightning merged models is usually 4-8
        num_inference_steps = input_data.get("num_inference_steps", 4) 
        guidance_scale = input_data.get("guidance_scale", 3.5)

        if not image_url:
            return {"error": "Missing 'image_url' parameter."}

        # Try to load image from S3 if URL appears to be S3
        if image_url.startswith("s3://") or (S3_ENDPOINT and image_url.startswith("http")):
            print("Loading image from S3...")
            image_data = download_from_s3(image_url)
            if image_data:
                input_image = load_image(BytesIO(image_data))
            else:
                print("Falling back to direct URL loading...")
                input_image = load_image(image_url)
        else:
            input_image = load_image(image_url)

        with torch.no_grad():
            output_image = pipe(
                image=input_image, 
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        buffered.seek(0)
        img_data = buffered.getvalue()
        img_str = base64.b64encode(img_data).decode("utf-8")

        result = {"output_image_base64": img_str, "prompt": prompt}

        # Upload to S3 if output path specified
        if output_s3_path:
            try:
                s3_client = get_s3_client()
                if s3_client:
                    parsed_url = urlparse(output_s3_path)
                    bucket = parsed_url.netloc.split('.')[0]
                    key = parsed_url.path.lstrip('/')
                    
                    print(f"Uploading to S3: s3://{bucket}/{key}")
                    s3_client.put_object(Bucket=bucket, Key=key, Body=img_data, ContentType="image/png")
                    result["output_s3_url"] = output_s3_path
                else:
                    print("S3 credentials not configured for upload")
            except Exception as e:
                print(f"S3 upload failed: {e}")
                result["s3_upload_error"] = str(e)

        return result
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
