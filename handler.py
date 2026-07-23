import runpod
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from PIL import Image

# Import additional components for optimization
from diffusers.models import QwenImageTransformer2DModel
from diffusers.models.attention_processor import QwenDoubleStreamAttnProcessorFA3
from diffusers.pipelines import QwenImageEditPlusPipeline

# Load model on startup with optimizations
def load_pipeline():
    """
    Load and optimize the Qwen Image Edit pipeline with Lightning LoRA and attention optimizations.
    """
    # Initialize pipeline with safetensors model
    pipe = QwenImageEditPlusPipeline.from_single_file(
        "path/to/Qwen-Rapid-AIO-NSFW-v21.safetensors",
        original_config="Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Load and fuse Lightning LoRA
    print("Loading lightning lora...")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning", 
        weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    )
    pipe.fuse_lora()
    print("Lightning lora fused.")
    
    # Apply transformer optimizations
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    
    # Enable memory efficient attention
    pipe.enable_attention_slicing()
    
    print("Pipeline loaded and optimized.")
    
    return pipe

# Load model on startup
pipe = load_pipeline()

def handler(event):
    """
    Runpod serverless handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        num_inference_steps = input_data.get("num_inference_steps", 4)
        guidance_scale = input_data.get("guidance_scale", 7.5)

        if not image_url:
            return {"error": "Missing 'image_url' parameter."}

        input_image = load_image(image_url)
        
        # Run inference with float16
        with torch.no_grad():
            output_image = pipe(
                image=input_image, 
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
