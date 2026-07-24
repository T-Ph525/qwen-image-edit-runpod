import runpod
from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel, GGUFQuantizationConfig
from diffusers.models.attention_processor import QwenDoubleStreamAttnProcessorFA3
from diffusers.utils import load_image
import torch
from io import BytesIO
import base64
from huggingface_hub import hf_hub_download

# Hugging Face Repository Information
REPO_ID = "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF"
FILENAME = "v23/v23/Qwen-Rapid-NSFW-v23_Q5_K.gguf"

def load_pipeline():
    """
    Load and optimize the Qwen Image Edit pipeline using GGUF transformer weights.
    """
    print("Downloading GGUF weights...")
    gguf_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    print("Loading GGUF quantized transformer...")
    # Load the quantized GGUF model into the transformer component
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16
    )

    print("Initializing Qwen Image Edit Pipeline...")
    # Load base pipeline and replace its transformer with the GGUF model
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

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
pipe = load_pipeline()

def handler(event):
    """
    Runpod serverless handler function. Receives job input and returns output.
    """
    try:
        input_data = event["input"]
        prompt = input_data.get("prompt", "Enhance the image")
        image_url = input_data.get("image_url")
        # Default steps for Rapid/Lightning merged models is usually 4-8
        num_inference_steps = input_data.get("num_inference_steps", 4) 
        guidance_scale = input_data.get("guidance_scale", 3.5)

        if not image_url:
            return {"error": "Missing 'image_url' parameter."}

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
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"output_image_base64": img_str, "prompt": prompt}
    except Exception as e:
        return {"error": str(e)}

# Required by Runpod
runpod.serverless.start({"handler": handler})
