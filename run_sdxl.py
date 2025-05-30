import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import traceback

# Load environment variables
prompt = os.environ.get("PROMPT", "A test image")
negative_prompt = os.environ.get("NEGATIVE_PROMPT", "")
guidance_scale = float(os.environ.get("GUIDANCE_SCALE", 7.5))
num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", 30))
width = int(os.environ.get("WIDTH", 1024))
height = int(os.environ.get("HEIGHT", 1024))
seed = os.environ.get("SEED")
generator = torch.manual_seed(int(seed)) if seed else None

# Force GPU and float16
device = "cuda"
dtype = torch.float16

print(f"🚀 Starting SDXL job")
print(f"🧠 Prompt: {prompt}")
print(f"📏 Size: {width}x{height} | Steps: {num_inference_steps} | Scale: {guidance_scale} | Seed: {seed or 'random'}")
print(f"🖥️ Device: {device} | DTYPE: {dtype}")

# Check model directory contents
try:
    model_files = os.listdir("./model")
    print(f"📁 /model contents: {model_files}")
except Exception as e:
    print(f"❌ Could not list /model directory: {e}")
    traceback.print_exc()
    exit(1)

# Load the model
try:
    print("📦 Loading SDXL Base model from /model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "./model",
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# Run inference
try:
    print("🎨 Generating image...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator
    )
    image = result.images[0]

    os.makedirs("/outputs", exist_ok=True)
    image_path = "/outputs/output.png"
    image.save(image_path)
    print(f"✅ Image successfully saved to {image_path}")
except Exception as e:
    print(f"❌ Inference failed: {e}")
    traceback.print_exc()
    exit(1)
