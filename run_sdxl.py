import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Load environment variables
prompt = os.environ.get("PROMPT", "A test image")
negative_prompt = os.environ.get("NEGATIVE_PROMPT", "")
guidance_scale = float(os.environ.get("GUIDANCE_SCALE", 7.5))
num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", 30))
width = int(os.environ.get("WIDTH", 1024))
height = int(os.environ.get("HEIGHT", 1024))
seed = os.environ.get("SEED")
force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"

# Setup device and dtype
device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
generator = torch.manual_seed(int(seed)) if seed else None

print(f"🚀 Running on: {device.upper()} | DTYPE: {dtype}")
print("📦 Loading SDXL Base model...")

# Load model from baked directory
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/models",
    torch_dtype=dtype,
    local_files_only=True
).to(device)

# Generate the image
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

# Save to output
os.makedirs("/outputs", exist_ok=True)
image_path = "/outputs/output.png"
image.save(image_path)
print(f"✅ Image saved to {image_path}")
