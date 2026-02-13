import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
device = "cpu"

import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny", torch_dtype=torch.float32
)
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.float32
).to(device)

prompt = """
A photorealistic overhead image of a cat reclining sideways in a flamingo pool floatie holding a margarita. 
The cat is floating leisurely in the pool and completely relaxed and happy.
"""

result = pipeline(
    prompt, 
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=10, 
    guidance_scale=3.5,
).images[0]

img = result.images[0]
img.save("cpu_flux_test.png")
print("Saved to cpu_flux_test.png")