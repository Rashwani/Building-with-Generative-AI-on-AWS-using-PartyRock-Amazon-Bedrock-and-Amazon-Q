import streamlit as st
import boto3
import json
import base64
from io import BytesIO
from PIL import Image

# Constants
REGION = "us-west-2"
SD_PRESETS = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

# Functions
def generate_image(prompt, model, style=None):
    if model == "Stable Diffusion":
        body = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 10,
            "seed": 0,
            "steps": 50,
            "style_preset": style if style != "None" else None,
        }
        model_id = "stability.stable-diffusion-xl-v1"
    else:
        body = {
            "textToImageParams": {"text": prompt},
            "taskType": "TEXT_IMAGE",
            "imageGenerationConfig": {
                "cfgScale": 10,
                "seed": 0,
                "quality": "standard",
                "width": 512,
                "height": 512,
                "numberOfImages": 1,
            },
        }
        model_id = "amazon.titan-image-generator-v1"

    body = json.dumps(body)
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())["artifacts"][0]["base64"]

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

# User interface
st.title("Building with Bedrock")
st.subheader("Image Generation Demo")

prompt = st.text_input("Enter your prompt:")
model = st.selectbox("Select model", ["Stable Diffusion", "Amazon Titan"])

if model == "Stable Diffusion":
    style = st.selectbox("Select style", SD_PRESETS)
else:
    style = None

user_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

if user_image is not None:
    user_image = Image.open(user_image)
    col1.image(user_image)

    if col1.button("Update Image"):
        image_base64 = generate_image(prompt, model, style)
        new_image = base64_to_image(image_base64)
        col2.image(new_image)
else:
    col1.write("No image uploaded")
