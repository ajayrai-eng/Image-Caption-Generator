import io
import base64
from PIL import Image
from flask import Flask, request, jsonify, render_template

# PyTorch & HuggingFace BLIP Imports
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# ==========================================
# 1. LOAD BLIP MODEL AT STARTUP
# ==========================================
print("Loading BLIP Transformer Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using the base BLIP model to ensure fast inference and low memory consumption
MODEL_NAME = "Salesforce/blip-image-captioning-base"

blip_processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"✅ BLIP Model ({MODEL_NAME}) loaded successfully on {device}!")


# ==========================================
# 2. CAPTION GENERATION LOGIC
# ==========================================
def generate_blip_caption(image: Image.Image) -> str:
    """Generates high-accuracy image captions using Salesforce BLIP."""
    try:
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption.capitalize()
    except Exception as e:
        print(f"Error during BLIP caption generation: {e}")
        return "An error occurred while generating the caption."


# ==========================================
# 3. FLASK ROUTES
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = None

        # A. Handle Live Camera Base64 Image (JSON Request)
        if request.is_json:
            data = request.get_json()
            image_data = data.get("image_data", "")

            if "," in image_data:
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # B. Handle Direct File Upload (Multipart Form Data)
        elif "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")

        if image is None:
            return jsonify({"success": False, "error": "No valid image provided"}), 400

        # Generate Caption using the BLIP Model
        caption = generate_blip_caption(image)

        return jsonify({"success": True, "caption": caption})

    except Exception as e:
        print(f"Prediction Route Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)