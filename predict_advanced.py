import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading Large Vision Transformer (BLIP Large)...")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Upgraded to 'large' model for high detail perception
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def predict_caption(image_path):
    if not os.path.exists(image_path):
        print(f"[!] Error: '{image_path}' file nahi mili!")
        return None

    raw_image = Image.open(image_path).convert('RGB')
    
    # Text prompt to guide the model for rich detailed output
    prompt = "a photography of"
    inputs = processor(raw_image, prompt, return_tensors="pt").to(device)

    # Advanced sampling strategy for full-scene accuracy
    out = model.generate(
        **inputs, 
        max_new_tokens=60, 
        num_beams=5,
        repetition_penalty=1.2
    )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    test_image_path = "meow.jpg"

    print(f"\nProcessing '{test_image_path}' for high detail...")
    caption = predict_caption(test_image_path)

    if caption:
        print("\n" + "="*60)
        print(f"DETAILED CAPTION: {caption.capitalize()}")
        print("="*60)