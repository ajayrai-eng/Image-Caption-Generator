import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# ==============================
# Paths
# ==============================

IMAGE_FOLDER = "dataset/images"
OUTPUT_FILE = "dataset/features.pkl"

# ==============================
# Load VGG16 Model
# ==============================

print("Loading VGG16 model...")

base_model = VGG16(weights="imagenet")

# Remove the final classification layer
model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

print("VGG16 model loaded successfully!")

# ==============================
# Extract Features
# ==============================

features = {}

image_files = os.listdir(IMAGE_FOLDER)

total = len(image_files)
count = 0

for image_name in image_files:

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    try:
        # Load image
        image = load_img(
            image_path,
            target_size=(224, 224)
        )

        # Convert image to array
        image = img_to_array(image)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Preprocess image
        image = preprocess_input(image)

        # Extract feature
        feature = model.predict(
            image,
            verbose=0
        )

        # Save feature
        features[image_name] = feature.flatten()

        count += 1

        if count % 100 == 0:
            print(
                f"Processed {count}/{total} images"
            )

    except Exception as e:

        print(
            f"Error processing {image_name}: {e}"
        )

# ==============================
# Save Features
# ==============================

with open(
    OUTPUT_FILE,
    "wb"
) as file:

    pickle.dump(
        features,
        file
    )

print("\n================================")
print("Feature Extraction Complete")
print("================================")
print("Total Features:", len(features))
print("Saved to:", OUTPUT_FILE)