import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model_final import create_final_model


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# MODEL PATHS
# ============================================================

TOKENIZER_PATH = "dataset/tokenizer.pkl"
MAX_LENGTH_PATH = "dataset/max_length.pkl"
WEIGHTS_PATH = "models/model_weights.weights.h5"


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# ============================================================
# LOAD MAX LENGTH
# ============================================================

print("Loading maximum caption length...")

with open(MAX_LENGTH_PATH, "rb") as f:
    max_length = pickle.load(f)


# ============================================================
# BUILD MODEL
# ============================================================

vocab_size = len(tokenizer.word_index) + 1

print("Building model architecture...")

model = create_final_model(
    vocab_size,
    max_length
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("Loading trained weights...")

model.load_weights(WEIGHTS_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD VGG16
# ============================================================

print("Loading VGG16...")

vgg_base = VGG16(
    weights="imagenet"
)

vgg_model = tf.keras.Model(
    inputs=vgg_base.inputs,
    outputs=vgg_base.layers[-2].output
)

print("VGG16 loaded successfully!")


# ============================================================
# CHECK FILE EXTENSION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# EXTRACT IMAGE FEATURES
# ============================================================

def extract_features(image_path):

    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = preprocess_input(
        image
    )

    features = vgg_model.predict(
        image,
        verbose=0
    )

    return features


# ============================================================
# INDEX TO WORD
# ============================================================

def idx_to_word(index):

    for word, word_index in tokenizer.word_index.items():

        if word_index == index:
            return word

    return None


# ============================================================
# GENERATE CAPTION
# ============================================================

def generate_caption(
    model_obj,
    image_feature,
    max_len
):

    in_text = "startseq"

    generated_words = []

    for _ in range(max_len):

        # Convert current sentence to sequence
        sequence = tokenizer.texts_to_sequences(
            [in_text]
        )[0]

        # Pad sequence
        sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="post"
        )

        # Predict next word
        predictions = model_obj.predict(
            [
                image_feature,
                sequence
            ],
            verbose=0
        )

        # Get top 10 predictions
        top_indices = np.argsort(
            predictions[0]
        )[-10:][::-1]

        selected_word = None

        last_word = in_text.split()[-1]

        # Select suitable word
        for index in top_indices:

            word = idx_to_word(
                index
            )

            if word is None:
                continue

            # Stop if end token
            if word in [
                "endseq",
                "end"
            ]:

                selected_word = "endseq"

                break

            # Avoid immediate repetition
            if word == last_word:
                continue

            selected_word = word

            break

        # Stop caption generation
        if (
            selected_word is None
            or
            selected_word == "endseq"
        ):

            break

        generated_words.append(
            selected_word
        )

        in_text += " " + selected_word

    # Final caption
    caption = " ".join(
        generated_words
    )

    return caption


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT CAPTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image file received."
            }), 400


        file = request.files["image"]


        # Check filename
        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400


        # Check extension
        if not allowed_file(
            file.filename
        ):

            return jsonify({
                "success": False,
                "error": "Invalid image format."
            }), 400


        # Secure filename
        filename = secure_filename(
            file.filename
        )


        # Create unique filename
        import time

        filename = (
            str(int(time.time()))
            + "_"
            + filename
        )


        # Image path
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )


        # Save image
        file.save(
            image_path
        )


        print(
            f"Processing image: {image_path}"
        )


        # Extract features
        features = extract_features(
            image_path
        )


        # Generate caption
        caption = generate_caption(
            model,
            features,
            max_length
        )


        print(
            f"Generated Caption: {caption}"
        )


        # If empty caption
        if not caption:

            caption = (
                "Unable to generate a meaningful caption."
            )


        # Return JSON
        return jsonify({

            "success": True,

            "caption": caption.capitalize(),

            "image": (
                "/"
                + image_path.replace(
                    "\\",
                    "/"
                )
            )

        })


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )