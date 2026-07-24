# VisionAI - AI Image Caption Generator

VisionAI is an AI-powered Image Caption Generator that automatically analyzes an uploaded image and generates a natural-language description.

The project combines **Computer Vision** and **Natural Language Processing** using **VGG16** for image feature extraction and a **CNN-LSTM based deep learning model** for caption generation.

---

## 🚀 Features

- Upload images through a web interface
- Generate captions automatically using AI
- VGG16-based image feature extraction
- CNN-LSTM deep learning architecture
- Flask-based web application
- Simple and user-friendly interface
- Supports JPG, JPEG, PNG and WEBP images

---

## 🧠 How It Works

The system follows this pipeline:

**Input Image**  
↓  
**VGG16 Feature Extraction**  
↓  
**Visual Feature Vector**  
↓  
**CNN-LSTM Caption Generation Model**  
↓  
**Generated Text Caption**

The uploaded image is first processed using VGG16 to extract meaningful visual features. These features are then passed to the trained caption generation model, which predicts the caption word by word.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- VGG16
- CNN
- LSTM
- Flask
- NumPy
- Pillow
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```text
Image-Caption-Generator/
│
├── app.py
├── feature_extractor.py
├── model_final.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── tokenizer.pkl
│   └── max_length.pkl
│
├── models/
│   └── model_weights.weights.h5
│
├── templates/
│   └── index.html
│
├── static/
│
└── .gitignore
⚙️ Installation
1. Clone the Repository
git clone https://github.com/ajayrai-eng/Image-Caption-Generator.git
2. Navigate to the Project Folder
cd Image-Caption-Generator
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

For Windows:

venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Flask application:

python app.py

The application will run locally at:

http://127.0.0.1:5000

Open this address in your web browser.

🖼️ Usage
Open the VisionAI web application.
Upload an image.
Preview the selected image.
Click on Generate Caption.
The AI model processes the image.
The generated caption is displayed on the screen.
🔬 Model Architecture

The project uses a combination of Computer Vision and Deep Learning techniques.

VGG16

VGG16 is used as a feature extractor to convert the input image into a numerical feature representation.

CNN

CNN-based visual features help the system understand the visual information present in the image.

LSTM

The LSTM network generates the caption sequentially, predicting the next word based on the image features and previously generated words.

📌 Current Status

The current version of the project is a working prototype demonstrating end-to-end image caption generation.

The model can process an uploaded image and generate an automatically predicted caption through the Flask web application.

The prediction quality can be further improved by training the model on larger and more diverse datasets and performing additional model optimization.

🔮 Future Scope
Train the model using significantly larger image-caption datasets
Improve caption accuracy and natural language quality
Implement advanced Transformer-based architectures
Add multilingual caption generation
Add voice output for generated captions
Deploy the application on a cloud platform
Improve inference speed and scalability
Add real-time image captioning using camera input
👨‍💻 Author

Ajay Rai

BE Information Technology Student

GitHub: ajayrai-eng

⭐ Project

If you find this project interesting, consider giving it a star ⭐
