from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import os
import requests
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "model.h5"
MODEL_URL = "https://github.com/kaanchiiii/Brain-Tumor-Detection-Model/releases/download/v1.0/model.h5"

# 🔽 Download model if it's not already present
if not os.path.exists(MODEL_PATH):
    print("🔄 Downloading model.h5...")
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    print("✅ model.h5 downloaded!")

# ✅ Load model
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("🚀 Model loaded successfully!")

# 🔍 Preprocessing helper
def preprocess_image(image):
    img = image.resize((150, 150))  # Adjust to match your model's expected size
    img = np.array(img) / 255.0
    img = img.reshape(1, 150, 150, 3)
    return img

# 🏠 Home route
@app.route('/')
def home():
    return render_template('index.html')

# 📤 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
