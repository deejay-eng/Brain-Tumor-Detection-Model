from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Paths
MODEL_PATH = 'model.h5'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("🚀 Model loaded successfully!")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    confidence = round(np.max(predictions) * 100, 2)
    class_index = np.argmax(predictions)

    # Class labels (you can change if needed)
    class_labels = ['No Tumor', 'Tumor']
    result = class_labels[class_index]

    return render_template('result.html',
                           result=result,
                           confidence=confidence,
                           image_path='/' + image_path)

if __name__ == '__main__':
    app.run(debug=True)
