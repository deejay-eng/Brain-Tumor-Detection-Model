from flask import Flask, render_template, request
import os
import requests
from keras.models import load_model
from detect import detect

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'model/model.h5'
MODEL_URL = 'https://github.com/kaanchiiii/Brain-Tumor-Detection-Model/releases/download/v1.0/model.h5'

# Ensure model is present or download it
def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model.h5 from GitHub Releases...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully.")
    else:
        print("Model already exists. Skipping download.")

download_model()
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded!', 400

    image = request.files['image']
    if image.filename == '':
        return 'No selected file!', 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    result, confidence = detect(image_path, model)

    return render_template('result.html', result=result, confidence=round(confidence*100, 2), image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
