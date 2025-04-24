import os
import requests
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(_name_)

MODEL_PATH = 'model/model.h5'
MODEL_FILE_ID = '1JFkLRc-Hzy39KHlrqoR68vOdzwW-_uWh'
MODEL_URL = f'https://drive.google.com/uc?export=download&id=https://drive.google.com/file/d/1JFkLRc-Hzy39KHlrqoR68vOdzwW-_uWh/view?usp=sharing'

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs('model', exist_ok=True)
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded.")

# Load the model
model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))  # Change if your model uses a different input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        result = prediction.tolist()
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if _name_ == '_main_':
    app.run(debug=True)
