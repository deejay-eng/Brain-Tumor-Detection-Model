from flask import Flask, render_template, request
import os
from keras.models import load_model
from detect import detect
import urllib.request

app = Flask(__name__)

# Path to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model once on startup
model_path = 'model.h5'

# Check if model is already downloaded or exists in the directory
if not os.path.exists(model_path):
    MODEL_URL = 'https://github.com/kaanchiiii/Brain-Tumor-Detection-Model/releases/download/v1.0/model.h5'
    urllib.request.urlretrieve(MODEL_URL, model_path)
    
model = load_model(model_path)

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

    # Save the image to a specific folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    try:
        # Call detect function from result.py for processing
        result, confidence = detect(image_path, model)

        # Render the result page with the output
        return render_template('result.html', result=result, confidence=round(confidence*100, 2), image_path=image_path)
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
