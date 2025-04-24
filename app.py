import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Create 'uploads' folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (assumes model.h5 is in the same directory as app.py)
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']

    if file.filename == '':
        return 'No file selected', 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))  # adjust size as per your model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize if model was trained this way

        # Predict using the model
        prediction = model.predict(img_array)

        # Assuming binary classification — 1 for Tumor, 0 for No Tumor
        result = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor Detected'
        return result

    return 'File processing error', 500

if __name__ == '__main__':
    app.run(debug=True)
