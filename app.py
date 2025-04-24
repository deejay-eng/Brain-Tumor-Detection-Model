import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model from expected path
MODEL_PATH = 'model/model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model.h5 not found at 'model/model.h5'. Check preDeployCommand or file path.")

model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    img = load_img(filepath, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = 'Tumor Detected' if prediction > 0.5 else 'No Tumor'

    return render_template('result.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
