import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
print("🚀 Model loaded successfully!")

def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediction = model.predict(img_tensor)[0][0]
    confidence = round(prediction * 100, 2) if prediction >= 0.5 else round((1 - prediction) * 100, 2)
    result = "Tumor Detected" if prediction >= 0.5 else "No Tumor Detected"
    return result, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        result, confidence = predict_tumor(filepath)
        image_path = url_for('static', filename='uploads/' + unique_filename)
        return render_template('result.html', result=result, confidence=confidence, image_path=image_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
