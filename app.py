import os

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your models
cnn_model = load_model('model/custom_cnn_model.h5')  # replace with actual path if different
efficientnet_model = load_model('model/efficientnet_model.h5')  # replace with actual path if different

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    image = preprocess_image(filepath)

    # Make predictions
    cnn_prediction = cnn_model.predict(image)
    efficientnet_prediction = efficientnet_model.predict(image)

    # Get predicted labels
    cnn_label = class_labels[np.argmax(cnn_prediction)]
    efficientnet_label = class_labels[np.argmax(efficientnet_prediction)]

    return jsonify({
        'cnn_prediction': cnn_label,
        'efficientnet_prediction': efficientnet_label
    })

if __name__ == '__main__':
    app.run(debug=True)
