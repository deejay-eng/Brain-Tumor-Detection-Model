from flask import Flask, render_template, request
import os
from keras.models import load_model
from detect import detect

app = Flask(__name__)
model = load_model('model/model.h5')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
