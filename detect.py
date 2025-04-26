from keras.preprocessing.image import load_img, img_to_array
import numpy as np

class_labels = ['pituitary', 'notumor', 'glioma', 'meningioma']

def detect(img_path, model, image_size=128):
    try:
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        if class_labels[predicted_class_index] == 'notumor':
            result = "No Tumor"
        else:
            result = f"Tumor: {class_labels[predicted_class_index]}"

        return result, confidence_score

    except Exception as e:
        return "Error", 0.0
