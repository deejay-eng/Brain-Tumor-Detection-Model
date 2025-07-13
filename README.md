# 🧠 Brain Tumor Detection Using Deep Learning

This project is a deep learning-based image classification system that detects the presence and type of brain tumor from MRI scans. It uses a **custom CNN architecture** as well as **EfficientNetB0**, a state-of-the-art pre-trained model, to classify brain MRI images into multiple categories.

---

## 🔍 Project Overview

Brain tumors can be life-threatening if not diagnosed early. Manual diagnosis via MRI images is time-consuming and can vary based on expertise. This project leverages the power of deep learning to automate the classification of brain tumors with high accuracy.

### 📂 Dataset
- The dataset used consists of labeled brain MRI images.
- Images are categorized into:
  - **Glioma**
  - **Meningioma**
  - **Pituitary tumor**
  - **No tumor**

> *(Note: The dataset is not included in this repository due to size and copyright. You can use any publicly available brain tumor MRI dataset.)*

---

## 🧠 Techniques Used

- Image Preprocessing: Resizing, normalization, augmentation
- **Custom CNN Model**: Built using Keras & TensorFlow
- **EfficientNetB0**: Fine-tuned with transfer learning
- Model Evaluation: Accuracy, Confusion Matrix, Classification Report

---

## 📦 Project Structure
brain-tumor-detection/
│
├── EfficientNetB0_model.ipynb       # Notebook using pre-trained model
├── Custom_CNN_model.ipynb           # Notebook using custom CNN
├── requirements.txt                 # List of required libraries
├── README.md                        # Project description
│
├── models/                          # (Optional) Saved model weights
├── images/                          # Sample input images
└── dataset/                         # Directory for your MRI image dataset


---

## 🛠️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```
### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```
Or manually install:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python scikit-learn
```
### Step 3: Run the Notebooks
Open either EfficientNetB0_model.ipynb or Custom_CNN_model.ipynb in Jupyter Notebook or Google Colab

Modify the dataset path as needed in the code

Run the cells to train and evaluate the models

### ✅ Output
After training, the model produces:

Accuracy and loss graphs

Confusion matrix for visualizing predictions

A classification report with precision, recall, and F1-score


### 🚀 Future Enhancements
Add a user-friendly web interface for uploading images

Deploy the model for live prediction

Test with a larger dataset and try Vision Transformer-based models

### 🧰 Tools & Technologies
Python

TensorFlow / Keras

OpenCV

Scikit-learn

Jupyter Notebook / Google Colab

EfficientNet (via Keras Applications)

### ✨ Acknowledgments
Dataset from Kaggle: Brain MRI Images for Brain Tumor Detection

Inspired by various open-source ML projects for health diagnostics.


### ⚠️ Disclaimer
This project is intended for educational purposes only. It is not suitable for medical diagnosis or treatment decisions.


### 📸 Sample Output

![image](https://github.com/user-attachments/assets/2af02940-e99a-48b6-980b-1f07e6f66ee9)

Tumor Image:

![image](https://github.com/user-attachments/assets/78cb37f8-badc-4966-86fb-d2f4b3a945b9)

Result Image:

![image](https://github.com/user-attachments/assets/ebd477eb-a185-41ea-b3b2-3ae35fbd720f)

--------


## 📫 Contact Me  
📧 Email: jaindhaani0919@gmail.com
💼 LinkedIn: www.linkedin.com/in/dhaani-jain-09b9482a0  
💻 GitHub: https://github.com/deejay-eng
🌐 Portfolio: https://deejay-eng.github.io/Portfolio/

---

⭐ Feel free to check out my work and connect with me! If you like my portfolio, consider starring this repository!


