# 🧠 Brain Tumor MRI Classification using Deep Learning

This project uses Convolutional Neural Networks (CNNs) with data augmentation to classify MRI brain images into tumor and non-tumor categories.

---

## 📥 Dataset

**Kaggle Dataset Link:**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing

dataset/
├── Training/
│ ├── yes/
│ └── no/
└── Testing/
├── yes/
└── no/

yaml
Copy
Edit

---

## 🎯 Objectives

- Binary classification: Tumor vs. No Tumor
- Apply data augmentation for better generalization
- Evaluate model using accuracy, precision, recall, F1-score, and confusion matrix

---

## 🔁 Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
🖼️ Sample MRI Image


🛠️ How to Run
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Prepare Dataset
Place the downloaded dataset inside the dataset/ folder.

Copy

Edit
dataset/
├── Training/
├── Testing/
4. Train the Model
bash
Copy
Edit
python train.py
5. Evaluate the Model
bash
Copy
Edit
python evaluate.py
📊 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix
