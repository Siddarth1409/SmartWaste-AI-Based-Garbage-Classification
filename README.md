# SmartWaste: AI-Based Garbage Classification ♻️

SmartWaste is a Deep Learning project that classifies waste into **Organic** and **Recyclable** categories using a Convolutional Neural Network (CNN).  
This project helps in promoting sustainable waste management by automating garbage classification.

---

## 🚀 Features
- Classifies waste images into **Organic** or **Recyclable**.
- Built using **TensorFlow/Keras** with CNN.
- Training on the **Waste Classification Dataset** from Kaggle.
- Includes visualization of training history and sample predictions.
- Easy-to-use prediction script for new images.

---

## 📂 Dataset
We use the **Waste Classification Data** by [techsash on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

Dataset contains:
- **TRAIN**  
  - `Organic/` → images of organic waste  
  - `Recyclable/` → images of recyclable waste  
- **TEST**  
  - `Organic/`  
  - `Recyclable/`

You can download it automatically with the provided KaggleHub script in the notebook/code.

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Siddarth1409/SmartWaste-AI-Based-Garbage-Classification.git
   cd SmartWaste-AI-Based-Garbage-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook notebooks/waste_classification.ipynb
   ```

---

## ▶️ Usage

### Training the Model
Run the notebook or script to train the CNN on the dataset:
```bash
python src/train.py
```
This will train the model and save it as `waste_classifier_cnn.h5`.

### Predicting New Images
Upload a test image and run:
```bash
python src/predict.py --image path/to/image.jpg
```
The model will output:
- **Organic**
- **Recyclable**

---

## 📊 Project Structure

```bash
SmartWaste-AI-Based-Garbage-Classification/
│── data/                          # (optional, dataset is large, not uploaded to GitHub)
│   ├── TRAIN/
│   │   ├── Organic/
│   │   └── Recyclable/
│   └── TEST/
│       ├── Organic/
│       └── Recyclable/
│
│── notebooks/
│   └── waste_classification.ipynb # Jupyter notebook (main code)
│
│── src/
│   ├── train.py                   # Training script
│   └── predict.py                 # Prediction script
│
│── requirements.txt               # Required libraries
│── waste_classifier_cnn.h5        # Saved model (not uploaded due to size)
│── README.md                      # Project documentation
```

---

## 📈 Results
- Achieved **high accuracy (~95%)** on validation dataset.
- Model effectively distinguishes between organic and recyclable waste.
- Training/Validation curves included in notebook.

---

## 🙌 Acknowledgements
- Dataset: [techsash / Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- Built with TensorFlow/Keras, OpenCV, Matplotlib, NumPy, and Pandas.

---
