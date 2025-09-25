SmartWaste: AI-Based Garbage Classification

This project uses Deep Learning (CNN) to classify waste images into two categories:

Organic

Recyclable

The model is trained on the Waste Classification Data
 dataset from Kaggle.
 📂 Project Structure
 SmartWaste-AI-Based-Garbage-Classification/
│
├── README.md                # Project description (this file)
├── requirements.txt         # Required dependencies
├── notebooks/               # Jupyter/Colab notebooks
│   └── SmartWaste.ipynb     # Full training & testing workflow
│
├── src/                     # Source code
│   ├── train.py             # Training script for CNN
│   ├── predict.py           # Prediction script for new images
│   └── utils.py             # Helper functions
│
├── models/                  # Trained models
│   └── waste_classifier_cnn.h5
│
├── test_images/             # Sample images for testing
│   ├── organic_sample.jpg
│   └── recyclable_sample.jpg
│
└── docs/                    # Documentation (optional)
    └── architecture.png
🚀 How to Run
1. Clone the repository
git clone https://github.com/Siddarth1409/SmartWaste-AI-Based-Garbage-Classification.git
cd SmartWaste-AI-Based-Garbage-Classification

2. Install dependencies
pip install -r requirements.txt

3. Download dataset

The dataset is not included in the repo. Download from Kaggle using:

import kagglehub
path = kagglehub.dataset_download("techsash/waste-classification-data")


It will create the following structure:

waste_classification_data/
    ├── DATASET/
    │   ├── TRAIN/
    │   │   ├── ORGANIC/
    │   │   └── RECYCLE/
    │   └── TEST/
    │       ├── ORGANIC/
    │       └── RECYCLE/

4. Train the model
python src/train.py

5. Test the model with a new image
python src/predict.py --image test_images/organic_sample.jpg

📊 Results

CNN trained on 224×224 RGB images.

Achieved ~85–90% accuracy on validation set.

Works in Google Colab or local Python environment.

📌 Future Improvements

Extend to multi-class classification (e.g., plastic, paper, glass).

Deploy as a web app or mobile app for real-world use.

Optimize model size for edge devices.

📝 Author

Developed by Siddarth Loni
