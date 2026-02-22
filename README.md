# Earthquake Early Warning System

This project is a **Python-based machine learning system** designed to process seismic signals and provide early earthquake warnings. A Convolutional Neural Network (CNN) model is trained to classify seismic data for predictive alerts.

---

## 📌 Problem Definition

- Detect P-waves in seismic signals before destructive S-waves arrive.

- Train a 1D CNN on labeled seismic waveform datasets to learn P-wave patterns.

- Use the trained model to predict P-waves in real time, enabling rapid alerts.

- Aim: Capture short-duration, high-frequency P-wave signatures for timely warning.

---

## 📂 Project Structure
```
earthquake-warning-system/
│
├── preprocessing.py        # Data preprocessing and train/test split
├── load_and_visualize.py   # Data loading and visualization
├── model_and_train.py      # CNN model creation and training
├── test.py                 # Model testing and evaluation
├── requirements.txt        # Dependency list
└── .gitignore              # Excludes unnecessary files
```

---

## 📊 Dataset Specification

Type: Synthetic time series dataset

Signal duration: 15 seconds

Sampling rate: 100 Hz → 1500 samples per signal

Classes:

- Class 1: Earthquake signal (contains P-wave onset and seismic activity)

- Class 0: Non-earthquake signal (background noise or normal vibration)

---

## ⚙️ Setup

1. Clone the repository:
```bash
   git clone git@github.com:hasanbahcecii/earthquake-warning-system.git
   cd earthquake-warning-system
```
2. Create and activate a virtual environment:
```bash

    python3 -m venv venv
    source venv/bin/activate
```
3. Install dependencies:
```bash

    pip install -r requirements.txt
```

---

## 🚀 Usage

Data Preprocessing
```bash
python preprocessing.py
```

2. Model Training
```bash

python model_and_train.py
```
3. Model Testing
```bash

python test.py
```

---

## 📊 Outputs

- Training and validation loss/accuracy are visualized with plots.

- The trained CNN model is saved as cnn_seismic_model.h5.

- Train/Test datasets are stored in the data/ directory.

---

## 🛠️ Technologies

    Python 3.12

    NumPy

    scikit-learn

    TensorFlow / Keras

    Matplotlib

---

## 📜 License

MIT