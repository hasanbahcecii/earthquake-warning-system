# Earthquake Early Warning System

This project is a **Python-based machine learning system** designed to process seismic signals and provide early earthquake warnings. A Convolutional Neural Network (CNN) model is trained to classify seismic data for predictive alerts.

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

## ⚙️ Setup

1. Clone the repository:
```bash
   git clone git@github.com:hasanbahcecii/earthquake-warning-system.git
   cd earthquake-warning-system
```
    Create and activate a virtual environment:
```bash

    python3 -m venv venv
    source venv/bin/activate
```
    Install dependencies:
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

    Training and validation loss/accuracy are visualized with plots.

    The trained CNN model is saved as cnn_seismic_model.h5.

    Train/Test datasets are stored in the data/ directory.

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