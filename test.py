import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns


# veri yolu ve model dosyası
DATA_DIR = "data"
MODEL_PATH = "cnn_seismic_model.h5"

# test verisini yükle
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# conv 1d için giriş şeklini değiştir
X_test = X_test[..., np.newaxis] # (200, 1500, 1)

# egitilmiş modeli yükle
model = tf.keras.models.load_model(MODEL_PATH)
print("Model başarılı bir şekilde yüklendi.")


# test verisi üzerinde tahmin
y_pred_prob = model.predict(X_test) # olasılıksal değer return eder
y_pred = (y_pred_prob > 0.5).astype(int) # 0.5 ten küçükler için false yani int olarak 0, büyükler için ise true yani int olaran 1 return et

# sınıflandırma metricleri hesapla
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1: {f1}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot= True, fmt= "d", cmap= "Blues", xticklabels=["Gürültü", "Deprem"], yticklabels=["Gürültü", "Deprem"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Etiket")
plt.title("Confusion Matrix")
plt.show()