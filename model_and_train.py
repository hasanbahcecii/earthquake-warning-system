import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# veri yolu

DATA_DIR = "data"

# egitim ve test verilerini yükleme
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")) # (800, 1500)
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")) # (200, 1500)  200 adet veri 1500 feature a sahip

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")) # (800,)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")) # (200,)  200 adet veri

# girişe uygun formata getir. Conv 1D için 3 boyut gerekli örnek, zaman ve kanal
X_train = X_train[..., np.newaxis] # (800, 1500, 1)
X_test = X_test[..., np.newaxis] # (200, 1500, 1)

# CNN modeli oluştur
model = tf.keras.Sequential()

import tensorflow as tf

# CNN modeli oluştur
model = tf.keras.Sequential()

# 1. conv katmanı
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation="relu", input_shape=(1500, 1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# 2. conv katmanı
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# 3. conv katmanı
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# flatten ve fully connected katmanlar
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))  # overfitting engellemek için
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # binary classification

# compile derleme
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# model özeti
model.summary()

# early stopping (validation kaybı iyileşmezse durdur)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# modeli eğit
history = model.fit(
    X_train, y_train,
    validation_data= (X_test, y_test),
    epochs = 3,
    batch_size= 32,
    callbacks = [early_stop]
)

# egitim geçmişi görselleştir
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "Eğitim Kaybı")
plt.plot(history.history["val_loss"], label = "Doğrulama Kaybı")
plt.title("Kayıp(loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Eğitim Doğruluk Değeri")
plt.plot(history.history["val_accuracy"], label = "Doğrulama Doğruluk Değeri")
plt.title("Doğruluk(accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# model save

model.save("cnn_seismic_model.h5")
print("Model başarılı bir şekile kaydedildi.")