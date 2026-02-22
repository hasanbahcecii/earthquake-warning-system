"""
Earthquake early warning system with 1D CNN
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import os

os.makedirs('data', exist_ok= True) # create data folder

# key parameters
SAMPLES = 1500 # each signal consists of 1500 samples
NUM_SIGNALS = 1000 # total signal number (500 earthquake, 500 noise)

# earthquake signal generator function
def generate_earthquake_signal():
    """
    -gaussian zarf ile modüle edilmiş bir sinüs sinyali
    -üzerine az miktarda gürültü ekle
    """

    t = np.linspace(0, 1, SAMPLES) # 0 - 1 saniye arası zaman vektörü
    freq = np.random.uniform(5, 15) # frekans 5, 15 Hz arası rastgele seçilir
    envelope = np.exp(-((t - 0.5)**2) / 0.01) # gaussian zarf fonksiyonu
    signal = envelope * np.sin(2*np.pi*freq*t)
    noise = np.random.normal(0, 0.1, SAMPLES) # kücük bir gürültü eklenir
    return signal + noise

#dummy = generate_earthquake_signal()

# noise signal generator function
def generate_noise_signal():
    """
    deprem dışı sinyal oluşturur (gürültü)
        -düşük frekans sinüs dalgaları
        -üzerine fazla miktarda beyaz gürültü ekle
    """

    t = np.linspace(0, 1, SAMPLES) # zaman vektörü oluştur
    base = np.sin(2*np.pi*np.random.uniform(0.1, 1) *t) # 0.1 ile 1 arası düşük frekanslı sinyal
    noise = np.random.normal(0, 0.3, SAMPLES) # samples kadar gürültü oluştur  
    return base + noise

#dummy = generate_noise_signal()

# boş veri listeleri
X = [] # signals
y = [] # labels (1 = deprem, 0 = gürültü)

# 500 adet deprem sinyali üret
for _ in range(NUM_SIGNALS // 2):
    X.append(generate_earthquake_signal())
    y.append(1) # deprem etiketi = 1 


# 500 adet gürültü sinyali üret
for _ in range(NUM_SIGNALS // 2):
    X.append(generate_noise_signal())
    y.append(0) # gürültü etiketi = 0     


# listeyi np array e cevir
X = np.array(X)
y = np.array(y)

# verileri karşılaştırmak için index oluştur
indices = np.arange(len(X)) # 0 - 999 indeksler
np.random.shuffle(indices) # indexleri karıştır
X = X[indices] # sinyalleri yeni sıraya göre düzenle
y = y[indices] # etiketleri yeni sıraya göre düzenle

# veriyi .np formatında kaydet
np.save("data/X_signals.npy", X)
np.save("data/y_labels.npy", y)

# ilk 3 sinyali görselleştir
for i in range(3):
    plt.plot(X[i], label = f"Label: {y[i]}") # sinyali cizdirir

plt.title("Örnek Sentetik Sinyaller")
plt.legend()
plt.show()