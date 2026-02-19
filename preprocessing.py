import numpy as np
from sklearn.model_selection import train_test_split
import os

# veri yolunu belirle
DATA_DIR = "data"

# sinyal ve etiket verisi yükle
X = np.load(os.path.join(DATA_DIR, "X_signals.npy")) # boyut: (1000, 1500)
y = np.load(os.path.join(DATA_DIR, "y_labels.npy")) # boyut: (1000,)

# normalization (standardization - StandartScaler) (ortalaması = 0, std = 1 olacak şekilde dağılıma yerleştirme)
def z_score_normalize(signal_batch):
    """
    tüm sinyalleri ornek bazli normalize et
    yani her satır (1 sample a karşılık gelir) ayrı normalize edilir
    """

    normalized = []
    for signal in signal_batch:
        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:
            std = 1 # bölme hatası vermesin diye

        norm_signal = (signal - mean) / std  
        normalized.append(norm_signal)
    return np.array(normalized)      

# normalizasyon işlemi
X_normalized = z_score_normalize(X)

# eğitim ve test seti ayrımı
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify= y)

print(f"Egitim veri boyutu: {X_train.shape}")
print(f"Test veri boyutu: {X_test.shape}")

# mormalize edilmiş verinin kaydedilmesi
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
