# Modul 04: Stratified K-Fold Cross Validation

## Tujuan

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Memahami konsep dasar **cross-validation**.
2. Memahami perbedaan train-test split dan K-Fold Cross Validation.
3. Menggunakan **Stratified K-Fold Cross Validation** untuk klasifikasi.
4. Menghitung accuracy, precision, dan recall pada setiap fold.
5. Menginterpretasikan performa model berdasarkan hasil beberapa fold.

## Persiapan

Sebelum memulai praktikum, siapkan:

- Google Colab.
- Google Drive.
- Dataset `diabetes.csv`.
- Library `pandas`, `numpy`, dan `scikit-learn`.
- Pemahaman mengenai Random Forest dan Pipeline dari Modul 03.

Dataset dapat diunduh melalui tautan berikut:

[📥 Download diabetes.csv](data/diabetes.csv)

!!! tip "Persiapan Dataset"
    Simpan dataset pada folder `praktikum_ai` di Google Drive.

    Contoh lokasi:

    ```text
    MyDrive/praktikum_ai/diabetes.csv
    ```

!!! note "Prasyarat"
    Pada Modul 03 kita menggunakan satu pembagian **training data** dan **testing data**.

    Pada modul ini, kita akan menguji model beberapa kali menggunakan pembagian data yang berbeda melalui **Stratified K-Fold Cross Validation**.

---

## Langkah-langkah Praktikum

### 1. Membaca Dataset

Hubungkan Google Colab dengan Google Drive.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Import Pandas.

```python
import pandas as pd
```

Tentukan lokasi dataset.

```python
filepath = '/content/drive/MyDrive/praktikum_ai/diabetes.csv'
```

Baca dataset.

```python
data = pd.read_csv(filepath)

data.head()
```

Periksa ukuran dataset.

```python
data.shape
```

Periksa informasi setiap kolom.

```python
data.info()
```

---

### 2. Mengingat Kembali Training dan Testing

Pada Modul 03, dataset dibagi satu kali menjadi:

```text
Dataset
   ↓
Training Data
   +
Testing Data
```

Contohnya:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Masalahnya, hasil evaluasi dapat dipengaruhi oleh **data mana yang kebetulan masuk ke training dan testing**.

Cross-validation mencoba mengurangi ketergantungan terhadap satu pembagian data saja.

---

### 3. Mengenal K-Fold Cross Validation

Pada K-Fold Cross Validation, dataset dibagi menjadi beberapa bagian yang disebut **fold**.

Misalnya kita menggunakan:

```text
k = 5
```

maka dataset dibagi menjadi lima bagian:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Pada setiap percobaan, satu fold digunakan sebagai testing data dan fold lainnya digunakan sebagai training data.

```text
Putaran 1 : Test | Train | Train | Train | Train
Putaran 2 : Train | Test | Train | Train | Train
Putaran 3 : Train | Train | Test | Train | Train
Putaran 4 : Train | Train | Train | Test | Train
Putaran 5 : Train | Train | Train | Train | Test
```

Dengan demikian, setiap data memiliki kesempatan menjadi **testing data**.

!!! note "Hasil Akhir"
    Jika menggunakan 5-fold cross-validation, model akan dilatih dan diuji sebanyak **5 kali**.

    Nilai performa dari kelima percobaan tersebut kemudian dapat dirata-ratakan.

---

### 4. Mengapa Menggunakan Stratified K-Fold?

Periksa distribusi target.

```python
data['diabetes'].value_counts()
```

Lihat proporsinya.

```python
data['diabetes'].value_counts(normalize=True)
```

Pada dataset klasifikasi, jumlah data setiap kelas dapat berbeda.

Sebagai contoh:

```text
diabetes = 0 → lebih banyak
diabetes = 1 → lebih sedikit
```

**Stratified K-Fold** berusaha mempertahankan proporsi kelas pada setiap fold.

Secara sederhana:

```text
Dataset Awal
0 = 90%
1 = 10%

        ↓ Stratified K-Fold

Fold 1 → kira-kira 90% : 10%
Fold 2 → kira-kira 90% : 10%
Fold 3 → kira-kira 90% : 10%
...
```

!!! note "K-Fold vs Stratified K-Fold"
    **K-Fold** membagi data menjadi beberapa fold.

    **Stratified K-Fold** juga membagi data menjadi beberapa fold, tetapi berusaha mempertahankan proporsi kelas target pada masing-masing fold.

    Untuk masalah klasifikasi dengan kelas yang tidak seimbang, Stratified K-Fold biasanya lebih sesuai.

---

### 5. Menentukan Feature dan Target

Gunakan feature yang sama seperti Modul 03.

#### Feature numerik

```python
numerical_features = [
    'age',
    'hypertension',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]
```

#### Feature kategorikal

```python
categorical_features = [
    'gender',
    'smoking_history'
]
```

Buat feature `X`.

```python
X = data[numerical_features + categorical_features]
```

Buat target `y`.

```python
y = data['diabetes']
```

Periksa hasilnya.

```python
print("Ukuran X:", X.shape)
print("Ukuran y:", y.shape)
```

---

### 6. Membuat Preprocessing

Feature kategorikal perlu diubah menjadi bentuk numerik menggunakan `OneHotEncoder`.

Import library.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
```

Buat transformer untuk feature kategorikal.

```python
categorical_transformer = OneHotEncoder(
    handle_unknown='ignore'
)
```

Gabungkan preprocessing dengan `ColumnTransformer`.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

!!! note "Feature Numerik"
    Feature numerik menggunakan `passthrough` karena Random Forest tidak membutuhkan scaling.

---

### 7. Membuat Model Random Forest

Import Pipeline dan Random Forest.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
```

Buat model.

```python
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'classifier',
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)
```

Alurnya:

```text
Data
  ↓
Preprocessing
  ↓
One-Hot Encoding
  ↓
Random Forest
  ↓
Prediksi
```

---

### 8. Membuat Stratified K-Fold

Import `StratifiedKFold`.

```python
from sklearn.model_selection import StratifiedKFold
```

Gunakan **5-fold cross-validation**.

```python
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Parameter:

- `n_splits=5` → data dibagi menjadi 5 fold.
- `shuffle=True` → data diacak sebelum dibagi.
- `random_state=42` → hasil pengacakan dapat direproduksi.

---

### 9. Menyiapkan Evaluasi Model

Kita akan menghitung tiga metric:

- **Accuracy**
- **Precision**
- **Recall**

Import metric yang diperlukan.

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
```

Siapkan list untuk menyimpan hasil setiap fold.

```python
accuracies = []
precisions = []
recalls = []
confusion_matrices = []
```

---

### 10. Training dan Testing pada Setiap Fold

Gunakan perulangan untuk menjalankan setiap fold.

```python
for fold, (train_index, test_index) in enumerate(
    skf.split(X, y),
    start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    confusion_matrices.append(cm)

    print(
        f"Fold {fold} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )
```

Setiap iterasi melakukan:

```text
Pilih Training & Testing Fold
          ↓
       model.fit()
          ↓
      model.predict()
          ↓
     Hitung Metrics
          ↓
     Simpan Hasil
```

!!! question "Coba Amati"
    Apakah nilai accuracy, precision, dan recall pada setiap fold sama?

    Jika berbeda, mengapa hal tersebut dapat terjadi?

---

### 11. Menghitung Rata-rata Performa

Import NumPy.

```python
import numpy as np
```

Hitung rata-rata setiap metric.

```python
print(
    f"Average Accuracy  : {np.mean(accuracies):.4f}"
)

print(
    f"Average Precision : {np.mean(precisions):.4f}"
)

print(
    f"Average Recall    : {np.mean(recalls):.4f}"
)
```

Kita juga dapat menghitung standar deviasi accuracy.

```python
print(
    f"Std. Accuracy     : {np.std(accuracies):.4f}"
)
```

!!! note "Mengapa Standar Deviasi?"
    Rata-rata menunjukkan performa model secara umum.

    Standar deviasi menunjukkan seberapa besar hasil antar-fold berubah.

    Nilai standar deviasi yang kecil menunjukkan hasil antar-fold relatif konsisten.

---

### 12. Menampilkan Hasil Setiap Fold

Agar lebih mudah dibaca, buat DataFrame.

```python
hasil_cv = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls
})

hasil_cv
```

Tambahkan baris rata-rata jika diperlukan.

```python
hasil_cv.mean(numeric_only=True)
```

!!! question "Coba Analisis"
    Fold mana yang menghasilkan accuracy tertinggi?

    Apakah fold tersebut juga memiliki precision dan recall tertinggi?

---

### 13. Mengenal Confusion Matrix

Confusion matrix menunjukkan hasil klasifikasi secara lebih rinci.

Untuk klasifikasi diabetes:

| | Prediksi 0 | Prediksi 1 |
|---|---:|---:|
| **Aktual 0** | True Negative | False Positive |
| **Aktual 1** | False Negative | True Positive |

Secara sederhana:

- **True Positive (TP)** → diabetes diprediksi diabetes.
- **True Negative (TN)** → tidak diabetes diprediksi tidak diabetes.
- **False Positive (FP)** → tidak diabetes diprediksi diabetes.
- **False Negative (FN)** → diabetes diprediksi tidak diabetes.

!!! warning "Perhatikan False Negative"
    Dalam konteks screening, False Negative berarti data dengan kelas positif diprediksi sebagai negatif.

    Karena itu, accuracy saja tidak selalu cukup untuk memahami performa model.

---

### 14. Menampilkan Confusion Matrix Setiap Fold

Gunakan `ConfusionMatrixDisplay`.

```python
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
```

Tampilkan confusion matrix dari setiap fold.

```python
for i, cm in enumerate(confusion_matrices):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )

    disp.plot()

    plt.title(
        f'Confusion Matrix - Fold {i+1}'
    )

    plt.show()
```

Perhatikan perbedaan confusion matrix antar-fold.

!!! question "Coba Amati"
    Pada fold mana jumlah False Negative paling banyak?

---

### 15. Membandingkan Cross-Validation dengan Train-Test Split

Pada train-test split:

```text
Dataset
  ↓
1x Training
+
1x Testing
```

Sedangkan pada 5-fold cross-validation:

```text
Dataset
  ↓
5 pembagian berbeda
  ↓
5x Training
+
5x Testing
  ↓
Rata-rata Performa
```

| Metode | Jumlah Evaluasi | Hasil |
|---|---:|---|
| Train-Test Split | 1 | Bergantung pada satu pembagian |
| 5-Fold CV | 5 | Rata-rata dari 5 pembagian |

!!! note "Bukan Berarti Cross-Validation Selalu Lebih Tinggi"
    Tujuan cross-validation bukan untuk membuat accuracy menjadi lebih tinggi.

    Cross-validation digunakan untuk memperoleh **estimasi performa yang lebih stabil** dari beberapa pembagian data.

---

## Ringkasan

Pada praktikum ini kita telah mempelajari:

```text
Dataset
   ↓
Feature (X) + Target (y)
   ↓
Stratified K-Fold
   ↓
┌──────────────┐
│ Fold 1       │
│ Fold 2       │
│ Fold 3       │
│ Fold 4       │
│ Fold 5       │
└──────────────┘
   ↓
Training + Testing
pada setiap fold
   ↓
Accuracy
Precision
Recall
Confusion Matrix
   ↓
Rata-rata Performa
```

Konsep utamanya:

> **Cross-validation mengevaluasi model menggunakan beberapa pembagian data, bukan hanya satu pembagian training dan testing.**

---

# Tugas Praktikum

Kerjakan menggunakan Google Colab dan dataset `diabetes.csv`.

## 1. Membandingkan Jumlah Fold

Jalankan Stratified K-Fold menggunakan:

- `n_splits = 3`
- `n_splits = 5`
- `n_splits = 10`

Catat rata-rata accuracy.

| Jumlah Fold | Average Accuracy |
|---:|---:|
| 3 | ... |
| 5 | ... |
| 10 | ... |

Manakah yang menghasilkan average accuracy tertinggi?

---

## 2. Membandingkan Precision dan Recall

Untuk eksperimen:

```text
n_splits = 5
```

catat:

- average accuracy,
- average precision,
- average recall.

| Metric | Nilai |
|---|---:|
| Accuracy | ... |
| Precision | ... |
| Recall | ... |

Metric mana yang memiliki nilai paling rendah?

---

## 3. Mengamati Variasi Antar-Fold

Gunakan hasil 5-fold cross-validation.

Catat accuracy setiap fold.

| Fold | Accuracy |
|---:|---:|
| 1 | ... |
| 2 | ... |
| 3 | ... |
| 4 | ... |
| 5 | ... |

Tentukan:

- fold dengan accuracy tertinggi,
- fold dengan accuracy terendah.

---

## 4. Confusion Matrix

Pilih satu confusion matrix dari hasil 5-fold.

Catat nilai:

- True Positive,
- True Negative,
- False Positive,
- False Negative.

Tuliskan satu kesimpulan berdasarkan confusion matrix tersebut.

---

## 5. Kesimpulan

Jelaskan dalam maksimal **3 kalimat**:

1. Mengapa cross-validation digunakan?
2. Apa fungsi `StratifiedKFold`?
3. Mengapa performa model sebaiknya tidak hanya dinilai dari accuracy?

---

## Pengumpulan

Simpan notebook dalam format:

```text
.ipynb
```

Gunakan nama file:

```text
NIM_Nama_Modul04.ipynb
```

Pastikan seluruh cell sudah dijalankan dan output terlihat sebelum notebook dikumpulkan.
