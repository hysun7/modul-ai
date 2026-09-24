# Modul 03: Random Forest dengan Data Numerik dan Kategorikal

## Tujuan

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Membedakan data numerik dan data kategorikal.
2. Menyiapkan data numerik dan kategorikal untuk Machine Learning.
3. Mengubah data kategorikal menjadi bentuk numerik menggunakan `OneHotEncoder`.
4. Menggabungkan proses preprocessing dan Random Forest menggunakan `Pipeline`.
5. Melatih dan mengevaluasi model Machine Learning menggunakan data numerik dan kategorikal.

## Persiapan

Sebelum memulai praktikum, siapkan:

- Google Colab.
- Google Drive.
- Dataset `diabetes.csv`.
- Library `pandas` dan `scikit-learn`.
- Pemahaman dasar mengenai Random Forest dari Modul 02.

Dataset dapat diunduh melalui tautan berikut:

[📥 Download diabetes.csv](data/diabetes.csv)

!!! tip "Persiapan Dataset"
    Download dataset kemudian simpan pada folder `praktikum_ai`
    di Google Drive.

    Contoh lokasi file:

    ```text
    MyDrive/praktikum_ai/diabetes.csv
    ```

!!! note "Prasyarat"
    Pada Modul 02 kita hanya menggunakan beberapa fitur numerik.

    Pada modul ini kita akan menggunakan **data numerik dan kategorikal secara bersamaan** untuk membuat model Machine Learning.

---

## Langkah-langkah Praktikum

### 1. Menghubungkan Google Drive

Hubungkan Google Colab dengan Google Drive.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Setelah berhasil terhubung, file pada Google Drive dapat diakses melalui:

```text
/content/drive/MyDrive/
```

---

### 2. Membaca Dataset

Import library Pandas.

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

---

### 3. Melihat Informasi Dataset

Periksa struktur dataset.

```python
data.info()
```

Dataset memiliki beberapa kolom:

```text
gender
age
hypertension
heart_disease
smoking_history
bmi
HbA1c_level
blood_glucose_level
diabetes
```

Periksa juga beberapa data pertama.

```python
data.head()
```

!!! question "Coba Amati"
    Berdasarkan hasil `data.info()`, kolom mana yang memiliki tipe data:

    - `int64`
    - `float64`
    - `object`

---

### 4. Mengenali Data Numerik dan Kategorikal

Data yang digunakan dalam Machine Learning dapat memiliki tipe yang berbeda.

#### Data Numerik

Data numerik merupakan data berupa angka yang memiliki nilai kuantitatif.

Contoh:

| Feature | Contoh |
|---|---:|
| `age` | 45 |
| `bmi` | 27.5 |
| `HbA1c_level` | 6.1 |
| `blood_glucose_level` | 140 |

Contoh kode:

```python
data[
    ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
].head()
```

---

#### Data Kategorikal

Data kategorikal menunjukkan suatu kelompok atau kategori.

Contoh:

| Feature | Contoh |
|---|---|
| `gender` | Male |
| `smoking_history` | never |
| `smoking_history` | former |

Contoh:

```python
data[['gender', 'smoking_history']].head()
```

Kita dapat melihat kategori yang terdapat pada sebuah kolom menggunakan:

```python
data['gender'].value_counts()
```

dan:

```python
data['smoking_history'].value_counts()
```

!!! note "Bagaimana dengan hypertension?"
    Kolom `hypertension` memiliki nilai `0` dan `1`.

    Walaupun secara konsep menunjukkan kategori **tidak / ya**, nilainya sudah berbentuk angka sehingga pada praktikum ini dapat digunakan langsung sebagai feature tanpa encoding tambahan.

---

### 5. Menentukan Feature dan Target

Pada praktikum ini kita akan memprediksi:

```text
diabetes
```

sebagai **target**.

Feature numerik yang digunakan:

```python
numerical_features = [
    'age',
    'hypertension',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]
```

Feature kategorikal yang digunakan:

```python
categorical_features = [
    'gender',
    'smoking_history'
]
```

Gabungkan feature tersebut ke dalam `X`.

```python
X = data[numerical_features + categorical_features]
```

Tentukan target `y`.

```python
y = data['diabetes']
```

Periksa hasilnya.

```python
X.head()
```

```python
y.head()
```

Secara sederhana:

```text
Numerical Features
age
hypertension
bmi
HbA1c_level
blood_glucose_level
            +
Categorical Features
gender
smoking_history
            ↓
            X

diabetes
    ↓
    y
```

---

### 6. Mengapa Data Kategorikal Perlu Diubah?

Perhatikan nilai pada kolom:

```python
data['gender'].head()
```

Output dapat berupa:

```text
Female
Female
Male
Female
Male
```

Komputer tidak dapat langsung menggunakan teks seperti `Male` dan `Female` dalam sebagian besar algoritma Machine Learning.

Karena itu, data kategorikal perlu diubah menjadi data numerik.

Salah satu metode yang dapat digunakan adalah **One-Hot Encoding**.

Sebagai contoh:

```text
gender
------
Male
Female
```

dapat diubah menjadi:

```text
gender_Female    gender_Male
      0               1
      1               0
```

---

### 7. Membuat Transformer Data Kategorikal

Import `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder
```

Buat transformer.

```python
categorical_transformer = OneHotEncoder(
    handle_unknown='ignore'
)
```

Parameter:

```text
handle_unknown='ignore'
```

digunakan agar model tetap dapat melakukan prediksi jika menemukan kategori yang tidak terdapat pada data training.

---

### 8. Membuat Preprocessor dengan ColumnTransformer

Kita memiliki dua jenis feature:

```text
Numerik      → digunakan langsung
Kategorikal  → OneHotEncoder
```

Gunakan `ColumnTransformer` untuk menentukan preprocessing yang berbeda untuk setiap kelompok feature.

```python
from sklearn.compose import ColumnTransformer
```

Buat preprocessor.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

`passthrough` berarti feature numerik diteruskan tanpa transformasi.

!!! note "Mengapa Tidak Menggunakan StandardScaler?"
    Random Forest merupakan model berbasis decision tree sehingga tidak membutuhkan penyamaan skala data.

    Misalnya `age` memiliki nilai puluhan sedangkan `HbA1c_level` memiliki nilai sekitar satuan, perbedaan skala tersebut tidak menjadi masalah utama bagi Random Forest.

    Model seperti **KNN** dan **SVM** memiliki karakteristik berbeda dan dapat membutuhkan scaling.

---

### 9. Membagi Data Training dan Testing

Import `train_test_split`.

```python
from sklearn.model_selection import train_test_split
```

Bagi data menjadi:

- 80% data training
- 20% data testing

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Periksa ukuran data.

```python
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
```

!!! note "Stratify"
    `stratify=y` digunakan agar proporsi kelas `diabetes = 0` dan `diabetes = 1` pada data training dan testing tetap menyerupai dataset awal.

---

### 10. Membuat Pipeline

Kita akan menggabungkan:

```text
Preprocessing
     ↓
Random Forest
```

menggunakan `Pipeline`.

Import library yang diperlukan.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
```

Buat pipeline.

```python
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier',
         RandomForestClassifier(
             n_estimators=100,
             random_state=42
         ))
    ]
)
```

Dengan Pipeline, proses berikut dilakukan secara otomatis:

```text
Input Data
    ↓
Numerical ──────────────┐
                        │
Categorical             │
    ↓                   │
OneHotEncoder           │
    ↓                   │
    └──────────┬────────┘
               ↓
        Random Forest
               ↓
           Prediksi
```

---

### 11. Melatih Model

Latih model menggunakan data training.

```python
model.fit(X_train, y_train)
```

Pada tahap ini Pipeline akan:

1. mengambil data numerik,
2. melakukan encoding terhadap data kategorikal,
3. menggabungkan seluruh feature,
4. melatih Random Forest.

---

### 12. Melakukan Prediksi

Gunakan data testing untuk melakukan prediksi.

```python
y_pred = model.predict(X_test)
```

Tampilkan beberapa hasil prediksi.

```python
y_pred[:10]
```

Bandingkan dengan nilai sebenarnya.

```python
y_test.head(10)
```

---

### 13. Menghitung Accuracy

Import `accuracy_score`.

```python
from sklearn.metrics import accuracy_score
```

Hitung accuracy.

```python
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.4f}")
```

Accuracy menunjukkan proporsi data testing yang berhasil diprediksi dengan benar oleh model.

!!! warning "Perhatikan Distribusi Kelas"
    Dataset diabetes memiliki jumlah kelas yang tidak selalu seimbang.

    Oleh karena itu, accuracy tidak boleh menjadi satu-satunya ukuran untuk menilai kualitas model.

    Evaluasi model yang lebih lengkap akan dibahas pada modul berikutnya.

---

### 14. Membandingkan dengan Model Numerik Saja

Pada Modul 02 kita membuat model menggunakan feature numerik.

Sekarang model menggunakan:

```text
Numerik
+
Kategorikal
```

Perhatikan nilai accuracy yang diperoleh.

!!! question "Coba Analisis"
    Apakah penambahan feature `gender` dan `smoking_history` membuat accuracy model berubah?

    Apakah penambahan jumlah feature selalu menjamin model menjadi lebih baik?

---

## Ringkasan

Pada praktikum ini kita melakukan proses:

```text
Dataset
   ↓
Pisahkan Feature dan Target
   ↓
┌────────────────────────────┐
│ Feature Numerik            │
│ → digunakan langsung       │
│                            │
│ Feature Kategorikal        │
│ → OneHotEncoder            │
└────────────────────────────┘
             ↓
      ColumnTransformer
             ↓
          Pipeline
             ↓
      Random Forest
             ↓
          Prediksi
             ↓
          Accuracy
```

Kode utama dapat diringkas sebagai berikut:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

numerical_features = [
    'age',
    'hypertension',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level'
]

categorical_features = [
    'gender',
    'smoking_history'
]

X = data[numerical_features + categorical_features]
y = data['diabetes']

categorical_transformer = OneHotEncoder(
    handle_unknown='ignore'
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier',
         RandomForestClassifier(
             n_estimators=100,
             random_state=42
         ))
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.4f}")
```

---

# Tugas Rumah

Gunakan dataset `diabetes.csv`.

## 1. Model A

Buat model untuk memprediksi:

```text
heart_disease
```

dengan feature:

```text
age
bmi
gender
hypertension
smoking_history
```

Tampilkan nilai accuracy model.

---

## 2. Model B

Buat model untuk memprediksi:

```text
heart_disease
```

dengan feature:

```text
age
HbA1c_level
gender
hypertension
smoking_history
```

Tampilkan nilai accuracy model.

---

## 3. Model C

Buat model untuk memprediksi:

```text
heart_disease
```

dengan feature:

```text
age
blood_glucose_level
gender
hypertension
smoking_history
```

Tampilkan nilai accuracy model.

---

## 4. Bandingkan Model

Buat tabel hasil:

| Model | Feature Numerik Utama | Accuracy |
|---|---|---:|
| A | `bmi` | ... |
| B | `HbA1c_level` | ... |
| C | `blood_glucose_level` | ... |

Bandingkan nilai accuracy ketiga model.

Apakah terdapat perbedaan hasil antara Model A, B, dan C?

---

## 5. Kesimpulan

Tuliskan kesimpulan dalam **maksimal 3 kalimat** mengenai:

- model yang menghasilkan accuracy tertinggi,
- pengaruh pemilihan feature terhadap hasil model,
- manfaat menggunakan Pipeline ketika dataset memiliki data numerik dan kategorikal.

---

## Pengumpulan

Simpan notebook pada Google Drive dan kumpulkan dalam format:

```text
.ipynb
```

Gunakan format nama file:

```text
NIM_Nama_Modul03.ipynb
```

Pastikan seluruh cell telah dijalankan dan output dapat terlihat sebelum notebook dikumpulkan.
