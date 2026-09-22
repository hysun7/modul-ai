# Modul 02: Membuat Model Machine Learning Menggunakan Random Forest

## Tujuan

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Memahami konsep dasar **feature**, **target**, data training, dan data testing.
2. Menyiapkan dataset untuk proses Machine Learning.
3. Membuat dan melatih model klasifikasi menggunakan **Random Forest**.
4. Menggunakan model untuk melakukan prediksi.
5. Mengevaluasi hasil prediksi model menggunakan accuracy.

## Persiapan

Sebelum memulai praktikum, siapkan:

- Google Colab.
- Google Drive.
- Dataset `diabetes.csv`.
- Library `pandas` dan `scikit-learn`.

Dataset dapat diunduh melalui tautan berikut:

[📥 Download diabetes.csv](data/diabetes_prediction_dataset.csv)

!!! tip "Persiapan Dataset"
    Download dataset kemudian simpan pada folder `praktikum_ai`
    di Google Drive.

    Contoh lokasi file:

    ```text
    MyDrive/praktikum_ai/diabetes.csv
    ```

!!! note "Prasyarat"
    Praktikum ini melanjutkan Modul 01. Pastikan Anda sudah memahami cara membaca dataset, memilih kolom, dan menggunakan DataFrame.

---

## Langkah-langkah Praktikum

### 1. Menghubungkan Google Drive

Hubungkan Google Colab dengan Google Drive.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Setelah Google Drive terhubung, dataset dapat diakses melalui folder:

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

Baca dataset menggunakan `pd.read_csv()`.

```python
data = pd.read_csv(filepath)

data.head()
```

---

### 3. Mengenali Dataset

Periksa ukuran dataset.

```python
data.shape
```

Kemudian lihat informasi setiap kolom.

```python
data.info()
```

Dataset memiliki beberapa atribut, antara lain:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`
- `diabetes`

Pada praktikum ini, kolom:

```text
diabetes
```

akan menjadi **target** yang ingin diprediksi.

!!! question "Coba Amati"
    Dari hasil `data.info()`, kolom mana yang memiliki tipe data numerik dan kolom mana yang memiliki tipe data kategorikal?

---

### 4. Memilih Feature Numerik

Untuk praktikum pertama Machine Learning, kita akan menggunakan empat feature numerik:

- `age`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`

Sedangkan target yang akan diprediksi adalah:

- `diabetes`

Buat dataset yang hanya berisi kolom tersebut.

```python
kolom = [
    'age',
    'bmi',
    'HbA1c_level',
    'blood_glucose_level',
    'diabetes'
]

dataset = data[kolom]

dataset.head()
```

Periksa kembali informasi dataset.

```python
dataset.info()
```

!!! note "Mengapa hanya data numerik?"
    Random Forest dapat menggunakan data numerik secara langsung.

    Pada modul berikutnya kita akan mempelajari bagaimana menggunakan feature numerik dan kategorikal secara bersamaan.

---

### 5. Memisahkan Feature dan Target

Dalam Machine Learning, data biasanya dipisahkan menjadi:

```text
Feature (X) → data yang digunakan untuk melakukan prediksi

Target (y)  → data yang ingin diprediksi
```

Pada dataset ini:

```text
age
bmi
HbA1c_level
blood_glucose_level
        ↓
        X

diabetes
        ↓
        y
```

Pisahkan feature dan target.

```python
X = dataset.drop('diabetes', axis=1)

y = dataset['diabetes']
```

Periksa data feature.

```python
X.head()
```

Periksa target.

```python
y.head()
```

---

### 6. Melihat Distribusi Target

Sebelum membuat model, lihat jumlah data pada setiap kelas.

```python
y.value_counts()
```

Untuk melihat dalam bentuk proporsi:

```python
y.value_counts(normalize=True)
```

!!! question "Coba Amati"
    Apakah jumlah data `diabetes = 0` dan `diabetes = 1` seimbang?

!!! note "Mengapa Ini Penting?"
    Jika satu kelas jauh lebih banyak dibandingkan kelas lainnya, nilai accuracy perlu dibaca dengan hati-hati.

    Model yang selalu memilih kelas mayoritas sekalipun dapat memperoleh accuracy yang terlihat tinggi.

---

### 7. Membagi Data Training dan Testing

Dataset akan dibagi menjadi:

- **80% training data** untuk melatih model.
- **20% testing data** untuk menguji model.

Import `train_test_split`.

```python
from sklearn.model_selection import train_test_split
```

Bagi dataset.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Periksa ukuran masing-masing data.

```python
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
```

!!! note "Parameter"
    `test_size=0.2` berarti 20% data digunakan sebagai testing data.

    `random_state=42` digunakan agar pembagian data dapat direproduksi.

    `stratify=y` digunakan agar proporsi kelas pada training dan testing tetap menyerupai dataset awal.

---

### 8. Membuat Model Random Forest

Import `RandomForestClassifier`.

```python
from sklearn.ensemble import RandomForestClassifier
```

Buat model.

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

`n_estimators=100` berarti Random Forest menggunakan 100 decision tree.

Secara sederhana:

```text
Data Training
     ↓
┌─────────┐
│ Tree 1  │
├─────────┤
│ Tree 2  │
├─────────┤
│ Tree 3  │
├─────────┤
│   ...   │
├─────────┤
│ Tree 100│
└─────────┘
     ↓
Menggabungkan hasil
     ↓
Prediksi akhir
```

---

### 9. Melatih Model

Gunakan fungsi `fit()` untuk melatih model.

```python
model.fit(X_train, y_train)
```

Pada proses ini, model mempelajari hubungan antara:

```text
age
bmi
HbA1c_level
blood_glucose_level
```

dengan:

```text
diabetes
```

---

### 10. Menguji Model dengan Satu Data

Ambil satu data dari `X_test`.

```python
data_uji = X_test.iloc[[0]]

data_uji
```

Perhatikan penggunaan:

```python
iloc[[0]]
```

agar data tetap dalam bentuk DataFrame.

Lakukan prediksi.

```python
prediksi = model.predict(data_uji)

print(prediksi)
```

Hasil:

```text
0
```

atau:

```text
1
```

sesuai dengan prediksi model.

---

### 11. Membandingkan Prediksi dan Data Sebenarnya

Lihat hasil prediksi.

```python
prediksi = model.predict(data_uji)[0]

print("Prediksi :", prediksi)
```

Kemudian lihat label sebenarnya.

```python
aktual = y_test.iloc[0]

print("Aktual   :", aktual)
```

Tampilkan keduanya.

```python
print("Prediksi :", prediksi)
print("Aktual   :", aktual)
```

!!! question "Coba Amati"
    Apakah prediksi model sama dengan nilai sebenarnya?

---

### 12. Melakukan Prediksi pada Seluruh Data Testing

Prediksi satu data belum cukup untuk mengetahui performa model.

Gunakan seluruh `X_test`.

```python
y_pred = model.predict(X_test)
```

Variabel `y_pred` sekarang berisi prediksi model untuk seluruh data testing.

Coba tampilkan beberapa hasil pertama.

```python
y_pred[:10]
```

Bandingkan dengan data sebenarnya.

```python
y_test.head(10)
```

---

### 13. Menghitung Accuracy

Import fungsi `accuracy_score`.

```python
from sklearn.metrics import accuracy_score
```

Hitung accuracy.

```python
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy model = {accuracy:.2f}")
```

Accuracy menunjukkan proporsi prediksi yang benar dibandingkan dengan seluruh data testing.

Sebagai contoh:

```text
Accuracy model = 0.97
```

berarti sekitar 97% data testing berhasil diklasifikasikan dengan benar.

!!! warning "Accuracy Bukan Satu-satunya Ukuran"
    Accuracy yang tinggi belum tentu berarti model sudah baik untuk semua kelas, terutama apabila jumlah data setiap kelas tidak seimbang.

    Evaluasi model akan dipelajari lebih lanjut pada praktikum berikutnya.

---

### 14. Membandingkan dengan Baseline

Sebagai pembanding sederhana, hitung proporsi kelas terbesar pada data testing.

```python
baseline = y_test.value_counts(normalize=True).max()

print(f"Baseline accuracy = {baseline:.2f}")
print(f"Random Forest     = {accuracy:.2f}")
```

Model seharusnya memberikan hasil yang lebih baik dibandingkan hanya selalu memilih kelas yang paling banyak.

!!! question "Coba Analisis"
    Apakah accuracy Random Forest lebih tinggi dibandingkan baseline?

---

## Ringkasan

Pada praktikum ini kita telah melakukan proses dasar Machine Learning:

```text
Membaca Dataset
       ↓
Memilih Feature
       ↓
Menentukan Target
       ↓
Train-Test Split
       ↓
Membuat Random Forest
       ↓
model.fit()
       ↓
model.predict()
       ↓
Evaluasi Accuracy
```

Kode utama untuk membuat model Machine Learning dapat diringkas sebagai berikut:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = dataset.drop('diabetes', axis=1)
y = dataset['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.2f}")
```

---

# Tugas Rumah

Kerjakan tugas berikut secara mandiri menggunakan Google Colab.

## 1. Mengubah Data Testing

Ubah pembagian dataset menjadi:

```text
70% training
30% testing
```

Latih kembali model Random Forest dan catat nilai accuracy yang diperoleh.

Bandingkan dengan pembagian 80:20 yang digunakan pada praktikum.

---

## 2. Mengubah Jumlah Decision Tree

Coba buat Random Forest dengan:

```python
n_estimators=50
```

dan:

```python
n_estimators=200
```

Catat accuracy dari masing-masing model.

Buat tabel sederhana:

| n_estimators | Accuracy |
|-------------:|---------:|
| 50 | ... |
| 100 | ... |
| 200 | ... |

Tuliskan satu kesimpulan berdasarkan hasil percobaan.

---

## 3. Menguji Data Secara Manual

Ambil satu data dari `X_test` selain data pertama.

Tampilkan:

- data feature,
- hasil prediksi model,
- label sebenarnya.

Apakah prediksi model benar?

---

## 4. Feature Importance

Random Forest dapat memberikan informasi mengenai feature yang paling banyak digunakan dalam proses prediksi.

Jalankan:

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance.sort_values(
    by='Importance',
    ascending=False
)
```

Jawab pertanyaan:

1. Feature mana yang memiliki nilai importance tertinggi?
2. Feature mana yang memiliki nilai importance terendah?
3. Apa yang dapat Anda simpulkan dari hasil tersebut?

!!! warning "Interpretasi"
    Feature importance menunjukkan seberapa besar feature digunakan oleh model untuk membuat prediksi. Nilai ini tidak membuktikan hubungan sebab-akibat.

---

## 5. Refleksi

Jelaskan dengan bahasa Anda sendiri:

1. Apa perbedaan **feature** dan **target**?
2. Mengapa dataset dibagi menjadi training dan testing?
3. Apa fungsi `model.fit()`?
4. Apa fungsi `model.predict()`?

Jawaban maksimal **1–2 kalimat untuk setiap pertanyaan**.

---

## Pengumpulan

Simpan notebook pada Google Drive dan kumpulkan dalam format:

```text
.ipynb
```

Gunakan format nama file:

```text
NIM_Nama_Modul02.ipynb
```

Pastikan seluruh cell telah dijalankan dan output dapat terlihat sebelum notebook dikumpulkan.
