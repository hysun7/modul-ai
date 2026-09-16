# Modul 01: Google Colab dan Membaca Data

## Tujuan

Setelah menyelesaikan praktikum ini, mahasiswa diharapkan mampu:

1. Menggunakan Google Colab sebagai environment pemrograman Python.
2. Mengakses dataset yang tersimpan pada Google Drive.
3. Membaca dataset menggunakan library Pandas.
4. Memahami struktur dasar data dalam bentuk DataFrame.
5. Menampilkan, memilih, dan melakukan analisis sederhana terhadap data.

## Persiapan

Sebelum memulai praktikum, siapkan:

- Akun Google untuk mengakses Google Colab dan Google Drive.
- Browser dan koneksi internet.
- Dataset yang digunakan pada praktikum:
  [📥 Download california_housing_test.csv](data/california_housing_test.csv)

!!! tip "Tips"
Jalankan setiap cell kode secara berurutan dan perhatikan output yang dihasilkan sebelum melanjutkan ke langkah berikutnya.

---

## Langkah-langkah Praktikum

### 1. Persiapan Environment

Buka [Google Colab](https://colab.research.google.com/) dan buat notebook baru.

Ubah nama notebook menjadi:

```text
modul01.ipynb
```

Google Colab memungkinkan kita menjalankan Python langsung melalui browser tanpa perlu melakukan instalasi Python pada komputer.

Untuk memastikan Python dapat dijalankan, coba jalankan kode berikut.

```python
import sys
print(sys.version)
```

Kemudian coba program Python sederhana.

```python
print("Hello Artificial Intelligence!")
```

!!! note "Menjalankan Cell"
Cell dapat dijalankan dengan menekan tombol **▶** di sebelah kiri cell atau menggunakan kombinasi tombol **Shift + Enter**.

---

### 2. Menghubungkan Google Drive

Dataset yang digunakan dapat disimpan pada Google Drive agar dapat digunakan kembali pada praktikum berikutnya.

Hubungkan Google Colab dengan Google Drive menggunakan kode berikut.

```python
from google.colab import drive

drive.mount('/content/drive')
```

Ikuti proses autentikasi hingga Google Drive berhasil terhubung.

File pada Google Drive kemudian dapat diakses melalui:

```text
/content/drive/MyDrive/
```

Misalnya dataset disimpan pada folder:

```text
MyDrive/praktikum_ai/california_housing_test.csv
```

maka lokasi file dapat ditulis sebagai:

```python
filepath = '/content/drive/MyDrive/praktikum_ai/california_housing_test.csv'
```

!!! warning "Perhatikan Lokasi File"
Nama folder dan nama file harus sesuai dengan lokasi dataset pada Google Drive masing-masing.

---

### 3. Import Library

Pada praktikum ini kita menggunakan library **Pandas**.

Import Pandas menggunakan:

```python
import pandas as pd
```

Pandas merupakan library Python yang banyak digunakan untuk membaca, mengolah, dan menganalisis data dalam bentuk tabel.

---

### 4. Membaca Dataset

Dataset dapat dibaca menggunakan fungsi `pd.read_csv()`.

Jika dataset berada di Google Drive:

```python
filepath = '/content/drive/MyDrive/praktikum_ai/california_housing_test.csv'

df = pd.read_csv(filepath)
```

Google Colab juga menyediakan dataset contoh yang sama pada folder `sample_data`.

Dataset tersebut dapat dibaca menggunakan:

```python
filepath = 'sample_data/california_housing_test.csv'

df = pd.read_csv(filepath)
```

Periksa tipe data dari variabel `df`.

```python
type(df)
```

Output menunjukkan bahwa `df` merupakan objek:

```text
pandas.core.frame.DataFrame
```

**DataFrame** merupakan struktur data Pandas yang berbentuk tabel dan terdiri dari baris serta kolom.

---

### 5. Menampilkan Data

Gunakan fungsi `head()` untuk melihat beberapa baris pertama dari dataset.

```python
df.head()
```

Secara default, `head()` menampilkan 5 baris pertama.

Untuk menampilkan 10 baris pertama:

```python
df.head(10)
```

Untuk melihat beberapa baris terakhir:

```python
df.tail()
```

!!! question "Coba Amati"
Menurut Anda, apa yang direpresentasikan oleh setiap **baris** dan **kolom** pada dataset tersebut?

---

### 6. Melihat Struktur Dataset

Untuk mengetahui ukuran dataset, gunakan:

```python
df.shape
```

Output memiliki format:

```text
(jumlah_baris, jumlah_kolom)
```

Untuk melihat seluruh nama kolom:

```python
df.columns
```

Untuk melihat informasi lebih lengkap mengenai dataset:

```python
df.info()
```

Fungsi `info()` memberikan informasi seperti:

* jumlah baris,
* nama kolom,
* jumlah nilai non-null,
* tipe data setiap kolom.

Untuk melihat statistik dasar dari kolom numerik:

```python
df.describe()
```

Fungsi `describe()` antara lain menampilkan:

* `count`: jumlah data,
* `mean`: nilai rata-rata,
* `std`: standar deviasi,
* `min`: nilai minimum,
* `max`: nilai maksimum.

---

### 7. Mengecek Missing Values

Dataset dapat memiliki nilai kosong atau **missing value**.

Gunakan:

```python
df.isnull().sum()
```

Kode tersebut menampilkan jumlah missing value pada setiap kolom.

!!! question "Coba Amati"
Apakah terdapat missing value pada dataset `california_housing_test.csv`?

---

### 8. Mengambil Satu Kolom

Untuk mengambil satu kolom dari DataFrame, gunakan:

```python
df['nama_kolom']
```

Contoh:

```python
df['total_rooms']
```

Untuk menampilkan 5 data pertama dari kolom tersebut:

```python
df['total_rooms'].head()
```

---

### 9. Mengambil Beberapa Kolom

Untuk mengambil lebih dari satu kolom, gunakan:

```python
df[['nama_kolom_1', 'nama_kolom_2']]
```

Contoh:

```python
df[['total_rooms', 'total_bedrooms']].head()
```

Contoh lainnya:

```python
df[['median_income', 'median_house_value']].head()
```

---

### 10. Menampilkan Data Berdasarkan Kondisi

Kita dapat menampilkan hanya baris yang memenuhi kondisi tertentu.

Misalnya kita ingin menampilkan data dengan:

```text
total_rooms > 500
```

Gunakan:

```python
df[df['total_rooms'] > 500]
```

Contoh lainnya:

```text
total_bedrooms < 200
```

Gunakan:

```python
df[df['total_bedrooms'] < 200]
```

Hasil filter dapat disimpan ke variabel baru.

```python
filtered_data = df[df['total_rooms'] > 500]

filtered_data.head()
```

Untuk mengetahui jumlah data yang memenuhi kondisi:

```python
filtered_data.shape
```

---

### 11. Menghitung Statistik Sederhana

Pandas menyediakan beberapa fungsi statistik sederhana.

Untuk menghitung rata-rata:

```python
df['total_rooms'].mean()
```

Untuk mencari nilai minimum:

```python
df['total_rooms'].min()
```

Untuk mencari nilai maksimum:

```python
df['total_rooms'].max()
```

Contoh lain, untuk menghitung rata-rata nilai rumah:

```python
df['median_house_value'].mean()
```

---

### 12. Menghapus Kolom dengan `drop()`

Kolom yang tidak diperlukan dapat dihapus menggunakan fungsi `drop()`.

Contoh:

```python
df_cleaned = df.drop(['longitude'], axis=1)
```

Periksa kembali struktur DataFrame.

```python
df_cleaned.info()
```

Perhatikan bahwa kolom `longitude` sudah tidak terdapat pada `df_cleaned`.

!!! note "Catatan"
DataFrame asli `df` tidak berubah karena hasil penghapusan kolom disimpan pada variabel baru yaitu `df_cleaned`.

---

## Latihan Praktikum

Gunakan dataset `california_housing_test.csv` untuk menyelesaikan latihan berikut.

### Latihan 1

Tampilkan:

* jumlah baris dan kolom dataset,
* seluruh nama kolom.

Gunakan:

```python
df.shape
df.columns
```

---

### Latihan 2

Tampilkan 5 data pertama dari kolom:

* `median_income`
* `median_house_value`

---

### Latihan 3

Tampilkan data dengan kondisi:

```text
median_income > 5
```

Kemudian tentukan berapa jumlah data yang memenuhi kondisi tersebut.

---

### Latihan 4

Untuk kolom `median_house_value`, hitung:

* rata-rata,
* nilai minimum,
* nilai maksimum.

---

## Menuju Machine Learning

Pada praktikum ini kita telah menggunakan Python untuk membaca dan memahami dataset.

Perhatikan kolom-kolom berikut:

```text
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
median_house_value
```

Misalkan kita ingin membuat model untuk memprediksi:

```text
median_house_value
```

Dalam Machine Learning, data dapat dibagi menjadi **feature** dan **target**.

```text
Feature / Input (X)
-------------------
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income

        ↓

 Machine Learning

        ↓

Target / Output (y)
-------------------
median_house_value
```

Kolom yang digunakan sebagai masukan disebut **feature**, sedangkan nilai yang ingin diprediksi disebut **target**.

Pada **Modul 02**, dataset ini akan digunakan untuk membuat model Machine Learning menggunakan **Random Forest**.

---

# Tugas Rumah

Kerjakan tugas berikut menggunakan Google Colab dan simpan notebook pada Google Drive.

## 1. Mengenali DataFrame

Cek tipe data dari variabel `df` menggunakan:

```python
type(df)
```

Jelaskan secara singkat apa yang dimaksud dengan **DataFrame**.

---

## 2. Seleksi Data

Tampilkan 5 data pertama dari kolom:

* `housing_median_age`
* `median_income`
* `median_house_value`

---

## 3. Filter Data

Tampilkan data dengan kondisi:

```text
total_rooms > 1000
```

Kemudian tentukan jumlah data yang memenuhi kondisi tersebut.

---

## 4. Statistik Sederhana

Untuk kolom `median_income`, hitung:

* rata-rata,
* nilai minimum,
* nilai maksimum.

---

## 5. Eksplorasi Mandiri

Pilih satu kolom numerik selain `median_income`.

Lakukan:

1. Tampilkan 5 data pertama.
2. Hitung nilai rata-rata.
3. Hitung nilai minimum.
4. Hitung nilai maksimum.
5. Tuliskan **1–2 kalimat kesimpulan** berdasarkan hasil analisis.

---

## Pengumpulan

Simpan notebook pada Google Drive dan kumpulkan dalam format:

```text
.ipynb
```

Gunakan format nama file:

```text
NIM_Nama_Modul01.ipynb
```
