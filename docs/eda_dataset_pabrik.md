# Exploratory Data Analysis — Dataset Pabrik Kalimantan Selatan

## 1. Informasi Dataset

Dataset Pabrik Kalimantan Selatan merupakan dataset deteksi kematangan tandan buah segar (TBS) kelapa sawit yang terdiri dari enam kelas.

Dataset dibagi menjadi tiga bagian:
- Training: 2.908 gambar
- Validation: 835 gambar
- Testing: 417 gambar

Total terdapat 4.160 gambar dan 14.558 anotasi objek.

## 2. Distribusi Kelas

| Kelas | Jumlah Objek | Persentase |
|---|---:|---:|
| Janjang kosong | 857 | 5,89% |
| Kurang masak | 2.575 | 17,69% |
| TBS abnormal | 2.599 | 17,85% |
| TBS masak | 2.973 | 20,42% |
| TBS mentah | 2.913 | 20,01% |
| Terlalu masak | 2.641 | 18,14% |

Kelas Janjang kosong memiliki jumlah objek paling sedikit, sedangkan TBS masak memiliki jumlah paling banyak. Hal ini menunjukkan ketidakseimbangan distribusi kelas.

## 3. Pemeriksaan Anotasi

Pemeriksaan terhadap 14.558 bounding box menunjukkan:
- Tidak ditemukan format anotasi yang tidak valid.
- Seluruh Class ID berada pada rentang 0–5.
- Tidak ditemukan koordinat di luar rentang 0–1.
- Tidak ditemukan ukuran bounding box yang tidak valid.
- Tidak ditemukan bounding box yang melewati batas gambar.
- Tidak ditemukan file anotasi kosong.

Pemeriksaan visual terhadap enam sampel gambar menunjukkan bahwa bounding box secara umum berada pada objek TBS. Pemeriksaan visual ini masih bersifat sampling.

## 4. Dimensi Gambar

Seluruh 4.160 gambar memiliki resolusi seragam 416 × 416 piksel dengan rasio aspek 1:1.

## 5. Pemeriksaan Kualitas Gambar

| Metrik | Train | Validation | Test |
|---|---:|---:|---:|
| Brightness | 76,44 | 76,59 | 76,48 |
| Contrast | 79,94 | 80,32 | 79,97 |
| Sharpness | 11.159,59 | 11.375,94 | 11.301,88 |

Seluruh gambar berhasil dibaca dan tidak ditemukan gambar rusak. Pemeriksaan sampel gambar dengan nilai sharpness terendah menunjukkan beberapa gambar tampak kurang tajam. Area hitam pada bagian atas dan bawah gambar juga perlu diperhatikan karena dapat memengaruhi pengukuran kualitas gambar.

## 6. Kesimpulan

Dataset Pabrik Kalimantan Selatan memiliki 4.160 gambar dengan 14.558 anotasi objek. Format anotasi dan batas bounding box lolos pemeriksaan numerik awal. Resolusi gambar seragam, sedangkan distribusi kelas menunjukkan jumlah Janjang kosong lebih sedikit dibandingkan kelas lainnya.

Hasil EDA ini digunakan sebagai dokumentasi kondisi awal dataset sebelum proses harmonisasi kelas, preprocessing, dan eksperimen model deteksi objek.

## 7. Lokasi Hasil Analisis

Seluruh hasil analisis dalam format CSV disimpan pada:

`results/eda_dataset_pabrik/`
