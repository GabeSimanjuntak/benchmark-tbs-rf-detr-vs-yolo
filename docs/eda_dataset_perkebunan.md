# Exploratory Data Analysis (EDA) — Dataset Perkebunan Kalimantan Tengah

## 1. Informasi Dataset

Dataset Perkebunan Kalimantan Tengah merupakan dataset citra tandan buah segar (TBS) kelapa sawit yang digunakan untuk penelitian deteksi kematangan TBS.

Dataset memiliki lima kelas:
- Abnormal
- Flower
- Ripe
- Underripe
- Unripe

Format anotasi yang digunakan adalah YOLO dengan struktur folder `images` dan `labels`.

## 2. Jumlah Data

Berdasarkan hasil pemeriksaan dataset yang tersedia secara lokal, diperoleh:

| Split | Jumlah Gambar | File Anotasi | Jumlah Objek | Anotasi Kosong |
|---|---:|---:|---:|---:|
| Train | 10.221 | 10.221 | 17.188 | 325 |
| Validation | 1.583 | 1.583 | 3.106 | 3 |
| Test | 765 | 765 | 1.531 | 5 |
| **Total** | **12.569** | **12.569** | **21.825** | **333** |

Terdapat 333 file anotasi kosong. File tersebut tidak memiliki objek yang dianotasi dan perlu dipertahankan sebagai bagian dari data selama belum ada alasan metodologis untuk menghapusnya.

**Catatan perbedaan jumlah data:** instruksi tugas mencantumkan 10.207 data train, 2.896 validation, dan 1.400 test (total 14.503 gambar). Sementara itu, hasil pemeriksaan dataset lokal menunjukkan 10.221 train, 1.583 validation, dan 765 test (total 12.569 gambar). Perbedaan ini perlu dikonfirmasi terhadap sumber atau versi dataset yang digunakan.

## 3. Distribusi Kelas

Hasil perhitungan distribusi objek berdasarkan anotasi:

| Class ID | Nama Kelas | Jumlah Objek | Persentase |
|---:|---|---:|---:|
| 0 | Abnormal | 3.259 | 14,93% |
| 1 | Flower | 3.106 | 14,23% |
| 2 | Ripe | 3.307 | 15,15% |
| 3 | Underripe | 3.091 | 14,16% |
| 4 | Unripe | 9.062 | 41,52% |
| | **Total** | **21.825** | **100%** |

Kelas Unripe merupakan kelas dengan jumlah objek terbanyak, yaitu 9.062 objek atau 41,52% dari keseluruhan anotasi. Empat kelas lainnya memiliki distribusi yang relatif berdekatan, dengan persentase sekitar 14–15%.

### 3.1 Distribusi Kelas per Split

| Nama Kelas | Train | Validation | Test |
|---|---:|---:|---:|
| Abnormal | 2.802 | 318 | 139 |
| Flower | 2.673 | 251 | 182 |
| Ripe | 2.669 | 443 | 195 |
| Underripe | 2.605 | 305 | 181 |
| Unripe | 6.439 | 1.789 | 834 |
| **Total** | **17.188** | **3.106** | **1.531** |

Dominasi kelas Unripe terlihat pada seluruh split. Kondisi ini menunjukkan adanya ketidakseimbangan distribusi kelas yang perlu diperhatikan dalam eksperimen pelatihan dan evaluasi model.

## 4. Validasi Anotasi Bounding Box

Pemeriksaan dilakukan terhadap format anotasi YOLO, ID kelas, koordinat, serta ukuran bounding box.

| Split | Total Bounding Box | Valid | Invalid Format | Invalid Class | Invalid Coordinate | Invalid Size |
|---|---:|---:|---:|---:|---:|---:|
| Train | 17.188 | 17.188 | 0 | 0 | 0 | 0 |
| Validation | 3.106 | 3.106 | 0 | 0 | 0 | 0 |
| Test | 1.531 | 1.531 | 0 | 0 | 0 | 0 |

Seluruh 21.825 bounding box lolos pemeriksaan struktural dan numerik.

Pemeriksaan batas bounding box juga menunjukkan bahwa tidak ditemukan kotak yang melewati batas gambar.

| Split | Total Bounding Box | Di Luar Batas | Dalam Batas |
|---|---:|---:|---:|
| Train | 17.188 | 0 | 17.188 |
| Validation | 3.106 | 0 | 3.106 |
| Test | 1.531 | 0 | 1.531 |

Hasil tersebut menunjukkan bahwa format dan koordinat anotasi memenuhi pemeriksaan dasar. Namun, validasi numerik tidak secara otomatis menjamin ketepatan label maupun posisi objek. Pemeriksaan visual tetap diperlukan.

## 5. Pemeriksaan Visual Anotasi

Visualisasi terhadap enam sampel gambar train menunjukkan bahwa bounding box berhasil ditampilkan pada objek TBS.

Beberapa karakteristik yang terlihat:
- Terdapat objek TBS yang saling bertumpuk atau berdekatan.
- Beberapa gambar memiliki tingkat ketajaman yang berbeda.
- Sudut pengambilan gambar bervariasi.
- Kondisi pencahayaan dan latar belakang perkebunan juga bervariasi.

Pemeriksaan ini merupakan sampling visual dan belum mewakili seluruh anotasi dataset.

## 6. Dimensi Gambar

Pemeriksaan dilakukan terhadap seluruh 12.569 gambar.

| Width | Height | Jumlah Gambar |
|---:|---:|---:|
| 640 px | 640 px | 12.569 |

Seluruh gambar memiliki resolusi 640 × 640 piksel dengan rasio aspek 1:1. Dimensi yang seragam mempermudah persiapan input model pada tahap eksperimen.

## 7. Analisis Kualitas Gambar

Kualitas gambar dianalisis menggunakan tiga indikator, yaitu brightness, contrast, dan sharpness.

| Split | Brightness | Contrast | Sharpness |
|---|---:|---:|---:|
| Train | 128,46 | 60,40 | 658,42 |
| Validation | 129,28 | 63,92 | 1.327,63 |
| Test | 132,89 | 63,92 | 1.324,66 |

### 7.1 Brightness

Nilai rata-rata brightness relatif berdekatan pada ketiga split. Train memiliki nilai 128,46, validation 129,28, dan test 132,89.

### 7.2 Contrast

Nilai contrast train sebesar 60,40, sedangkan validation dan test masing-masing sebesar 63,92. Hal ini menunjukkan adanya perbedaan tingkat kontras antara train dan kedua split lainnya.

### 7.3 Sharpness

Nilai rata-rata sharpness train sebesar 658,42, lebih rendah dibandingkan validation sebesar 1.327,63 dan test sebesar 1.324,66.

Hasil boxplot juga memperlihatkan perbedaan distribusi sharpness antara train, validation, dan test. Nilai sharpness merupakan indikator berbasis Laplacian variance sehingga tidak dapat digunakan sendirian untuk menyimpulkan bahwa suatu gambar buram atau berkualitas buruk.

## 8. Kesimpulan EDA

Berdasarkan hasil Exploratory Data Analysis, Dataset Perkebunan Kalimantan Tengah memiliki 12.569 gambar dan 21.825 objek anotasi yang tersebar pada lima kelas.

Temuan utama:
1. Kelas Unripe mendominasi distribusi anotasi dengan persentase 41,52%.
2. Terdapat 333 file anotasi kosong.
3. Seluruh bounding box lolos pemeriksaan format, ID