# Exploratory Data Analysis (EDA) — Dataset Malaysia

## 1. Informasi Dataset

Dataset Malaysia yang digunakan dalam penelitian ini adalah *Outdoor Tenera Oil Palm Fruit Image Dataset* (2024), yang diperoleh melalui Zenodo.

Dataset berisi gambar tandan buah segar (TBS) kelapa sawit yang dikategorikan ke dalam lima kelas:
- Damaged
- Empty
- Overripe
- Ripe
- Unripe

Dataset dibagi menjadi dua bagian, yaitu FFBtrain dan FFBtest. Format gambar yang ditemukan adalah JPG dengan label kelas berdasarkan struktur folder.

## 2. Jumlah Data

Berdasarkan pemeriksaan dataset lokal, diperoleh hasil berikut:

| Split | Jumlah Gambar |
|---|---:|
| Train | 371 |
| Test | 95 |
| **Total** | **466** |

Dataset ini tidak memiliki folder validation pada struktur yang diperiksa.

## 3. Distribusi Kelas

Hasil pemeriksaan jumlah gambar berdasarkan kelas:

| Kelas | Train | Test | Total |
|---|---:|---:|---:|
| Damaged | 12 | 3 | 15 |
| Empty | 9 | 3 | 12 |
| Overripe | 60 | 14 | 74 |
| Ripe | 160 | 41 | 201 |
| Unripe | 130 | 34 | 164 |
| **Total** | **371** | **95** | **466** |

Kelas Ripe merupakan kelas dengan jumlah gambar terbanyak, yaitu 201 gambar. Sementara itu, kelas Empty memiliki jumlah paling sedikit, yaitu 12 gambar.

Distribusi tersebut menunjukkan ketidakseimbangan jumlah gambar antarkelas, terutama pada kelas Ripe dan Empty.

## 4. Pemeriksaan Format File

Pemeriksaan seluruh file pada direktori dataset menemukan 466 file dengan ekstensi `.JPG`.

Tidak ditemukan file anotasi dengan ekstensi `.txt`, `.xml`, atau `.json` pada pemeriksaan tersebut.

Dengan demikian, label yang tersedia pada salinan dataset ini merupakan label kategori berdasarkan nama folder kelas, bukan anotasi bounding box.

## 5. Pemeriksaan Visual

Visualisasi dilakukan dengan mengambil satu sampel gambar dari setiap kelas train.

Hasil pemeriksaan menunjukkan:
- Setiap sampel memperlihatkan tandan buah sebagai objek utama.
- Terdapat perbedaan kondisi fisik tandan pada setiap kategori.
- Sudut pengambilan gambar bervariasi.
- Kondisi pencahayaan dan latar belakang juga bervariasi.
- Beberapa gambar memiliki ketajaman yang berbeda.

Pemeriksaan visual ini merupakan sampling dan belum mewakili seluruh gambar dalam dataset.

## 6. Dimensi Gambar

Pemeriksaan dimensi dilakukan terhadap seluruh 466 gambar.

| Width | Height | Jumlah Gambar |
|---:|---:|---:|
| 757 | 568 | 233 |
| 824 | 464 | 6 |
| 2250 | 4000 | 31 |
| 2400 | 3200 | 63 |
| 2756 | 2381 | 1 |
| 3024 | 4032 | 28 |
| 3200 | 2400 | 23 |
| 4032 | 3024 | 80 |
| 4160 | 3120 | 1 |
| **Total** | | **466** |

Hasil tersebut menunjukkan bahwa dataset memiliki sembilan variasi resolusi gambar, dengan orientasi portrait maupun landscape.

Oleh karena itu, diperlukan penyesuaian ukuran input dan rasio aspek pada tahap preprocessing.

## 7. Analisis Kualitas Gambar

Kualitas gambar dianalisis menggunakan tiga indikator, yaitu brightness, contrast, dan sharpness.

| Split | Brightness | Contrast | Sharpness |
|---|---:|---:|---:|
| Train | 118,78 | 51,83 | 1.899,43 |
| Test | 118,61 | 50,70 | 2.072,33 |

### 7.1 Brightness

Nilai rata-rata brightness train sebesar 118,78 dan test sebesar 118,61. Kedua split memiliki tingkat kecerahan rata-rata yang relatif berdekatan.

### 7.2 Contrast

Nilai rata-rata contrast train sebesar 51,83, sedangkan test sebesar 50,70. Perbedaannya relatif kecil.

### 7.3 Sharpness

Nilai rata-rata sharpness train sebesar 1.899,43 dan test sebesar 2.072,33. Test memiliki rata-rata sharpness lebih tinggi dibandingkan train.

Nilai sharpness dihitung menggunakan Laplacian variance. Indikator ini mengukur respons tepi gambar dan tidak dapat digunakan secara langsung sebagai ukuran mutlak kualitas gambar.

## 8. Keterbatasan Dataset

Berdasarkan hasil pemeriksaan, terdapat beberapa keterbatasan:

1. Jumlah gambar relatif kecil, yaitu 466 gambar.
2. Distribusi kelas tidak seimbang.
3. Dataset hanya memiliki train dan test.
4. Tidak ditemukan anotasi bounding box pada file yang diperiksa.
5. Resolusi dan rasio aspek gambar bervariasi.

Keterbatasan anotasi menjadi perhatian khusus karena penelitian ini menggunakan model deteksi objek RF-DETR dan YOLO yang membutuhkan bounding box sebagai ground truth.

Oleh karena itu, diperlukan strategi anotasi tambahan atau penyesuaian rancangan eksperimen sebelum dataset digunakan untuk evaluasi deteksi objek.

## 9. Kesimpulan EDA

Berdasarkan hasil Exploratory Data Analysis, Dataset Malaysia memiliki 466 gambar yang terbagi menjadi 371 gambar train dan 95 gambar test dengan lima kategori kelas.

Kelas Ripe merupakan kelas dominan dengan 201 gambar, sedangkan Empty memiliki jumlah paling sedikit, yaitu 12 gambar.

Dataset memiliki variasi resolusi, orientasi, pencahayaan, kontras, dan ketajaman gambar. Pemeriksaan file juga menunjukkan bahwa dataset yang tersedia menggunakan label kategori berdasarkan folder dan belum menyediakan anotasi bounding box.

Temuan ini menjadi dasar untuk menentukan strategi preprocessing, harmonisasi kelas, dan penanganan anotasi sebelum Dataset Malaysia digunakan dalam eksperimen benchmark lintas domain.

## 10. Lokasi Hasil EDA

Hasil analisis disimpan pada:

`results/eda_dataset_malaysia/`

File yang dihasilkan:
- `distribusi_kelas.csv`
- `dimensi_gambar.csv`
- `kualitas_gambar.csv`

Notebook analisis:

`notebooks/03_eda_dataset_malaysia.ipynb`
