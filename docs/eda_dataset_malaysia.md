# Exploratory Data Analysis dan Preprocessing Dataset Malaysia

## 1. Sumber Dataset

Dataset Malaysia menggunakan *Outdoor Tenera Oil Palm Fruit Image Dataset* yang diperoleh dari Zenodo. Dataset asli terdiri atas 466 gambar yang dikategorikan ke dalam lima kelas, yaitu Damaged, Empty, Overripe, Ripe, dan Unripe.

Untuk eksperimen deteksi objek, dilakukan anotasi bounding box menggunakan Roboflow. Sebanyak 16 gambar duplikat dihapus sehingga dataset hasil anotasi terdiri atas 450 gambar.

## 2. Hasil Anotasi

Anotasi difokuskan pada dua kelas yang digunakan dalam eksperimen utama, yaitu Unripe dan Ripe.

Hasil ekspor Roboflow v4 terdiri atas:

| Split | Jumlah gambar | Jumlah objek |
|---|---:|---:|
| Train | 361 | 1.499 |
| Test | 89 | 703 |
| **Total** | **450** | **2.202** |

## 3. Distribusi Kelas

| Kelas | Train | Test | Total |
|---|---:|---:|---:|
| Unripe | 425 | 189 | 614 |
| Ripe | 1.074 | 514 | 1.588 |
| **Total** | **1.499** | **703** | **2.202** |

Distribusi menunjukkan bahwa kelas Ripe memiliki jumlah objek lebih banyak dibandingkan Unripe. Ketidakseimbangan ini perlu diperhatikan ketika menginterpretasikan hasil evaluasi model.

## 4. Preprocessing dan Harmonisasi Label

Dataset Roboflow menggunakan urutan kelas Ripe sebagai ID 0 dan Unripe sebagai ID 1. Untuk menyamakan kelas dengan dataset Pabrik Kalimantan Selatan dan Perkebunan Kalimantan Tengah, dilakukan remapping sebagai berikut:

| Kelas | ID Roboflow | ID standar penelitian |
|---|---:|---:|
| Unripe | 1 | 0 |
| Ripe | 0 | 1 |

Gambar disalin tanpa perubahan, sedangkan ID kelas pada file anotasi YOLO disesuaikan. Koordinat bounding box dipertahankan.

## 5. Validasi Anotasi

Pemeriksaan dilakukan terhadap jumlah gambar, file label, ID kelas, dan koordinat bounding box.

| Pemeriksaan | Hasil |
|---|---:|
| Label hilang | 0 |
| Label tanpa gambar | 0 |
| ID kelas tidak valid | 0 |
| Bounding box tidak valid | 0 |
| Label kosong | 0 |

Seluruh file anotasi pada train dan test berhasil melewati validasi struktural.

## 6. Struktur Dataset Hasil Preprocessing

Dataset hasil preprocessing disimpan pada `datasets/processed/dataset-malaysia/` dengan struktur:

```text
dataset-malaysia/
├── train/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Konfigurasi kelas menggunakan dua kelas, yaitu Unripe (ID 0) dan Ripe (ID 1).

## 7. Catatan Eksperimen

Dataset Malaysia dipersiapkan sebagai domain pengujian lintas domain. Dalam skenario Malaysia sebagai dataset yang ditahan (*held-out domain*), data Malaysia tidak digunakan untuk training maupun pemilihan hyperparameter. Pembagian train dan test asli tetap dipertahankan sebagai informasi asal data; protokol evaluasi akhir harus ditetapkan secara konsisten untuk seluruh skenario eksperimen.

