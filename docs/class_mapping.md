# Harmonisasi Kelas (Class Mapping)

## 1. Tujuan

Harmonisasi kelas dilakukan untuk menyelaraskan kategori kematangan Tandan Buah Segar (TBS) dari tiga dataset publik yang memiliki definisi dan jumlah kelas berbeda.

Eksperimen utama menggunakan dua kelas bersama, yaitu Unripe dan Ripe, agar evaluasi lintas domain menggunakan taksonomi yang konsisten.

## 2. Taksonomi Kelas Bersama

| Class ID | Kelas Target | Definisi Umum |
|---:|---|---|
| 0 | Unripe | Tandan buah belum mencapai tingkat kematangan panen |
| 1 | Ripe | Tandan buah telah mencapai tingkat kematangan panen |

## 3. Pemetaan Kelas Asli

| Dataset | ID Asli | Kelas Asli | ID Baru | Kelas Target |
|---|---:|---|---:|---|
| Pabrik Kalsel | 4 | TBS mentah | 0 | Unripe |
| Pabrik Kalsel | 3 | TBS masak | 1 | Ripe |
| Kalteng | 4 | Unripe | 0 | Unripe |
| Kalteng | 2 | Ripe | 1 | Ripe |
| Malaysia | - | Unripe | 0 | Unripe |
| Malaysia | - | Ripe | 1 | Ripe |

## 4. Kelas yang Tidak Digunakan

Kelas yang tidak memiliki pasangan dalam taksonomi bersama tidak disertakan pada eksperimen utama.

- Pabrik Kalsel: Janjang kosong, Kurang masak, TBS abnormal, Terlalu masak.
- Kalteng: Abnormal, Flower, Underripe.
- Malaysia: Damaged, Empty, Overripe.

Data asli tetap disimpan dan tidak dihapus.

## 5. Pertimbangan Metodologis

Pemetaan dilakukan berdasarkan kesamaan kategori kematangan dan definisi kelas dari sumber dataset. Kesetaraan label tidak berarti karakteristik visual dan kondisi akuisisi gambar identik.

Dataset Pabrik Kalsel menggambarkan TBS pada lingkungan pabrik, Dataset Kalteng menggambarkan TBS pada pohon di perkebunan, dan Dataset Malaysia menggambarkan TBS dari lingkungan perkebunan di Malaysia.

Perbedaan tersebut dipertahankan sebagai karakteristik domain dalam eksperimen generalisasi.

## 6. Keterbatasan Anotasi

Dataset Pabrik Kalsel dan Kalteng memiliki anotasi bounding box. Dataset Malaysia yang digunakan memiliki label kategori berdasarkan folder, tetapi belum menyediakan anotasi bounding box.

Oleh karena itu, anotasi bounding box tambahan diperlukan sebelum Dataset Malaysia dapat digunakan sebagai data uji deteksi objek dengan metrik mAP.

## 7. Tujuan Eksperimen

Taksonomi dua kelas ini digunakan sebagai dasar perbandingan RF-DETR, YOLO11, dan YOLOv8 dalam eksperimen benchmark lintas domain.
