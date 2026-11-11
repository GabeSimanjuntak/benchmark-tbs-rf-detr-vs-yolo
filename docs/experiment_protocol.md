# Protokol Eksperimen Benchmark Lintas Domain

## 1. Tujuan

Eksperimen ini bertujuan membandingkan performa RF-DETR, YOLO11, dan YOLOv8 dalam mendeteksi kematangan tandan buah segar kelapa sawit pada domain pabrik dan perkebunan di Indonesia serta dataset outdoor Malaysia.

## 2. Harmonisasi Kelas

Seluruh dataset menggunakan dua kelas:

| Class ID | Kelas |
|---|---|
| 0 | Unripe |
| 1 | Ripe |

Kelas yang tidak termasuk dalam dua kategori tersebut tidak digunakan pada eksperimen utama.

## 3. Skenario Leave-One-Dataset-Out

Tiga skenario pengujian ditetapkan:

### LODO-1: Pengujian pada Pabrik Kalimantan Selatan

- Training: Kalteng train dan Malaysia train.
- Validation: Kalteng valid.
- Testing: Pabrik test.

### LODO-2: Pengujian pada Perkebunan Kalimantan Tengah

- Training: Pabrik train dan Malaysia train.
- Validation: Pabrik valid.
- Testing: Kalteng test.

### LODO-3: Pengujian pada Dataset Malaysia

- Training: Pabrik train dan Kalteng train.
- Validation: Pabrik valid dan Kalteng valid.
- Testing: Malaysia test.

## 4. Ketentuan Penggunaan Data

1. Dataset pengujian tidak boleh digunakan untuk training maupun pemilihan hyperparameter.
2. Split asli masing-masing dataset dipertahankan.
3. Data validation hanya berasal dari dataset training yang memiliki split validation.
4. Data test Malaysia berjumlah 89 gambar.
5. Dataset Malaysia train berjumlah 361 gambar dan hanya digunakan sebagai data training ketika Malaysia bukan domain target.

## 5. Model yang Dibandingkan

- RF-DETR.
- YOLO11.
- YOLOv8.

Eksperimen menggunakan harmonisasi kelas, ukuran input, dan protokol evaluasi yang konsisten.

## 6. Metrik Evaluasi

- Precision.
- Recall.
- F1-score.
- mAP@0.5.
- mAP@0.5:0.95.
- FPS atau latency inferensi.

## 7. Analisis Cross-Domain

Performa model dibandingkan pada ketiga domain pengujian. Penurunan mAP dianalisis untuk mengetahui kemampuan generalisasi model terhadap perbedaan kondisi pengambilan gambar dan karakteristik dataset.

## 8. Catatan Keterbatasan

Dataset Malaysia memiliki jumlah gambar lebih sedikit dibandingkan kedua dataset Indonesia. Selain itu, anotasi Malaysia diperoleh melalui anotasi manual, sedangkan kedua dataset Indonesia menggunakan anotasi dari sumber dataset masing-masing. Perbedaan proses anotasi dan karakteristik data perlu dipertimbangkan dalam interpretasi hasil.
