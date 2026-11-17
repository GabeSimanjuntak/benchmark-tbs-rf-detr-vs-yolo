# Exploratory Data Analysis Hasil Harmonisasi

## 1. Tujuan

Analisis ini dilakukan untuk mengetahui distribusi dua kelas target yang digunakan dalam benchmark lintas domain, yaitu Unripe dan Ripe. Ketiga dataset diselaraskan menggunakan pemetaan kelas yang telah ditetapkan sebelumnya.

## 2. Distribusi Kelas

| Dataset | Unripe | Ripe | Total Bounding Box |
|---|---:|---:|---:|
| Pabrik Kalsel | 2.563 | 807 | 3.370 |
| Perkebunan Kalteng | 3.236 | 2.878 | 6.114 |
| Malaysia v7 | 200 | 203 | 403 |
| **Total** | **5.999** | **3.888** | **9.887** |

## 3. Analisis

Dataset Pabrik Kalsel menunjukkan dominasi kelas Unripe sebesar 76,05%, sedangkan kelas Ripe sebesar 23,95%. Distribusi ini menunjukkan ketidakseimbangan kelas yang cukup besar.

Dataset Perkebunan Kalteng memiliki distribusi yang lebih berdekatan, dengan Unripe sebesar 52,94% dan Ripe sebesar 47,06%.

Dataset Malaysia v7 menunjukkan distribusi yang hampir seimbang, yaitu Unripe sebesar 49,63% dan Ripe sebesar 50,37%.

## 4. Implikasi terhadap Benchmark

Perbedaan distribusi kelas pada ketiga dataset menjadi salah satu faktor yang perlu diperhatikan dalam evaluasi generalisasi lintas domain. Perbedaan proporsi kelas dapat memengaruhi kecenderungan prediksi model dan interpretasi hasil evaluasi.

Seluruh statistik dihitung berdasarkan anotasi hasil filtering yang digunakan dalam pembangunan LODO v2. Jumlah tersebut merepresentasikan bounding box, bukan jumlah gambar.

## 5. Analisis Visual Lintas Domain

Pemeriksaan visual dilakukan menggunakan enam sampel gambar dari dataset hasil harmonisasi, yang terdiri atas kelas Unripe dan Ripe pada masing-masing domain Pabrik Kalsel, Perkebunan Kalteng, dan Malaysia v7.

Hasil pengamatan menunjukkan adanya perbedaan karakteristik visual antardomain, meliputi:

1. **Pabrik Kalsel:** gambar memperlihatkan tandan buah di atas permukaan tanah dengan latar relatif sederhana. Beberapa gambar memiliki area hitam pada bagian atas dan bawah.
2. **Perkebunan Kalteng:** gambar memiliki lingkungan visual yang lebih kompleks, dengan keberadaan daun, pelepah, dan vegetasi. Pada sampel tertentu terlihat penurunan ketajaman gambar.
3. **Malaysia v7:** gambar memiliki variasi latar, tekstur permukaan tanah, pencahayaan, dan sudut pengambilan yang berbeda dari kedua dataset Indonesia.

Selain itu, tampilan visual kelas Unripe dan Ripe tidak selalu seragam. Perbedaan warna dan kondisi permukaan tandan menunjukkan bahwa pengenalan kelas tidak dapat hanya mengandalkan satu karakteristik visual.

Perbedaan karakteristik tersebut menjadi salah satu tantangan dalam pengujian generalisasi lintas domain. Model diharapkan mampu mengenali kelas target meskipun kondisi lingkungan dan kualitas gambar berbeda.

Pemeriksaan ini bersifat sampling terhadap enam gambar dan tidak dimaksudkan untuk merepresentasikan seluruh karakteristik setiap dataset.