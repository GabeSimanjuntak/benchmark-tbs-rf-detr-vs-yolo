# Visual Audit Dataset Malaysia v7

## 1. Tujuan
Melakukan pemeriksaan visual terhadap anotasi bounding box pada dataset Malaysia v7 yang telah difilter menjadi dua kelas, yaitu Unripe dan Ripe.

## 2. Hasil Pemeriksaan
- Training: 283 gambar.
- Testing: 69 gambar.
- Training dengan satu bounding box: 247 gambar.
- Training dengan lebih dari satu bounding box: 36 gambar.
- Testing dengan satu bounding box: 69 gambar.
- Total bounding box: 403.

## 3. Distribusi Multi-Bounding Box Training

| Bounding Box per Gambar | Jumlah Gambar |
|---|---:|
| 1 | 247 |
| 2 | 24 |
| 3 | 9 |
| 4 | 3 |
| Total | 283 |

## 4. Temuan Visual
Pemeriksaan dilakukan terhadap 36 gambar training yang memiliki lebih dari satu bounding box, dalam tiga batch visualisasi.

Sebagian besar gambar dengan beberapa bounding box memperlihatkan keberadaan lebih dari satu tandan buah segar pada satu gambar. Dengan demikian, jumlah bounding box yang lebih dari satu tidak secara otomatis menunjukkan kesalahan anotasi.

Beberapa anotasi yang saling berdekatan atau berpotensi mencakup bagian pelepah dan serat perlu ditinjau lebih lanjut.

## 5. Kesimpulan
Secara umum, anotasi Malaysia v7 menunjukkan konsistensi yang lebih baik dibandingkan versi sebelumnya. Tidak ditemukan indikasi kesalahan anotasi massal pada pemeriksaan visual.

Dataset dipertahankan untuk eksperimen LODO v2, dengan catatan bahwa beberapa anotasi individual yang meragukan masih perlu diperiksa lebih lanjut.