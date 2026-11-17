
# Annotation Quality Audit

## 1. Annotation Provenance

- Dataset Kalsel: menggunakan anotasi bawaan dari sumber dataset.
- Dataset Kalteng: menggunakan anotasi bawaan dari sumber dataset.
- Dataset Malaysia: anotasi bounding box dibuat secara manual oleh kelompok menggunakan Roboflow.

## 2. Kalteng Annotation Audit

Hasil pemeriksaan menemukan 15 bounding box dengan koordinat
full-image (width = 1 dan height = 1).

| Split | Jumlah |
|---|---:|
| Train | 14 |
| Validation | 0 |
| Test | 1 |
| Total | 15 |

Temuan tambahan:
- Beberapa gambar memiliki kualitas visual yang sangat rendah.
- frame_6968 memiliki anotasi dua kelas pada satu gambar.
- frame_001357 pada test memiliki bounding box yang mencakup hampir seluruh gambar.

Seluruh anotasi asli dipertahankan pada eksperimen utama.
Temuan ini dicatat sebagai keterbatasan kualitas anotasi dataset sumber.

## 3. Malaysia Annotation Audit

Dataset Malaysia v7 menggunakan anotasi manual kelompok.
Standar anotasi yang digunakan adalah satu bounding box untuk
setiap tandan TBS.

Audit visual dilakukan terhadap sampel training dan testing,
termasuk gambar dengan lebih dari satu bounding box.

## 4. Visual Audit Dataset Pabrik Kalimantan Selatan

### 4.1. Prosedur Audit

Pemeriksaan visual dilakukan terhadap dataset Pabrik Kalimantan Selatan yang memiliki anotasi bounding box dalam format YOLO. Pemeriksaan difokuskan pada dua kelas target, yaitu TBS masak (kelas asli ID 3) dan TBS mentah (kelas asli ID 4).

Audit dilakukan dengan mengambil enam sampel gambar secara acak dari masing-masing split train, validation, dan test, sehingga total terdapat 18 gambar yang diperiksa secara visual.

### 4.2. Hasil Audit

| Split | Total Gambar | Gambar dengan Kelas Target | Sampel Visual |
|---|---:|---:|---:|
| Train | 2.908 | 2.046 | 6 |
| Validation | 835 | 578 | 6 |
| Test | 417 | 265 | 6 |
| **Total** | **4.160** | **2.889** | **18** |

Hasil pemeriksaan menunjukkan bahwa bounding box secara umum mengarah pada objek tandan buah segar yang dianotasi. Beberapa kondisi yang ditemukan meliputi:

1. Bounding box yang berdekatan atau bertumpang tindih akibat posisi tandan yang berdekatan.
2. Beberapa tandan terlihat pada gambar tetapi tidak memiliki bounding box. Kondisi ini dicatat sebagai temuan visual dan belum dapat dipastikan sebagai kesalahan anotasi.
3. Variasi warna, ukuran, dan bentuk tandan dalam kelas yang sama.
4. Adanya area hitam (black padding) pada bagian atas dan bawah gambar.

### 4.3. Kesimpulan

Berdasarkan pemeriksaan terhadap 18 sampel, tidak ditemukan indikasi kesalahan anotasi yang sistematis. Secara umum, anotasi bounding box terlihat mengarah pada objek TBS yang dimaksud.

Namun, pemeriksaan ini bersifat sampling dan belum merepresentasikan keseluruhan kualitas anotasi dataset. Seluruh anotasi asli dipertahankan tanpa perubahan agar integritas dataset sumber tetap terjaga.