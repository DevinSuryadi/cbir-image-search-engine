# Image Search Engine PACD

Project ini adalah image search engine berbasis Content-Based Image Retrieval
(CBIR). Sistem akan mencari gambar yang mirip berdasarkan isi visual gambar,
bukan berdasarkan nama file atau metadata.

Tahap saat ini: **Tahap 4 - Indexing**.

## Rencana Baseline

- Descriptor: region-based HSV color histogram
- Similarity metric: cosine distance
- Index storage: pickle file
- Output: top-k gambar paling mirip, nilai distance, dan waktu query

## Struktur Folder

```text
.
├── context.txt
├── README.md
├── build_index.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── descriptors.py
│   ├── indexing.py
│   └── preprocessing.py
├── data/
│   └── images/
└── models/
```

Keterangan:

- `context.txt`: penjelasan project, arsitektur, dan tahapan implementasi
- `requirements.txt`: daftar library Python awal
- `build_index.py`: script untuk membuat index gambar
- `src/preprocessing.py`: fungsi membaca, validasi, dan konversi gambar
- `src/descriptors.py`: fungsi ekstraksi descriptor histogram HSV
- `src/indexing.py`: fungsi build, save, dan load index
- `data/images/`: lokasi dataset gambar
- `models/`: lokasi penyimpanan index/model hasil ekstraksi fitur

## Library Awal

- OpenCV: membaca dan memproses gambar
- NumPy: operasi array dan vector
- Matplotlib: visualisasi gambar dan hasil pencarian

## Instalasi Dependency

```bash
pip install -r requirements.txt
```

## Tahapan Saat Ini

Tahap preprocessing sudah mencakup:

1. Membaca gambar dari file.
2. Memvalidasi gambar berhasil dibaca.
3. Mengubah gambar dari BGR ke HSV.
4. Menyiapkan fungsi agar bisa digunakan oleh proses indexing dan query.

Tahap feature descriptor sudah mencakup:

1. Membagi gambar menjadi 5 region.
2. Menghitung histogram HSV pada setiap region.
3. Melakukan normalisasi histogram.
4. Menggabungkan histogram menjadi satu descriptor.

Tahap indexing sudah mencakup:

1. Membaca semua gambar dari `data/images/`.
2. Menghitung descriptor setiap gambar.
3. Menyimpan path gambar dan descriptor.
4. Menyimpan index ke file di folder `models/`.

## Build Index

Setelah dataset gambar dimasukkan ke `data/images/`, index dapat dibuat dengan:

```bash
python build_index.py --image-dir data/images --index-path models/index.pkl --verbose
```

Output index tidak perlu di-commit karena file `models/*.pkl` sudah diabaikan oleh `.gitignore`.

## Tahapan Berikutnya

Tahap berikutnya adalah membuat query search:

1. Membaca gambar query.
2. Menghitung descriptor query.
3. Memuat index dari `models/index.pkl`.
4. Menghitung cosine distance.
5. Mengurutkan hasil dan mengambil top-k gambar paling mirip.
