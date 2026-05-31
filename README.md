# Image Search Engine PACD

Project ini adalah image search engine berbasis Content-Based Image Retrieval
(CBIR). Sistem akan mencari gambar yang mirip berdasarkan isi visual gambar,
bukan berdasarkan nama file atau metadata.

Tahap saat ini: **Tahap 5 - Query Search**.

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
├── query_search.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── descriptors.py
│   ├── indexing.py
│   ├── preprocessing.py
│   └── search.py
├── data/
│   └── images/
└── models/
```

Keterangan:

- `context.txt`: penjelasan project, arsitektur, dan tahapan implementasi
- `requirements.txt`: daftar library Python awal
- `build_index.py`: script untuk membuat index gambar
- `query_search.py`: script untuk mencari gambar yang mirip dengan query
- `src/preprocessing.py`: fungsi membaca, validasi, dan konversi gambar
- `src/descriptors.py`: fungsi ekstraksi descriptor histogram HSV
- `src/indexing.py`: fungsi build, save, dan load index
- `src/search.py`: fungsi cosine distance dan pencarian top-k
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

Tahap query search sudah mencakup:

1. Membaca gambar query.
2. Menghitung descriptor query.
3. Memuat index dari `models/index.pkl`.
4. Menghitung cosine distance.
5. Mengurutkan hasil berdasarkan distance terkecil.
6. Mengembalikan top-k gambar paling mirip.

## Build Index

Setelah dataset gambar dimasukkan ke `data/images/`, index dapat dibuat dengan:

```bash
python build_index.py --image-dir data/images --index-path models/index.pkl --verbose
```

Output index tidak perlu di-commit karena file `models/*.pkl` sudah diabaikan oleh `.gitignore`.

## Query Search

Setelah index dibuat, pencarian gambar mirip dapat dijalankan dengan:

```bash
python query_search.py --query data/images/nama_gambar.jpg --index-path models/index.pkl --top-k 10
```

Semakin kecil nilai distance, semakin mirip gambar hasil dengan gambar query.

## Tahapan Berikutnya

Tahap berikutnya adalah membuat visualisasi hasil:

1. Menampilkan gambar query.
2. Menampilkan top-k hasil pencarian.
3. Menampilkan nilai distance.
4. Menampilkan waktu pencarian.
