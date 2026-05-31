# Image Search Engine PACD

Project ini adalah image search engine berbasis Content-Based Image Retrieval
(CBIR). Sistem akan mencari gambar yang mirip berdasarkan isi visual gambar,
bukan berdasarkan nama file atau metadata.

Tahap saat ini: **Tahap 6 - Result Visualization**.

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
├── compare_indexes.py
├── evaluate_search.py
├── query_search.py
├── requirements.txt
├── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── descriptors.py
│   ├── evaluation.py
│   ├── indexing.py
│   ├── preprocessing.py
│   ├── search.py
│   └── visualization.py
├── data/
│   └── images/
└── models/
```

Keterangan:

- `context.txt`: penjelasan project, arsitektur, dan tahapan implementasi
- `requirements.txt`: daftar library Python awal
- `build_index.py`: script untuk membuat index gambar
- `compare_indexes.py`: script membandingkan beberapa index dengan precision@k
- `query_search.py`: script untuk mencari gambar yang mirip dengan query
- `evaluate_search.py`: script evaluasi precision@k berdasarkan folder kategori
- `streamlit_app.py`: dashboard sederhana untuk testing dan visualisasi
- `src/preprocessing.py`: fungsi membaca, validasi, dan konversi gambar
- `src/descriptors.py`: fungsi ekstraksi descriptor histogram HSV
- `src/indexing.py`: fungsi build, save, dan load index
- `src/search.py`: fungsi cosine distance dan pencarian top-k
- `src/evaluation.py`: fungsi evaluasi hasil pencarian
- `src/visualization.py`: fungsi menampilkan dan menyimpan grid hasil pencarian
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

Tahap visualisasi hasil sudah mencakup:

1. Menampilkan gambar query.
2. Menampilkan top-k hasil dalam grid.
3. Menampilkan nilai distance pada setiap hasil.
4. Menyimpan grid hasil ke file gambar.

## Build Index

Setelah dataset gambar dimasukkan ke `data/images/`, index dapat dibuat dengan:

```bash
python build_index.py --image-dir data/images --index-path models/index.pkl --verbose
```

Secara default, index memakai descriptor HSV. Untuk membuat index HSV dan HOG
secara terpisah:

```bash
python build_index.py --image-dir data/images --index-path models/index-hsv.pkl --descriptor hsv --verbose
python build_index.py --image-dir data/images --index-path models/index-hog.pkl --descriptor hog --verbose
```

Output index tidak perlu di-commit karena file `models/*.pkl` sudah diabaikan oleh `.gitignore`.

## Query Search

Setelah index dibuat, pencarian gambar mirip dapat dijalankan dengan:

```bash
python query_search.py --query data/images/nama_gambar.jpg --index-path models/index.pkl --top-k 10
```

Semakin kecil nilai distance, semakin mirip gambar hasil dengan gambar query.

Untuk menampilkan hasil dalam window Matplotlib:

```bash
python query_search.py --query data/images/nama_gambar.jpg --index-path models/index.pkl --top-k 10 --show
```

Untuk menyimpan hasil ke file gambar:

```bash
python query_search.py --query data/images/nama_gambar.jpg --index-path models/index.pkl --top-k 10 --save outputs/result.png
```

## Streamlit Dashboard

Dashboard sederhana dapat dijalankan untuk testing visual:

```bash
streamlit run streamlit_app.py
```

Dashboard mendukung dua cara query:

1. Upload gambar query.
2. Memasukkan path gambar query lokal.

Pastikan `models/index.pkl` sudah dibuat sebelum menjalankan pencarian.

## Evaluation

Jika dataset disusun berdasarkan folder kategori, evaluasi sederhana dapat
dilakukan dengan precision@k. Parent folder gambar dianggap sebagai label.

Contoh:

```text
data/images/
├── Borobudur-Temple/
├── Eiffel-Tower/
└── Taj-Mahal/
```

Jalankan evaluasi:

```bash
python evaluate_search.py --index-path models/index.pkl --top-k 10
```

Untuk membandingkan descriptor HSV dan HOG:

```bash
python evaluate_search.py --index-path models/index-hsv.pkl --top-k 10
python evaluate_search.py --index-path models/index-hog.pkl --top-k 10
```

Untuk membuat index kombinasi HSV + HOG:

```bash
python build_index.py --image-dir data/images --index-path models/index-hsv-hog.pkl --descriptor hsv_hog --verbose
```

Untuk membandingkan beberapa index sekaligus dan melihat descriptor terbaik
berdasarkan precision@k:

```bash
python compare_indexes.py --index-paths models/index-hsv.pkl models/index-hog.pkl models/index-hsv-hog.pkl --top-k 10
```

Untuk melihat hasil per query:

```bash
python evaluate_search.py --index-path models/index.pkl --top-k 10 --show-details
```

Jika ingin mencoba sebagian query saja:

```bash
python evaluate_search.py --index-path models/index.pkl --top-k 10 --max-queries 20
```

Catatan: query image itu sendiri dikeluarkan dari hasil evaluasi agar nilai
precision tidak naik hanya karena sistem menemukan gambar yang sama persis.

## Tahapan Berikutnya

Tahap berikutnya adalah peningkatan descriptor:

1. Menambahkan descriptor HOG.
2. Membandingkan HSV vs HOG.
3. Mencoba kombinasi HSV + HOG.
4. Membandingkan hasilnya menggunakan precision@k.
