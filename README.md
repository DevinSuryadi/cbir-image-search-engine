# Image Search Engine PACD

Project ini adalah image search engine berbasis Content-Based Image Retrieval
(CBIR). Sistem akan mencari gambar yang mirip berdasarkan isi visual gambar,
bukan berdasarkan nama file atau metadata.

Tahap saat ini: **Tahap 1 - Persiapan Project**.

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
├── requirements.txt
├── data/
│   └── images/
└── models/
```

Keterangan:

- `context.txt`: penjelasan project, arsitektur, dan tahapan implementasi
- `requirements.txt`: daftar library Python awal
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

## Tahapan Berikutnya

Tahap berikutnya adalah membuat preprocessing gambar:

1. Membaca gambar dari file.
2. Memvalidasi gambar berhasil dibaca.
3. Mengubah gambar dari BGR ke HSV.
4. Menyiapkan fungsi agar bisa digunakan oleh proses indexing dan query.
