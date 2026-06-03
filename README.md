# CBIR Image Search Engine

Image search engine berbasis Content-Based Image Retrieval (CBIR). Sistem mencari gambar yang mirip berdasarkan isi visual gambar, bukan nama file atau metadata.

## Fitur

- Preprocessing gambar BGR ke HSV
- Descriptor HSV color histogram berbasis region
- Descriptor HOG untuk bentuk dan tekstur
- Descriptor kombinasi HSV + HOG
- ORB reranking untuk mengurutkan ulang kandidat berdasarkan keypoint matching
- Indexing descriptor ke file pickle
- Query top-k gambar paling mirip
- Evaluasi precision@k berbasis folder kategori
- Dashboard Streamlit untuk upload query dan melihat hasil

## Struktur Project

```text
.
├── context.txt
├── manage.py
├── README.md
├── requirements.txt
├── streamlit_app.py
├── data/
│   └── images/
├── models/
├── outputs/
└── src/
    ├── __init__.py
    └── cbir/
        ├── __init__.py
        ├── descriptors.py
        ├── evaluation.py
        ├── indexing.py
        ├── preprocessing.py
        ├── reranking.py
        ├── search.py
        └── visualization.py
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Dataset

Letakkan dataset gambar di:

```text
data/images/
```

Struktur kategori disarankan memakai folder:

```text
data/images/
├── Borobudur-Temple/
├── Eiffel-Tower/
└── Taj-Mahal/
```

Nama folder digunakan sebagai label saat evaluasi precision@k.

## Rebuild Semua Index

Jalankan ini setelah menambah, menghapus, atau mengubah dataset:

```bash
python manage.py rebuild --image-dir data/images --models-dir models --top-k 10 --verbose
```

Command tersebut membuat:

```text
models/index-hsv.pkl
models/index-hog.pkl
models/index-hsv-hog.pkl
models/index-best.pkl
```

`index-best.pkl` adalah salinan index dengan precision@k terbaik.

## Build Satu Index

```bash
python manage.py build --image-dir data/images --index-path models/index-hog.pkl --descriptor hog --verbose
```

Pilihan descriptor:

```text
hsv
hog
hsv_hog
```

## Query

```bash
python manage.py query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-hog.pkl --top-k 10 --show
```

Query dengan ORB reranking:

```bash
python manage.py query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-hog.pkl --top-k 10 --orb-rerank --candidate-k 30 --show
```

Simpan visualisasi hasil:

```bash
python manage.py query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-hog.pkl --top-k 10 --save outputs/result.png
```

## Evaluasi

```bash
python manage.py evaluate --index-path models/index-hog.pkl --top-k 10
```

Evaluasi dengan ORB reranking:

```bash
python manage.py evaluate --index-path models/index-hog.pkl --top-k 10 --orb-rerank --candidate-k 30
```

Bandingkan beberapa index:

```bash
python manage.py compare --index-paths models/index-hsv.pkl models/index-hog.pkl models/index-hsv-hog.pkl --top-k 10
```

## Streamlit

```bash
streamlit run streamlit_app.py
```

Dashboard mendukung:

- upload query image
- memilih index/descriptor
- top-k slider
- ORB reranking
- tampilan hasil dalam grid

## Catatan Deploy

Program saat ini membaca dataset dan index dari file lokal project. Jika deploy ke Streamlit Cloud, dataset dan index harus tersedia di environment deploy. Untuk dataset kecil, gambar bisa ikut GitHub. Untuk dataset besar atau dinamis, gunakan storage eksternal seperti Supabase Storage atau S3.

## File Yang Tidak Perlu Di-Commit

File index dan output visualisasi tidak perlu di-commit:

```text
models/*.pkl
outputs/
```
