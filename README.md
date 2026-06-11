# CBIR Image Search Engine

Image search engine berbasis Content-Based Image Retrieval (CBIR). Sistem mencari gambar yang mirip berdasarkan isi visual gambar.

## Fitur

- Preprocessing gambar BGR ke HSV
- Descriptor HSV color histogram berbasis region
- Descriptor HOG untuk bentuk dan tekstur
- Descriptor kombinasi HSV + HOG
- ORB reranking untuk mengurutkan ulang kandidat berdasarkan keypoint matching
- BoVW + TF-IDF untuk local-feature image retrieval
- Geometric verification untuk reranking kandidat BoVW
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
        ├── bovw.py
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

Jalankan ini setelah menambah, menghapus, atau mengubah dataset. Command ini
membangun semua index yang dibutuhkan aplikasi, termasuk BoVW untuk fusion:

```bash
python manage.py prepare --image-dir data/images --models-dir models --top-k 10 --verbose
```

Jika hanya ingin rebuild index klasik tanpa BoVW:

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

Query dengan weighted HOG + ORB reranking:

```bash
python manage.py query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-hog.pkl --top-k 10 --orb-rerank --candidate-k 50 --rerank-strategy weighted --distance-weight 0.4 --orb-weight 0.6 --show
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

Evaluasi weighted HOG + ORB reranking:

```bash
python manage.py evaluate --index-path models/index-hog.pkl --top-k 10 --orb-rerank --candidate-k 50 --rerank-strategy weighted --distance-weight 0.4 --orb-weight 0.6
```

Bandingkan beberapa index:

```bash
python manage.py compare --index-paths models/index-hsv.pkl models/index-hog.pkl models/index-hsv-hog.pkl --top-k 10
```

## BoVW + TF-IDF

BoVW adalah layer CBIR tambahan berbasis local feature. Metode ini cocok untuk
landmark dan bangunan karena memakai visual words dari keypoint lokal.

Build BoVW index dengan SIFT:

```bash
python manage.py bovw-build --image-dir data/images --index-path models/index-bovw.pkl --feature sift --vocabulary-size 256 --verbose
```

Jika SIFT tidak tersedia di OpenCV lokal, gunakan AKAZE:

```bash
python manage.py bovw-build --image-dir data/images --index-path models/index-bovw.pkl --feature akaze --vocabulary-size 256 --verbose
```

Query BoVW:

```bash
python manage.py bovw-query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-bovw.pkl --top-k 10 --show
```

Query BoVW dengan geometric verification:

```bash
python manage.py bovw-query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-bovw.pkl --top-k 10 --verify-top-k 50 --show
```

Evaluasi BoVW:

```bash
python manage.py bovw-evaluate --index-path models/index-bovw.pkl --top-k 10
```

Evaluasi BoVW dengan geometric verification:

```bash
python manage.py bovw-evaluate --index-path models/index-bovw.pkl --top-k 10 --verify-top-k 50
```

## Rank Fusion

Rank fusion menggabungkan beberapa ranking dari metode berbeda. Default aplikasi
menggunakan:

```text
HOG + weighted ORB reranking
HSV + HOG
BoVW + geometric verification
```

Query fusion:

```bash
python manage.py fusion-query --query "data/images/Borobudur-Temple/Borobudur.jpg" --top-k 10 --show
```

Evaluasi fusion:

```bash
python manage.py fusion-evaluate --top-k 10
```

Untuk memakai BoVW saja di Streamlit, ubah `config/search_config.json`:

```json
{
  "method": "bovw",
  "index_path": "models/index-bovw.pkl",
  "top_k": 10,
  "verify_top_k": 50
}
```

## Streamlit

```bash
streamlit run streamlit_app.py
```

Dashboard memakai parameter otomatis dari:

```text
config/search_config.json
```

Default konfigurasi:

```text
method: fusion
top_k: 10
rrf_k: 60
```

Dashboard mendukung:

- upload query image
- pencarian otomatis memakai konfigurasi terbaik yang tersimpan
- tampilan hasil dalam grid

## Catatan Deploy

Program saat ini membaca dataset dan index dari file lokal project. Jika deploy ke Streamlit Cloud, dataset dan index harus tersedia di environment deploy. Untuk dataset kecil, gambar bisa ikut GitHub. Untuk dataset besar atau dinamis, gunakan storage eksternal seperti Supabase Storage atau S3.
