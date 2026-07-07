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
- CLIP pretrained deep embedding untuk semantic image retrieval
- Indexing descriptor ke file pickle
- Query top-k gambar paling mirip
- Evaluasi precision@k berbasis folder kategori
- Dashboard Streamlit untuk upload query dan melihat hasil

> [!IMPORTANT]
> Proyek ini sekarang menggunakan struktur **Monorepo** dengan pembagian folder `backend/` dan `frontend/`. 
> Semua modul Python, dataset, model, dan skrip CLI yang didokumentasikan di bawah ini berada di dalam folder `backend/`. 
> Sebelum menjalankan perintah apa pun di bawah, masuklah ke folder backend terlebih dahulu: `cd backend`.

## Struktur Project

```text
.
├── README.md
├── cbir_upgrade_plan.md
├── frontend/          # Folder proyek Next.js (Frontend)
└── backend/           # Folder proyek Python (Backend)
    ├── manage.py
    ├── requirements.txt
    ├── streamlit_app.py
    ├── data/
    │   └── images/    # Tempat menyimpan dataset gambar
    ├── models/
    └── src/
        └── cbir/
```

## Instalasi

Masuk ke folder `backend` terlebih dahulu:
```bash
cd backend
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

Default classic index CLI sekarang memakai HOG, sehingga `query` dan `evaluate`
tanpa `--index-path` akan memakai `models/index-hog.pkl`.

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

## Deep Learning CLIP

Deep learning digunakan sebagai layer tambahan untuk perbandingan dengan metode
klasik. Model tidak dilatih dari nol, tetapi memakai pretrained CLIP sebagai
feature extractor.

Build CLIP index:

```bash
python manage.py deep-build --image-dir data/images --index-path models/index-clip.pkl --verbose
```

Evaluasi CLIP:

```bash
python manage.py deep-evaluate --index-path models/index-clip.pkl --top-k 10
```

Query CLIP:

```bash
python manage.py deep-query --query "data/images/Borobudur-Temple/Borobudur.jpg" --index-path models/index-clip.pkl --top-k 10 --show
```

Notebook eksperimen:

```text
notebooks/deep_learning_search.ipynb
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
- pilihan Classic Fusion atau Deep Learning CLIP
- tampilan hasil dalam grid

## Catatan Deploy

Program membaca dataset dan index dari file lokal project. Saat deploy ke
Streamlit Cloud, "lokal" berarti filesystem server Streamlit, bukan laptop.
Karena itu dataset dan index harus tersedia di environment deploy.

Untuk mode Deep Learning CLIP, pastikan file index berikut ikut tersedia:

```text
models/index-clip.pkl
```

Jika file tersebut sudah dibuat di lokal tetapi belum muncul di GitHub, tambahkan
ke commit:

```bash
git add models/index-clip.pkl
git commit -m "chore: add CLIP index for deployment"
git push
```

Untuk dataset kecil, gambar dan index dapat ikut GitHub. Jika index `.pkl`
diabaikan `.gitignore`, tambahkan secara paksa hanya file yang dibutuhkan:

```bash
git add -f models/index-clip.pkl
```

Untuk dataset besar atau dinamis, gunakan storage eksternal seperti Supabase
Storage atau S3.

Dashboard Streamlit memakai cache untuk deep learning:

- index CLIP di-load satu kali
- model CLIP di-load satu kali
- query berikutnya hanya menghitung embedding gambar upload
