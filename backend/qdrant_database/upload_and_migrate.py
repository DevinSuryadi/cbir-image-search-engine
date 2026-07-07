import os
import time
import uuid
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from supabase import create_client, Client

from src.cbir.deep_embedding import load_clip_model, compute_clip_embeddings
from src.cbir.files import canonicalize_image_path, list_image_files
from src.cbir.labels import get_category_label

BACKEND_ROOT = Path(__file__).resolve().parents[1]
env_path = BACKEND_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

DATASET_DIR = BACKEND_ROOT / "data" / "images"  
COLLECTION_NAME = "caltech101_clip"
SUPABASE_BUCKET = "images"

def migrate_and_upload():
    qdrant_url = os.getenv("Cluster_Endpoint") or os.getenv("CLUSTER_ENDPOINT")
    qdrant_api_key = os.getenv("api_key") or os.getenv("API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Error: 'Cluster_Endpoint' dan 'api_key' harus diatur di berkas .env untuk Qdrant Cloud.")
    if not supabase_url or not supabase_key:
        raise ValueError("Error: 'SUPABASE_URL' dan 'SUPABASE_KEY' harus diatur di berkas .env untuk Supabase Storage.")

    print("Menghubungkan ke Qdrant Cloud...")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120.0)
    
    print("Menghubungkan ke Supabase Client...")
    supabase_client: Client = create_client(supabase_url, supabase_key)

    print("Memuat model CLIP...")
    model, processor, device = load_clip_model()

    print(f"Mencari gambar di {DATASET_DIR}...")
    image_paths = list_image_files(DATASET_DIR)
    total_images = len(image_paths)
    print(f"Menemukan {total_images} gambar.")

    if total_images == 0:
        print("Tidak ditemukan gambar untuk diunggah.")
        return

    print("Menentukan dimensi vektor dari gambar pertama...")
    sample_embeddings = compute_clip_embeddings(
        image_paths=[image_paths[0]],
        model=model,
        processor=processor,
        device=device,
        batch_size=1
    )
    vector_size = int(sample_embeddings.shape[1])
    print(f"Dimensi vektor terdeteksi: {vector_size}")

    print(f"Membuat/Merekreasi koleksi '{COLLECTION_NAME}' di Qdrant Cloud...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    batch_size = 16
    points = []

    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"\n--- Memproses batch {i//batch_size + 1} ({len(batch_paths)} gambar) ---")

        public_urls = []
        for path in batch_paths:
            category = get_category_label(path)
            filename = path.name
            remote_path = f"{category}/{filename}"
            mime_type, _ = mimetypes.guess_type(path) or ("image/jpeg", None)

            uploaded = False
            for attempt in range(1, 4):
                try:
                    with open(path, "rb") as file_data:
                        supabase_client.storage.from_(SUPABASE_BUCKET).upload(
                            path=remote_path,
                            file=file_data,
                            file_options={"content-type": mime_type, "cache-control": "3600"}
                        )
                    uploaded = True
                    break
                except Exception as e:
                    if "Duplicate" in str(e) or "already exists" in str(e).lower():
                        print(f"Gambar {filename} sudah ada di Supabase. Mengabaikan upload.")
                        uploaded = True
                        break
                    print(f"Percobaan upload {attempt}/3 gagal untuk {filename}: {e}")
                    time.sleep(2)

            if uploaded:
                url_info = supabase_client.storage.from_(SUPABASE_BUCKET).get_public_url(remote_path)
                public_urls.append(url_info)
            else:
                print(f"Peringatan: Gagal mengunggah {filename} ke Supabase. Menggunakan local path fallback.")
                public_urls.append(canonicalize_image_path(path))

        embeddings = compute_clip_embeddings(
            image_paths=batch_paths,
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size
        )

        for path, embedding, public_url in zip(batch_paths, embeddings, public_urls):
            category = get_category_label(path)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "image_path": public_url, 
                        "category": category,
                        "local_path": canonicalize_image_path(path)
                    }
                )
            )

        if len(points) >= 48 or (i + batch_size) >= total_images:
            print(f"Mengunggah {len(points)} vektor ke Qdrant Cloud...")
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        wait=True,
                        points=points
                    )
                    break
                except Exception as e:
                    print(f"Gagal mengunggah vektor ke Qdrant (percobaan {attempt}/{max_retries}): {e}")
                    if attempt == max_retries:
                        raise e
                    print("Menunggu sebelum mencoba kembali...")
                    time.sleep(3 * attempt)
            points = []

    print("\nMigrasi selesai! Seluruh gambar telah diunggah ke Supabase Storage dan diindeks di Qdrant Cloud.")

if __name__ == "__main__":
    migrate_and_upload()
