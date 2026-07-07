import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.cbir.deep_embedding import load_clip_model, compute_clip_embeddings
from src.cbir.files import canonicalize_image_path, list_image_files
from src.cbir.labels import get_category_label

# Load .env file explicitly
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

DATASET_DIR = Path("data/images")  
COLLECTION_NAME = "caltech101_clip"     

def migrate():
    # Load connection parameters from environment variables
    qdrant_url = os.getenv("Cluster_Endpoint") or os.getenv("CLUSTER_ENDPOINT")
    qdrant_api_key = os.getenv("api_key") or os.getenv("API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "Error: 'Cluster_Endpoint' dan 'api_key' harus diatur di dalam berkas .env "
            "untuk koneksi Qdrant Cloud. Lokal fallback dinonaktifkan."
        )

    print(f"Menghubungkan ke Qdrant Cloud Cluster di: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    print(f"Membuat koleksi '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    print("Memuat model CLIP...")
    model, processor, device = load_clip_model()

    print(f"Mencari gambar di {DATASET_DIR}...")
    image_paths = list_image_files(DATASET_DIR)
    total_images = len(image_paths)
    print(f"Menemukan {total_images} gambar.")

    batch_size = 16
    points = []
    
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Memproses batch {i//batch_size + 1} ({len(batch_paths)} gambar)...")

        embeddings = compute_clip_embeddings(
            image_paths=batch_paths,
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size
        )

        for path, embedding in zip(batch_paths, embeddings):
            stable_path = canonicalize_image_path(path)
            category = get_category_label(path)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embedding.tolist(),
                    payload={
                        "image_path": stable_path,
                        "category": category
                    }
                )
            )

        if len(points) >= 100 or (i + batch_size) >= total_images:
            print(f"Mengunggah {len(points)} vektor ke Qdrant...")
            client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points
            )
            points = []

    print("Migrasi selesai! Seluruh dataset Caltech-101 berhasil dimasukkan ke Qdrant.")

if __name__ == "__main__":
    migrate()