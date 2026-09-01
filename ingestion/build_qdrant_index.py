"""
Step 5 (dense arm): Build the Qdrant hybrid-index collection.

Loads the precomputed BGE-M3 embeddings from Step 4 (data/embeddings/
embeddings.npy + chunk_ids.jsonl) -- no re-embedding -- aligns each vector's
row to its chunk_id, pulls full chunk metadata from
data/chunks/section_aware_chunks.jsonl, and upserts everything into a
Qdrant collection named "research_papers" (1024-dim, cosine distance).

Assumes a Qdrant container is already running and reachable at localhost:6333
(see docker run instructions in the Step 5 task).
"""

import json
import logging
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_NPY = DATA_DIR / "embeddings" / "embeddings.npy"
CHUNK_IDS_JSONL = DATA_DIR / "embeddings" / "chunk_ids.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"
VECTOR_SIZE = 1024
UPSERT_BATCH_SIZE = 256


def load_data():
    vecs = np.load(EMBEDDINGS_NPY)

    with open(CHUNK_IDS_JSONL, encoding="utf-8") as f:
        chunk_ids_ordered = [json.loads(line)["chunk_id"] for line in f]

    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks_by_id = {json.loads(line)["chunk_id"]: json.loads(line) for line in f}

    if vecs.shape[0] != len(chunk_ids_ordered):
        raise ValueError(
            f"Row count mismatch: embeddings.npy has {vecs.shape[0]} rows, "
            f"chunk_ids.jsonl has {len(chunk_ids_ordered)} entries."
        )

    return vecs, chunk_ids_ordered, chunks_by_id


def build_points(vecs, chunk_ids_ordered, chunks_by_id):
    points = []
    for row_idx, chunk_id in enumerate(chunk_ids_ordered):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise KeyError(f"chunk_id {chunk_id!r} (row {row_idx}) not found in {CHUNKS_JSONL}")

        payload = {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "section_title": chunk["section_title"],
            "chunk_index_in_section": chunk["chunk_index_in_section"],
            "pillar_tags": chunk["pillar_tags"],
            "chunk_text": chunk["chunk_text"],
        }
        points.append(PointStruct(id=row_idx, vector=vecs[row_idx].tolist(), payload=payload))

    return points


def main():
    vecs, chunk_ids_ordered, chunks_by_id = load_data()
    logger.info("Loaded %d vectors (dim=%d) and %d chunk records", vecs.shape[0], vecs.shape[1], len(chunks_by_id))

    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        logger.info("Collection %s already exists -- recreating for a clean build", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created collection %s (size=%d, distance=COSINE)", COLLECTION_NAME, VECTOR_SIZE)

    points = build_points(vecs, chunk_ids_ordered, chunks_by_id)

    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        logger.info("Upserted batch %d-%d / %d", i, i + len(batch), len(points))

    info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 70)
    print("QDRANT INDEX BUILD REPORT")
    print("=" * 70)
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Vector size: {VECTOR_SIZE}, distance: COSINE")
    print(f"Points count: {info.points_count}")
    print(f"Status: {info.status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
