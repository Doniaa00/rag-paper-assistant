"""
Step 4: BGE-M3 embedding.

Reads data/chunks/section_aware_chunks.jsonl (1,834 quality-filtered,
section-aware chunks from Step 3) and embeds each chunk's chunk_text with
BAAI/bge-m3 (via FlagEmbedding), forced onto CPU to preserve VRAM for the
reranker and LLM later in the pipeline.

Outputs to data/embeddings/:
  - embeddings.npy   : float32 array, shape [N, 1024]
  - chunk_ids.jsonl  : one line per chunk, same order as embeddings.npy,
                       {chunk_id, paper_id, section_title, chunk_index_in_section}

Embedding only -- no indexing or vector database work happens here.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "embeddings.npy"
CHUNK_IDS_JSONL = EMBEDDINGS_DIR / "chunk_ids.jsonl"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32
MAX_LENGTH = 512
EMBEDDING_DIM = 1024


def load_chunks():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # The Step 3 schema has no chunk_id; synthesize a deterministic one here:
    # a zero-padded per-paper sequence number, so it's unique, stable across
    # re-runs (same input order -> same ids), and still traceable to paper_id.
    per_paper_counter = {}
    for chunk in chunks:
        pid = chunk["paper_id"]
        idx = per_paper_counter.get(pid, 0)
        chunk["chunk_id"] = f"{pid}_{idx:03d}"
        per_paper_counter[pid] = idx + 1

    return chunks


def main():
    chunks = load_chunks()
    logger.info("Loaded %d chunks from %s", len(chunks), CHUNKS_JSONL)

    logger.info("Loading %s (CPU only, fp32)...", MODEL_NAME)
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices="cpu")

    texts = [c["chunk_text"] for c in chunks]

    logger.info("Embedding %d chunks in batches of %d...", len(texts), BATCH_SIZE)
    start = time.monotonic()
    result = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    elapsed = time.monotonic() - start
    dense_vecs = np.asarray(result["dense_vecs"], dtype=np.float32)
    logger.info("Embedding finished in %.1fs (%.2fs/chunk)", elapsed, elapsed / len(texts))

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, dense_vecs)
    with open(CHUNK_IDS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "section_title": c["section_title"],
                "chunk_index_in_section": c["chunk_index_in_section"],
            }, ensure_ascii=False) + "\n")

    validate_and_report(dense_vecs, chunks, elapsed)


def validate_and_report(vecs: np.ndarray, chunks: list, elapsed: float):
    print("\n" + "=" * 70)
    print("BGE-M3 EMBEDDING REPORT")
    print("=" * 70)

    n_chunks = len(chunks)
    shape_ok = vecs.shape == (n_chunks, EMBEDDING_DIM)
    print(f"\nTotal chunks embedded: {n_chunks}")
    print(f"Embedding time: {elapsed:.1f}s ({elapsed/60:.1f} min), {elapsed/n_chunks:.3f}s/chunk avg")
    print(f"Array shape: {vecs.shape}  (expected ({n_chunks}, {EMBEDDING_DIM}))  {'OK' if shape_ok else 'MISMATCH'}")

    nan_rows = np.where(np.isnan(vecs).any(axis=1))[0]
    zero_rows = np.where(~vecs.any(axis=1))[0]
    print(f"\nNaN check: {len(nan_rows)} rows contain NaN  {'OK' if len(nan_rows)==0 else 'FAIL -- rows: ' + str(nan_rows.tolist())}")
    print(f"All-zero check: {len(zero_rows)} rows are all-zero  {'OK' if len(zero_rows)==0 else 'FAIL -- rows: ' + str(zero_rows.tolist())}")

    npy_size = EMBEDDINGS_NPY.stat().st_size
    ids_size = CHUNK_IDS_JSONL.stat().st_size
    print(f"\n{EMBEDDINGS_NPY.name}: {npy_size:,} bytes ({npy_size/1024/1024:.2f} MB)")
    print(f"{CHUNK_IDS_JSONL.name}: {ids_size:,} bytes ({ids_size/1024:.1f} KB)")

    print(f"\nOutput directory: {EMBEDDINGS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
