"""
Step 6: Reciprocal Rank Fusion (RRF) over the dense (Qdrant) and sparse
(BM25) retrieval arms built independently in Step 5.

dense_search / sparse_search each return a plain ranked list of chunk_ids
(rank 1 first). rrf_fuse combines two such lists into one fused ranking via
the standard RRF formula: score(doc) = sum over lists containing doc of
1 / (k + rank), rank starting at 1. A doc appearing in both lists sums both
contributions.
"""

import json
import pickle
import sys
from pathlib import Path

from qdrant_client import QdrantClient

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
BM25_INDEX_PKL = DATA_DIR / "indexes" / "bm25_index.pkl"
BM25_MAPPING_JSONL = DATA_DIR / "indexes" / "bm25_mapping.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "research_papers"

# Reuse the exact tokenizer the BM25 index was built with, rather than
# re-implementing it here and risking drift.
sys.path.insert(0, str(INGESTION_DIR))
from build_bm25_index import tokenize  # noqa: E402

_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
    return _model


def dense_search(query: str, top_k: int = 20):
    """Embed query with BGE-M3, search Qdrant, return ranked chunk_ids (rank 1 first)."""
    model = _get_model()
    query_vec = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]

    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    return [hit.payload["chunk_id"] for hit in hits]


def sparse_search(query: str, top_k: int = 20):
    """Tokenize query, search BM25, return ranked chunk_ids (rank 1 first)."""
    with open(BM25_INDEX_PKL, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_MAPPING_JSONL, encoding="utf-8") as f:
        mapping = [json.loads(line) for line in f]

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_positions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [mapping[position]["chunk_id"] for position in ranked_positions]


def rrf_fuse(dense_results: list, sparse_results: list, k: int = 60):
    """Combine two ranked chunk_id lists via Reciprocal Rank Fusion.

    Returns a list of (chunk_id, fused_score) sorted descending by score.
    """
    scores = {}
    for rank, chunk_id in enumerate(dense_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(sparse_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _rank_provenance(chunk_id: str, dense_results: list, sparse_results: list):
    dense_rank = dense_results.index(chunk_id) + 1 if chunk_id in dense_results else None
    sparse_rank = sparse_results.index(chunk_id) + 1 if chunk_id in sparse_results else None
    return dense_rank, sparse_rank


def main():
    query = "How does SMOTE generate synthetic samples?"
    top_k = 20
    rrf_k = 60
    fused_top_n = 10

    print("\n" + "=" * 70)
    print(f"RRF FUSION -- query: {query!r}")
    print(f"top_k={top_k} per arm, RRF k={rrf_k}")
    print("=" * 70)

    dense_results = dense_search(query, top_k=top_k)
    sparse_results = sparse_search(query, top_k=top_k)
    fused = rrf_fuse(dense_results, sparse_results, k=rrf_k)

    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks_by_id = {json.loads(line)["chunk_id"]: json.loads(line) for line in f}

    print(f"\nFused top-{fused_top_n}:\n")
    for rank, (chunk_id, score) in enumerate(fused[:fused_top_n], start=1):
        chunk = chunks_by_id[chunk_id]
        dense_rank, sparse_rank = _rank_provenance(chunk_id, dense_results, sparse_results)
        dense_note = f"dense #{dense_rank}" if dense_rank else f"dense: not in top-{top_k}"
        sparse_note = f"sparse #{sparse_rank}" if sparse_rank else f"sparse: not in top-{top_k}"

        print(f"[{rank}] fused_score={score:.5f}  ({dense_note}, {sparse_note})")
        print(f"    chunk_id: {chunk_id}")
        print(f"    paper_id: {chunk['paper_id']}")
        print(f"    section_title: {chunk['section_title']}")
        preview = chunk["chunk_text"][:150].replace("\n", " ")
        print(f"    chunk_text: {preview}...")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
