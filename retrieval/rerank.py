"""
Step 7: Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Takes the fused top-10 from RRF (Step 6) and reorders them by a cross-
encoder's own (query, chunk_text) relevance score. This is the first
GPU-resident component in the pipeline -- loaded fp16 on CUDA, since dense
embedding (Step 4) was deliberately kept on CPU to preserve VRAM for this
and the LLM later.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rrf_fusion import dense_search, sparse_search, rrf_fuse  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- the reranker is meant to run on GPU (fp16).")
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True, devices="cuda")
    return _reranker


def rerank(query: str, candidate_chunks: list, top_n: int = 5):
    """Score each (query, chunk_text) pair with the cross-encoder and return
    the top_n candidates reordered by the reranker's own relevance score.

    candidate_chunks: list of dicts, each with at least a "chunk_text" key.
    Returns: list of (chunk_dict, reranker_score), sorted descending, length top_n.
    """
    reranker = _get_reranker()
    pairs = [(query, c["chunk_text"]) for c in candidate_chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    scored = list(zip(candidate_chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def _report_vram(label: str):
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"{label}: allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")


def main():
    query = "How does SMOTE generate synthetic samples?"
    top_k = 20
    rrf_k = 60
    fused_top_n = 10
    rerank_top_n = 5

    print("\n" + "=" * 70)
    print(f"FULL PIPELINE -- query: {query!r}")
    print("dense top-20 -> sparse top-20 -> RRF fuse -> top-10 -> rerank -> top-5")
    print("=" * 70)

    print("\nGPU status before loading reranker:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    _report_vram("  VRAM before load")

    reranker = _get_reranker()
    # FlagReranker allocates GPU memory lazily on first inference, not at
    # construction -- a warm-up call is needed to get a真实 "after load"
    # VRAM reading rather than a premature 0 MB (weights aren't materialized
    # on the GPU until the first forward pass).
    reranker.compute_score([("warmup", "warmup")], normalize=True)
    torch.cuda.synchronize()
    print("\nReranker loaded onto CUDA (fp16).")
    _report_vram("  VRAM after load (post warm-up)")

    dense_results = dense_search(query, top_k=top_k)
    sparse_results = sparse_search(query, top_k=top_k)
    fused = rrf_fuse(dense_results, sparse_results, k=rrf_k)[:fused_top_n]

    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        import json
        chunks_by_id = {json.loads(line)["chunk_id"]: json.loads(line) for line in f}

    fused_rank_by_id = {chunk_id: rank for rank, (chunk_id, _score) in enumerate(fused, start=1)}
    candidate_chunks = [chunks_by_id[chunk_id] for chunk_id, _score in fused]

    reranked = rerank(query, candidate_chunks, top_n=rerank_top_n)
    _report_vram("\n  VRAM after inference")

    print(f"\nReranked top-{rerank_top_n}:\n")
    for new_rank, (chunk, score) in enumerate(reranked, start=1):
        fused_rank = fused_rank_by_id[chunk["chunk_id"]]
        movement = fused_rank - new_rank
        if movement > 0:
            move_note = f"moved up {movement}"
        elif movement < 0:
            move_note = f"moved down {-movement}"
        else:
            move_note = "unchanged"
        print(f"[{new_rank}] reranker_score={score:.5f}  (was fused #{fused_rank}, {move_note})")
        print(f"    chunk_id: {chunk['chunk_id']}")
        print(f"    paper_id: {chunk['paper_id']}")
        print(f"    section_title: {chunk['section_title']}")
        preview = chunk["chunk_text"][:150].replace("\n", " ")
        print(f"    chunk_text: {preview}...")
        print()

    target_id = "2402.17398_006"
    target_in_reranked = next((i for i, (c, _s) in enumerate(reranked, start=1) if c["chunk_id"] == target_id), None)
    print(f"Tracking {target_id} (fused #{fused_rank_by_id.get(target_id, 'not in fused top-10')}): ", end="")
    if target_in_reranked:
        print(f"reranked to #{target_in_reranked}")
    else:
        print(f"not in reranked top-{rerank_top_n}")

    print("=" * 70)


if __name__ == "__main__":
    main()
