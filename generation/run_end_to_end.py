"""
Step 8 end-to-end test: dense -> sparse -> RRF fuse -> rerank (top-5) -> generate.

Runs the full pipeline on the project's standard validation query and prints
the generated answer plus a VRAM check while the reranker and LLM are both
resident, as they would be in a real query.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from rrf_fusion import dense_search, sparse_search, rrf_fuse  # noqa: E402
from rerank import rerank, _get_reranker  # noqa: E402
from generation.local_ollama import OllamaBackend  # noqa: E402

CHUNKS_JSONL = PROJECT_ROOT / "data" / "chunks" / "section_aware_chunks.jsonl"


def _report_vram(label: str):
    # torch.cuda.memory_allocated() only sees THIS process's own CUDA
    # allocations (the reranker). Ollama runs as a separate OS process
    # (llama-server.exe) with its own independent CUDA context, so the only
    # way to see the true combined total across both is nvidia-smi.
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"{label} (this process only): allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        used_mb, total_mb = (int(x.strip()) for x in result.stdout.strip().split(","))
        print(f"{label} (system-wide, nvidia-smi): {used_mb} MB / {total_mb} MB total")
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as exc:
        print(f"{label}: nvidia-smi query failed ({exc})")


def main():
    query = "How does SMOTE generate synthetic samples?"
    top_k = 20
    rrf_k = 60
    fused_top_n = 10
    rerank_top_n = 5

    print("\n" + "=" * 70)
    print(f"STEP 8 END-TO-END -- query: {query!r}")
    print("dense top-20 -> sparse top-20 -> RRF fuse -> top-10 -> rerank top-5 -> generate")
    print("=" * 70)

    _report_vram("\nVRAM baseline (nothing loaded yet)")

    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks_by_id = {json.loads(line)["chunk_id"]: json.loads(line) for line in f}

    dense_results = dense_search(query, top_k=top_k)
    sparse_results = sparse_search(query, top_k=top_k)
    fused = rrf_fuse(dense_results, sparse_results, k=rrf_k)[:fused_top_n]
    candidate_chunks = [chunks_by_id[chunk_id] for chunk_id, _score in fused]

    reranked = rerank(query, candidate_chunks, top_n=rerank_top_n)
    top5_chunks = [chunk for chunk, _score in reranked]

    print("\nReranked top-5 evidence set (feeding the LLM):")
    for i, (chunk, score) in enumerate(reranked, start=1):
        print(f"  [{i}] score={score:.4f}  {chunk['paper_id']} / {chunk['section_title']}")

    # Reranker is already loaded (via rerank() above). Check VRAM before the
    # LLM adds its own footprint, then again after, to see the combined total.
    _report_vram("\nVRAM with reranker loaded (before LLM call)")

    backend = OllamaBackend(model="qwen2.5:3b-instruct")
    answer = backend.generate(query, top5_chunks)

    _report_vram("VRAM with reranker + LLM both resident (after LLM call)")

    print("\n" + "-" * 70)
    print("GENERATED ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()
