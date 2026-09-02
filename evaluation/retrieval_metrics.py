"""
Step 9b: Retrieval metrics.

Paper-level retrieval quality metrics, computed from a ranked list of
chunk_ids (rank 1 first) against the expected source_paper_id for a
question. Each chunk_id encodes its paper_id as
"{paper_id}_{3-digit zero-padded index}" (see Step 4's embed_chunks.py),
so paper_id is recovered directly from the chunk_id string rather than
requiring a separate lookup table.
"""

import re

_CHUNK_ID_PATTERN = re.compile(r"^(.+)_(\d{3})$")


def paper_id_from_chunk_id(chunk_id: str) -> str:
    match = _CHUNK_ID_PATTERN.match(chunk_id)
    if not match:
        raise ValueError(f"chunk_id {chunk_id!r} does not match the expected '{{paper_id}}_{{3 digits}}' format")
    return match.group(1)


def hit_rate(retrieved_chunk_ids: list, source_paper_id: str, k: int) -> bool:
    """True if any of the top-k retrieved chunks belong to source_paper_id."""
    top_k = retrieved_chunk_ids[:k]
    return any(paper_id_from_chunk_id(cid) == source_paper_id for cid in top_k)


def reciprocal_rank(retrieved_chunk_ids: list, source_paper_id: str) -> float:
    """1/rank of the first chunk belonging to source_paper_id, or 0 if absent."""
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if paper_id_from_chunk_id(chunk_id) == source_paper_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_chunk_ids: list, source_paper_id: str, k: int) -> bool:
    """Same logic as hit_rate at a specific k -- kept as a separate named
    function per the architecture doc's terminology (Hit Rate@k and
    Recall@k are the same computation at the single-relevant-document
    granularity used here)."""
    return hit_rate(retrieved_chunk_ids, source_paper_id, k)
