"""
Step 10 follow-up: verify whether the post-rerank ties observed across
RRF k=10/60/100 in experiment2_fusion_sweep.py reflect a genuinely stable
fused ranking at this pool depth, or just different candidate sets that
happen to reduce to the same correctness outcome after reranking.

For 3 of the 24 non-negative eval-set questions -- one factual/high-
confidence hit, one comparative/high-confidence hit, and one comparative
question flagged in Step 9b as borderline at the fused stage (fused
hit@5=False, first_hit_rank=9, i.e. the correct paper was in the top-10
pool but RRF ranked it outside the top-5) -- this prints the actual top-10
fused chunk_id set for k=10, k=60, k=100 using the production rrf_fuse()
from rrf_fusion.py, diffs the three sets, and reports the reranked outcome
for each.
"""

import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from rrf_fusion import dense_search, sparse_search, rrf_fuse  # noqa: E402
from rerank import rerank  # noqa: E402
from evaluation.retrieval_metrics import paper_id_from_chunk_id, hit_rate  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

DEPTH = 20          # dense/sparse top_k per arm, matches the sweep's default depth
FUSED_TOP_N = 10
RERANK_TOP_N = 5
K_VALUES = [10, 60, 100]

# q002: factual, fused stage hit@5=True at rank 1 (confident, clean case)
# q012: comparative, fused stage hit@5=True at rank 1 (confident, clean case)
# q015: comparative, fused stage hit@5=FALSE at rank 9 (Step 9b borderline flag --
#       correct paper was in the top-10 pool but RRF ranked it 6th-10th)
QUESTION_IDS = ["q002", "q012", "q015"]


def load_questions():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["question_id"]: r for r in rows}


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def main():
    questions = load_questions()
    chunks_by_id = load_chunks_by_id()

    for qid in QUESTION_IDS:
        q = questions[qid]
        query = q["question"]
        source_paper_id = q["source_paper_id"].strip()

        print("=" * 100)
        print(f"{qid} ({q['question_type']}, {q['pillar']}): {query}")
        print(f"Expected source paper: {source_paper_id}")
        print("=" * 100)

        dense_ids = dense_search(query, top_k=DEPTH)
        sparse_ids = sparse_search(query, top_k=DEPTH)

        pool_lists = {}
        pool_sets = {}

        for k in K_VALUES:
            fused_pairs = rrf_fuse(dense_ids, sparse_ids, k=k)[:FUSED_TOP_N]
            fused_ids = [cid for cid, _score in fused_pairs]
            pool_lists[k] = fused_ids
            pool_sets[k] = set(fused_ids)

            candidate_chunks = [chunks_by_id[cid] for cid in fused_ids]
            reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
            reranked_ids = [c["chunk_id"] for c, _score in reranked_pairs]
            top_paper = paper_id_from_chunk_id(reranked_ids[0]) if reranked_ids else None
            correct_at_top = top_paper == source_paper_id
            hit5 = hit_rate(reranked_ids, source_paper_id, 5)

            print(f"\n-- k={k} --")
            print(f"Fused top-10 chunk_ids (rank order): {fused_ids}")
            print(f"Reranked top-5 chunk_ids: {reranked_ids}")
            print(f"Reranked #1 -> paper_id: {top_paper}  (correct: {correct_at_top})   reranked hit@5: {hit5}")

        print("\n-- SET COMPARISON (top-10 fused chunk_ids, membership only) --")
        set10, set60, set100 = pool_sets[10], pool_sets[60], pool_sets[100]
        if set10 == set60 == set100:
            print("IDENTICAL membership across k=10, 60, 100.")
            if pool_lists[10] == pool_lists[60] == pool_lists[100]:
                print("Order is ALSO identical -- genuinely stable ranking, not just membership.")
            else:
                print("Order DIFFERS between k values even though membership is identical.")
        else:
            print("Membership DIFFERS across k values.")
            only_in_10 = set10 - (set60 | set100)
            only_in_60 = set60 - (set10 | set100)
            only_in_100 = set100 - (set10 | set60)
            print(f"  Only in k=10 pool:  {sorted(only_in_10) or '(none)'}")
            print(f"  Only in k=60 pool:  {sorted(only_in_60) or '(none)'}")
            print(f"  Only in k=100 pool: {sorted(only_in_100) or '(none)'}")
            common = set10 & set60 & set100
            print(f"  Common to all three: {len(common)} / {FUSED_TOP_N} chunks")

        print()


if __name__ == "__main__":
    main()
