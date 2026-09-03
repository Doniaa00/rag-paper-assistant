"""
Step 11: verify the citation-reliability guard (generation/citation_guard.py)
against the already-generated Step 9c full-scale answers.

Deliberately does NOT regenerate any answers -- the generated text in
data/evaluation/generation_eval_results.csv is reused as-is. What this
script DOES need that the CSV doesn't store is each question's actual
evidence_chunks (with section_title, not just paper_id -- the guard's
repair path matches on section_title), so retrieval is re-run for the 24
non-negative questions using the exact same parameters as
evaluation/run_generation_eval.py (DENSE_TOP_K=20, SPARSE_TOP_K=20,
RRF_K=60, FUSED_TOP_N=10, RERANK_TOP_N=5). Retrieval has no randomness
(dense embedding, BM25, RRF, and the cross-encoder reranker are all
deterministic), so this reproduces the exact evidence set each stored
answer was actually generated against -- cross-checked below against the
CSV's own evidence_paper_ids column as a sanity check.
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
from generation.citation_guard import validate_and_repair_citations  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
GENERATION_RESULTS_CSV = DATA_DIR / "evaluation" / "generation_eval_results.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FUSED_TOP_N = 10
RERANK_TOP_N = 5

HIGHLIGHT_QIDS = ["q014", "q017", "q018"]


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def load_generation_results():
    with open(GENERATION_RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["question_id"]: r for r in rows if r["question_type"].strip() != "negative"}


def load_eval_questions():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["question_id"]: r for r in rows if r["question_type"].strip() != "negative"}


def retrieve_top5(query: str, chunks_by_id: dict):
    dense_ids = dense_search(query, top_k=DENSE_TOP_K)
    sparse_ids = sparse_search(query, top_k=SPARSE_TOP_K)
    fused_pairs = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)[:FUSED_TOP_N]
    candidate_chunks = [chunks_by_id[cid] for cid, _score in fused_pairs]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    return [chunk for chunk, _score in reranked_pairs]


def main():
    questions = load_eval_questions()
    generation_results = load_generation_results()
    chunks_by_id = load_chunks_by_id()

    qids = list(questions.keys())
    print(f"Loaded {len(qids)} non-negative questions and their stored Step 9c answers.")

    total_repaired = 0
    total_stripped = 0
    questions_with_actions = 0
    highlight_results = {}
    mismatch_warnings = []

    for i, qid in enumerate(qids, start=1):
        q = questions[qid]
        query = q["question"]
        stored_answer = generation_results[qid]["answer"]
        stored_evidence_paper_ids = generation_results[qid]["evidence_paper_ids"].split(";")

        print(f"[{i}/{len(qids)}] {qid}: {query[:60]}...")
        evidence_chunks = retrieve_top5(query, chunks_by_id)

        reconstructed_paper_ids = [c["paper_id"] for c in evidence_chunks]
        if reconstructed_paper_ids != stored_evidence_paper_ids:
            mismatch_warnings.append((qid, stored_evidence_paper_ids, reconstructed_paper_ids))

        result = validate_and_repair_citations(stored_answer, evidence_chunks)
        n_repaired = sum(1 for a in result["actions_taken"] if a["action"] == "repaired")
        n_stripped = sum(1 for a in result["actions_taken"] if a["action"] == "stripped")
        total_repaired += n_repaired
        total_stripped += n_stripped
        if result["actions_taken"]:
            questions_with_actions += 1

        if qid in HIGHLIGHT_QIDS:
            highlight_results[qid] = {
                "before": stored_answer,
                "after": result["repaired_answer"],
                "actions": result["actions_taken"],
                "evidence_paper_ids": reconstructed_paper_ids,
            }

    print("\n" + "=" * 100)
    print("CITATION GUARD -- VERIFICATION AGAINST STEP 9c'S 24-QUESTION GENERATION EVAL")
    print("=" * 100)

    if mismatch_warnings:
        print(f"\nWARNING: {len(mismatch_warnings)} question(s) had evidence_paper_ids that didn't match the stored CSV")
        print("(retrieval may not be perfectly deterministic, or the index changed since Step 9c ran):")
        for qid, stored, reconstructed in mismatch_warnings:
            print(f"  {qid}: stored={stored}  reconstructed={reconstructed}")
    else:
        print("\nRetrieval reconstruction check: all 24 questions' re-run evidence_paper_ids matched the")
        print("stored CSV exactly -- confirms retrieval determinism and that the evidence sets used below")
        print("are identical to what Step 9c's answers were actually generated against.")

    print(f"\nQuestions scored: {len(qids)}")
    print(f"Questions with at least one guard action: {questions_with_actions}")
    print(f"Total repaired citations: {total_repaired}")
    print(f"Total stripped citations: {total_stripped}")

    for qid in HIGHLIGHT_QIDS:
        r = highlight_results[qid]
        print("\n" + "#" * 100)
        print(f"# {qid}  (evidence paper_ids: {sorted(set(r['evidence_paper_ids']))})")
        print("#" * 100)
        print(f"\nActions taken ({len(r['actions'])}):")
        for a in r["actions"]:
            if a["action"] == "repaired":
                print(f"  REPAIRED: {a['original']!r} -> {a['repaired_to']!r}")
                print(f"            reason: {a['reason']}")
            else:
                print(f"  STRIPPED: {a['original']!r}")
                print(f"            reason: {a['reason']}")
        print(f"\n--- BEFORE ---\n{r['before']}")
        print(f"\n--- AFTER ---\n{r['after']}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
