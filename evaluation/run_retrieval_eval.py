"""
Step 9b: Retrieval evaluation runner.

For each non-negative question in data/evaluation/eval_set.csv, runs all
four retrieval stages -- dense-only (top-20), sparse-only/BM25 (top-20),
RRF-fused (top-10), and reranked (top-5) -- reusing the existing functions
from retrieval/rrf_fusion.py and retrieval/rerank.py. Scores each stage with
Hit Rate@5, Hit Rate@10, MRR, and an exact-section-match rate (bonus signal,
not a pass/fail requirement), then prints a stage-comparison table and
writes full per-question detail to data/evaluation/retrieval_eval_results.csv.

negative-type questions are excluded -- they test the "insufficient
evidence" fallback at the generation stage, not retrieval quality.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from rrf_fusion import dense_search, sparse_search, rrf_fuse  # noqa: E402
from rerank import rerank  # noqa: E402
from evaluation.retrieval_metrics import hit_rate, reciprocal_rank  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
RESULTS_CSV = DATA_DIR / "evaluation" / "retrieval_eval_results.csv"

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FUSED_TOP_N = 10
RERANK_TOP_N = 5

STAGES = ["dense", "sparse", "fused", "reranked"]


def load_eval_questions():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["question_type"].strip() != "negative"]


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def run_all_stages(query: str, chunks_by_id: dict):
    dense_ids = dense_search(query, top_k=DENSE_TOP_K)
    sparse_ids = sparse_search(query, top_k=SPARSE_TOP_K)

    fused_pairs = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)[:FUSED_TOP_N]
    fused_ids = [chunk_id for chunk_id, _score in fused_pairs]

    candidate_chunks = [chunks_by_id[cid] for cid in fused_ids]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    reranked_ids = [chunk["chunk_id"] for chunk, _score in reranked_pairs]

    return {
        "dense": dense_ids,
        "sparse": sparse_ids,
        "fused": fused_ids,
        "reranked": reranked_ids,
    }


def exact_section_hit(stage_chunk_ids: list, source_paper_id: str, source_section: str, chunks_by_id: dict) -> bool:
    if not source_section:
        return False
    for chunk_id in stage_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk and chunk["paper_id"] == source_paper_id and chunk["section_title"] == source_section:
            return True
    return False


def main():
    questions = load_eval_questions()
    chunks_by_id = load_chunks_by_id()
    print(f"Loaded {len(questions)} non-negative questions from {EVAL_SET_CSV.name}")

    per_question_rows = []
    # per_stage_metrics[stage] = list of dicts with hit5, hit10, rr, section_hit
    per_stage_metrics = {stage: [] for stage in STAGES}
    all_stages_missed = []

    for i, q in enumerate(questions, start=1):
        qid = q["question_id"]
        query = q["question"]
        source_paper_id = q["source_paper_id"].strip()
        source_section = q["source_section"].strip()

        print(f"[{i}/{len(questions)}] {qid}: {query[:60]}...")
        stage_results = run_all_stages(query, chunks_by_id)

        question_all_missed = True
        for stage in STAGES:
            stage_ids = stage_results[stage]
            hit5 = hit_rate(stage_ids, source_paper_id, 5)
            hit10 = hit_rate(stage_ids, source_paper_id, 10)
            rr = reciprocal_rank(stage_ids, source_paper_id)
            first_hit_rank = int(round(1 / rr)) if rr > 0 else None
            section_hit = exact_section_hit(stage_ids, source_paper_id, source_section, chunks_by_id)

            if rr > 0:
                question_all_missed = False

            per_stage_metrics[stage].append({
                "hit5": hit5, "hit10": hit10, "rr": rr, "section_hit": section_hit,
                "has_section": bool(source_section), "question_type": q["question_type"].strip(),
            })

            per_question_rows.append({
                "question_id": qid,
                "question": query,
                "pillar": q["pillar"],
                "question_type": q["question_type"],
                "source_paper_id": source_paper_id,
                "source_section": source_section,
                "stage": stage,
                "hit_at_5": hit5,
                "hit_at_10": hit10,
                "first_hit_rank": first_hit_rank if first_hit_rank is not None else "",
                "reciprocal_rank": round(rr, 4),
                "exact_section_hit": section_hit,
            })

        if question_all_missed:
            all_stages_missed.append((qid, query, source_paper_id))

    write_results_csv(per_question_rows)
    print_report(per_stage_metrics, all_stages_missed)


def write_results_csv(rows: list):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question", "pillar", "question_type", "source_paper_id", "source_section",
        "stage", "hit_at_5", "hit_at_10", "first_hit_rank", "reciprocal_rank", "exact_section_hit",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_report(per_stage_metrics: dict, all_stages_missed: list):
    print("\n" + "=" * 78)
    print("RETRIEVAL EVALUATION -- STAGE COMPARISON")
    print("=" * 78)

    n = len(next(iter(per_stage_metrics.values())))
    print(f"\nQuestions scored: {n}\n")

    header = f"{'Stage':<12}{'Hit Rate@5':>14}{'Hit Rate@10':>14}{'MRR':>10}{'Exact Section Match':>22}"
    print(header)
    print("-" * len(header))
    for stage in STAGES:
        records = per_stage_metrics[stage]
        hr5 = sum(r["hit5"] for r in records) / n
        hr10 = sum(r["hit10"] for r in records) / n
        mrr = sum(r["rr"] for r in records) / n
        applicable = [r for r in records if r["has_section"]]
        section_rate = (sum(r["section_hit"] for r in applicable) / len(applicable)) if applicable else float("nan")
        print(f"{stage:<12}{hr5:>14.1%}{hr10:>14.1%}{mrr:>10.3f}{section_rate:>22.1%}")

    print("\nNote: 'fused' only has 10 candidates and 'reranked' only has 5, so their")
    print("Hit Rate@10 cannot exceed what's already in their (shorter) candidate list --")
    print("this is expected, not a bug. MRR is likewise computed only within each")
    print("stage's own candidate-pool depth (20/20/10/5), so it is not perfectly")
    print("apples-to-apples across stages with different pool sizes.")

    question_types = sorted({r["question_type"] for r in per_stage_metrics[STAGES[0]]})
    if len(question_types) > 1:
        print("\n" + "-" * 78)
        print("BREAKDOWN BY QUESTION TYPE")
        print("-" * 78)
        for qtype in question_types:
            type_records = {stage: [r for r in per_stage_metrics[stage] if r["question_type"] == qtype] for stage in STAGES}
            type_n = len(type_records[STAGES[0]])
            print(f"\n{qtype} (n={type_n}):")
            print(header)
            print("-" * len(header))
            for stage in STAGES:
                records = type_records[stage]
                hr5 = sum(r["hit5"] for r in records) / type_n
                hr10 = sum(r["hit10"] for r in records) / type_n
                mrr = sum(r["rr"] for r in records) / type_n
                applicable = [r for r in records if r["has_section"]]
                section_rate = (sum(r["section_hit"] for r in applicable) / len(applicable)) if applicable else float("nan")
                print(f"{stage:<12}{hr5:>14.1%}{hr10:>14.1%}{mrr:>10.3f}{section_rate:>22.1%}")

    if all_stages_missed:
        print(f"\nFLAG -- questions where ALL FOUR stages failed to retrieve the correct paper ({len(all_stages_missed)}):")
        for qid, query, source_paper_id in all_stages_missed:
            print(f"  {qid}: expected paper {source_paper_id!r} -- {query}")
    else:
        print("\nNo questions were missed by all four stages.")

    print(f"\nPer-question detail written to: {RESULTS_CSV}")
    print("=" * 78)


if __name__ == "__main__":
    main()
