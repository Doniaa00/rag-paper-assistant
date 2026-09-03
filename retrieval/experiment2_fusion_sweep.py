"""
Step 10: RRF parameter sweep & fusion strategy comparison (Experiment 2).

Reuses dense_search/sparse_search from retrieval/rrf_fusion.py, rerank from
retrieval/rerank.py, and hit_rate/reciprocal_rank from
evaluation/retrieval_metrics.py to compare fusion strategies against the
current production default (RRF, k=60, top-20 per-retriever pool, fused
down to top-10 before reranking).

Arms tested, all scored on the same n=24 non-negative eval-set questions:
  1. RRF k-sweep: k = 10, 20, 40, 60 (current default), 100
  2. Weighted RRF: dense weighted 2x and 3x relative to sparse (k=60)
  3. Dense-only: no fusion at all, dense top-10 straight into the reranker
  4. Candidate-pool depth sweep (top-10/20/30 per retriever before fusing),
     run ONLY on whichever of arms 1-2 scores best -- not the full
     cross-product, per the task spec.

For every arm we score both the raw fusion/dense output AND that same
candidate set after reranking, since Step 9's findings showed reranking can
mask a weak fusion stage.

Efficiency note: dense_search/sparse_search are the only calls that hit the
BGE-M3 model / BM25 index, and both are deterministic given (query, top_k).
Rather than re-querying them per arm, each question's dense/sparse results
are fetched ONCE at the deepest pool size any arm needs (top-30) and then
sliced down for shallower arms -- 24 dense_search + 24 sparse_search calls
total for the whole sweep, not one pair per arm.

Writes the full comparison table to
data/evaluation/experiment2_fusion_results.csv.
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

from rrf_fusion import dense_search, sparse_search  # noqa: E402
from rerank import rerank  # noqa: E402
from evaluation.retrieval_metrics import hit_rate, reciprocal_rank  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
RESULTS_CSV = DATA_DIR / "evaluation" / "experiment2_fusion_results.csv"

MAX_POOL_DEPTH = 30   # deepest per-retriever pool any arm needs
DEFAULT_DEPTH = 20    # matches Step 9b's baseline dense/sparse top_k
FUSED_TOP_N = 10      # candidates handed to the reranker -- held constant
                      # across every fusion/dense-only arm so the sweep
                      # isolates fusion strategy / pool depth, not reranker
                      # input size
RERANK_TOP_N = 5
DEPTH_SWEEP_DEPTHS = [10, 20, 30]


def weighted_rrf_fuse(dense_results: list, sparse_results: list, k: int,
                       dense_weight: float = 1.0, sparse_weight: float = 1.0):
    """RRF generalized with per-arm weights. dense_weight=sparse_weight=1.0
    reduces to the standard unweighted RRF used elsewhere in the project."""
    scores = {}
    for rank, chunk_id in enumerate(dense_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight / (k + rank)
    for rank, chunk_id in enumerate(sparse_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def load_eval_questions():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["question_type"].strip() != "negative"]


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def base_arm_configs():
    return [
        {"name": "rrf_k10", "group": "rrf_k_sweep", "k": 10, "dense_weight": 1.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "rrf_k20", "group": "rrf_k_sweep", "k": 20, "dense_weight": 1.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "rrf_k40", "group": "rrf_k_sweep", "k": 40, "dense_weight": 1.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "rrf_k60_current_default", "group": "rrf_k_sweep", "k": 60, "dense_weight": 1.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "rrf_k100", "group": "rrf_k_sweep", "k": 100, "dense_weight": 1.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "weighted_dense2x_k60", "group": "weighted", "k": 60, "dense_weight": 2.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
        {"name": "weighted_dense3x_k60", "group": "weighted", "k": 60, "dense_weight": 3.0, "sparse_weight": 1.0, "depth": DEFAULT_DEPTH},
    ]


def compute_metrics(ids: list, source_paper_id: str, k5: int = 5, k10: int = 10):
    return {
        "hit5": hit_rate(ids, source_paper_id, k5),
        "hit10": hit_rate(ids, source_paper_id, k10),
        "rr": reciprocal_rank(ids, source_paper_id),
    }


def run_fusion_arm(query: str, dense_pool: list, sparse_pool: list, config: dict,
                    chunks_by_id: dict, source_paper_id: str):
    depth = config["depth"]
    dense_d = dense_pool[:depth]
    sparse_d = sparse_pool[:depth]
    fused_pairs = weighted_rrf_fuse(
        dense_d, sparse_d, k=config["k"],
        dense_weight=config["dense_weight"], sparse_weight=config["sparse_weight"],
    )[:FUSED_TOP_N]
    raw_ids = [cid for cid, _score in fused_pairs]

    candidate_chunks = [chunks_by_id[cid] for cid in raw_ids]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    reranked_ids = [c["chunk_id"] for c, _score in reranked_pairs]

    return {
        "raw": compute_metrics(raw_ids, source_paper_id),
        "reranked": compute_metrics(reranked_ids, source_paper_id),
    }


def run_dense_only_arm(query: str, dense_pool: list, chunks_by_id: dict, source_paper_id: str):
    raw_ids = dense_pool[:FUSED_TOP_N]
    candidate_chunks = [chunks_by_id[cid] for cid in raw_ids]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    reranked_ids = [c["chunk_id"] for c, _score in reranked_pairs]

    return {
        "raw": compute_metrics(raw_ids, source_paper_id),
        "reranked": compute_metrics(reranked_ids, source_paper_id),
    }


def aggregate(records: list, stage: str):
    n = len(records)
    hr5 = sum(r[stage]["hit5"] for r in records) / n
    hr10 = sum(r[stage]["hit10"] for r in records) / n
    mrr = sum(r[stage]["rr"] for r in records) / n
    return hr5, hr10, mrr


def pick_winner(per_config_records: dict, configs: list):
    """Winner among the RRF k-sweep and weighted-fusion arms only, by
    reranked Hit Rate@5 (primary) then reranked MRR (tie-break) -- these are
    the two arm families the depth sweep (arm 4) is run against, per spec."""
    candidate_names = [c["name"] for c in configs if c["group"] in ("rrf_k_sweep", "weighted")]

    def score(name):
        hr5, _hr10, mrr = aggregate(per_config_records[name], "reranked")
        return (hr5, mrr)

    return max(candidate_names, key=score)


def main():
    questions = load_eval_questions()
    chunks_by_id = load_chunks_by_id()
    print(f"Loaded {len(questions)} non-negative questions from {EVAL_SET_CSV.name}")

    configs = base_arm_configs()
    per_config_records = {c["name"]: [] for c in configs}
    dense_only_records = []
    pools_by_question = []  # cache (dense_pool, sparse_pool, source_paper_id) for the depth-sweep pass

    for i, q in enumerate(questions, start=1):
        qid = q["question_id"]
        query = q["question"]
        source_paper_id = q["source_paper_id"].strip()
        print(f"[{i}/{len(questions)}] {qid}: {query[:60]}...")

        dense_pool = dense_search(query, top_k=MAX_POOL_DEPTH)
        sparse_pool = sparse_search(query, top_k=MAX_POOL_DEPTH)
        pools_by_question.append((query, dense_pool, sparse_pool, source_paper_id))

        for config in configs:
            result = run_fusion_arm(query, dense_pool, sparse_pool, config, chunks_by_id, source_paper_id)
            per_config_records[config["name"]].append(result)

        dense_only_records.append(run_dense_only_arm(query, dense_pool, chunks_by_id, source_paper_id))

    winner_name = pick_winner(per_config_records, configs)
    winner_config = next(c for c in configs if c["name"] == winner_name)
    print(f"\nWinner of RRF k-sweep + weighted arms (by reranked HR@5, tie-break reranked MRR): {winner_name}")

    depth_sweep_rows = []
    for depth in DEPTH_SWEEP_DEPTHS:
        name = f"depth{depth}_{winner_name}"
        if depth == winner_config["depth"]:
            # Same fusion params at the same depth as the already-computed
            # winner row -- retrieval is deterministic, so this would
            # reproduce identical numbers. Reuse rather than recompute.
            depth_sweep_rows.append({
                "name": name, "group": "depth_sweep", "k": winner_config["k"],
                "dense_weight": winner_config["dense_weight"], "sparse_weight": winner_config["sparse_weight"],
                "depth": depth, "records": per_config_records[winner_name],
                "note": f"= {winner_name} (same depth, reused rather than recomputed)",
            })
            continue

        records = []
        depth_config = {**winner_config, "depth": depth}
        for query, dense_pool, sparse_pool, source_paper_id in pools_by_question:
            records.append(run_fusion_arm(query, dense_pool, sparse_pool, depth_config, chunks_by_id, source_paper_id))
        depth_sweep_rows.append({
            "name": name, "group": "depth_sweep", "k": winner_config["k"],
            "dense_weight": winner_config["dense_weight"], "sparse_weight": winner_config["sparse_weight"],
            "depth": depth, "records": records, "note": "",
        })

    write_results_csv(configs, per_config_records, dense_only_records, depth_sweep_rows, len(questions))
    print_report(configs, per_config_records, dense_only_records, depth_sweep_rows, winner_name, len(questions))


def _row_from_records(name, group, k, dense_weight, sparse_weight, depth, fused_top_n, records, n, note=""):
    raw_hr5, raw_hr10, raw_mrr = aggregate(records, "raw")
    rr_hr5, rr_hr10, rr_mrr = aggregate(records, "reranked")
    return {
        "config_name": name,
        "group": group,
        "rrf_k": k if k is not None else "",
        "dense_weight": dense_weight if dense_weight is not None else "",
        "sparse_weight": sparse_weight if sparse_weight is not None else "",
        "per_retriever_depth": depth,
        "fused_top_n": fused_top_n,
        "n_questions": n,
        "raw_hit_rate_at_5": round(raw_hr5, 4),
        "raw_hit_rate_at_10": round(raw_hr10, 4),
        "raw_mrr": round(raw_mrr, 4),
        "reranked_hit_rate_at_5": round(rr_hr5, 4),
        "reranked_hit_rate_at_10": round(rr_hr10, 4),
        "reranked_mrr": round(rr_mrr, 4),
        "note": note,
    }


def write_results_csv(configs, per_config_records, dense_only_records, depth_sweep_rows, n):
    fieldnames = [
        "config_name", "group", "rrf_k", "dense_weight", "sparse_weight",
        "per_retriever_depth", "fused_top_n", "n_questions",
        "raw_hit_rate_at_5", "raw_hit_rate_at_10", "raw_mrr",
        "reranked_hit_rate_at_5", "reranked_hit_rate_at_10", "reranked_mrr", "note",
    ]

    rows = []
    for config in configs:
        rows.append(_row_from_records(
            config["name"], config["group"], config["k"], config["dense_weight"], config["sparse_weight"],
            config["depth"], FUSED_TOP_N, per_config_records[config["name"]], n,
        ))
    rows.append(_row_from_records(
        "dense_only", "dense_only", None, None, None, FUSED_TOP_N, FUSED_TOP_N, dense_only_records, n,
        note="no fusion -- dense top-10 straight into reranker",
    ))
    for d in depth_sweep_rows:
        rows.append(_row_from_records(
            d["name"], d["group"], d["k"], d["dense_weight"], d["sparse_weight"],
            d["depth"], FUSED_TOP_N, d["records"], n, note=d["note"],
        ))

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_report(configs, per_config_records, dense_only_records, depth_sweep_rows, winner_name, n):
    print("\n" + "=" * 100)
    print("EXPERIMENT 2 -- FUSION STRATEGY SWEEP (Step 10)")
    print("=" * 100)
    print(f"\nQuestions scored: {n} (non-negative eval-set questions)")
    print(f"Reranker input fixed at top-{FUSED_TOP_N} candidates across every arm ({RERANK_TOP_N} kept after reranking).\n")

    header = f"{'Config':<28}{'k':>5}{'dW':>5}{'sW':>5}{'depth':>7}  |{'raw HR@5':>10}{'raw HR@10':>11}{'raw MRR':>9}  |{'rr HR@5':>10}{'rr HR@10':>10}{'rr MRR':>9}"
    print(header)
    print("-" * len(header))

    def print_group(name, group_configs_or_records):
        print(f"\n[{name}]")
        for entry in group_configs_or_records:
            print(entry)

    for group_label, group_key in [("RRF k-sweep", "rrf_k_sweep"), ("Weighted fusion", "weighted")]:
        print(f"\n-- {group_label} --")
        for config in configs:
            if config["group"] != group_key:
                continue
            records = per_config_records[config["name"]]
            raw_hr5, raw_hr10, raw_mrr = aggregate(records, "raw")
            rr_hr5, rr_hr10, rr_mrr = aggregate(records, "reranked")
            marker = "  <-- winner" if config["name"] == winner_name else ""
            print(f"{config['name']:<28}{config['k']:>5}{config['dense_weight']:>5.1f}{config['sparse_weight']:>5.1f}{config['depth']:>7}  |"
                  f"{raw_hr5:>10.1%}{raw_hr10:>11.1%}{raw_mrr:>9.3f}  |{rr_hr5:>10.1%}{rr_hr10:>10.1%}{rr_mrr:>9.3f}{marker}")

    print("\n-- Dense-only (no fusion) --")
    raw_hr5, raw_hr10, raw_mrr = aggregate(dense_only_records, "raw")
    rr_hr5, rr_hr10, rr_mrr = aggregate(dense_only_records, "reranked")
    print(f"{'dense_only':<28}{'--':>5}{'--':>5}{'--':>5}{FUSED_TOP_N:>7}  |"
          f"{raw_hr5:>10.1%}{raw_hr10:>11.1%}{raw_mrr:>9.3f}  |{rr_hr5:>10.1%}{rr_hr10:>10.1%}{rr_mrr:>9.3f}")

    print(f"\n-- Candidate-pool depth sweep (on winner: {winner_name}) --")
    for d in depth_sweep_rows:
        raw_hr5, raw_hr10, raw_mrr = aggregate(d["records"], "raw")
        rr_hr5, rr_hr10, rr_mrr = aggregate(d["records"], "reranked")
        note = f"  ({d['note']})" if d["note"] else ""
        print(f"{d['name']:<28}{d['k']:>5}{d['dense_weight']:>5.1f}{d['sparse_weight']:>5.1f}{d['depth']:>7}  |"
              f"{raw_hr5:>10.1%}{raw_hr10:>11.1%}{raw_mrr:>9.3f}  |{rr_hr5:>10.1%}{rr_hr10:>10.1%}{rr_mrr:>9.3f}{note}")

    print(f"\nFull comparison table written to: {RESULTS_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()
