"""
Step 12: prepare a human-vs-judge validation sample.

Selects 9 questions from the full 24-question Step 9c generation eval
(data/evaluation/generation_eval_results.csv) covering: both confirmed
unfaithful ratings, both false-refusal cases, and a mix of
fully_faithful / faithful_but_imprecise ratings across factual and
comparative question types. (q015 and q016 happen to be both the
confirmed-unfaithful cases AND the false-refusal cases -- the same two
questions satisfy both selection criteria, not four separate ones.)

Writes two files:
  - data/evaluation/human_validation_sample.md: question + full generated
    answer + full evidence chunks ONLY -- exactly the inputs
    judge_faithfulness itself was given, nothing more. No judge output of
    any kind (faithfulness/relevancy ratings or reasoning, citation
    checks, refusal flags) appears here, so it can be reviewed blind.
  - data/evaluation/human_validation_judge_answers.md: the judge's actual
    ratings/reasoning and the other automated checks for the same 9
    questions, meant to be opened only after forming an independent
    judgment from the first file.

Evidence chunks are reconstructed by re-running retrieval (dense/sparse/
fuse/rerank) rather than regenerating anything -- retrieval is
deterministic, and this is cross-checked below against each question's
stored evidence_paper_ids from the original Step 9c run.
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

DATA_DIR = PROJECT_ROOT / "data"
GENERATION_RESULTS_CSV = DATA_DIR / "evaluation" / "generation_eval_results.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
BLIND_MD = DATA_DIR / "evaluation" / "human_validation_sample.md"
JUDGE_MD = DATA_DIR / "evaluation" / "human_validation_judge_answers.md"

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60
FUSED_TOP_N = 10
RERANK_TOP_N = 5

# q015, q016: the two confirmed-unfaithful ratings AND the two false-refusal
#   cases (same two questions satisfy both criteria).
# q005, q011: faithful_but_imprecise, factual.
# q027: faithful_but_imprecise, comparative.
# q002, q009: fully_faithful, factual (fraud_detection / anomaly_detection).
# q012, q023: fully_faithful, comparative (fraud_detection / class_imbalance).
SELECTED_QIDS = ["q002", "q005", "q009", "q011", "q012", "q015", "q016", "q023", "q027"]


def load_generation_results():
    with open(GENERATION_RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["question_id"]: r for r in rows}


def load_chunks_by_id():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"]: json.loads(line) for line in f}


def retrieve_top5(query: str, chunks_by_id: dict):
    dense_ids = dense_search(query, top_k=DENSE_TOP_K)
    sparse_ids = sparse_search(query, top_k=SPARSE_TOP_K)
    fused_pairs = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)[:FUSED_TOP_N]
    candidate_chunks = [chunks_by_id[cid] for cid, _score in fused_pairs]
    reranked_pairs = rerank(query, candidate_chunks, top_n=RERANK_TOP_N)
    return [chunk for chunk, _score in reranked_pairs]


def main():
    results = load_generation_results()
    chunks_by_id = load_chunks_by_id()

    mismatch_warnings = []
    per_question_evidence = {}

    for i, qid in enumerate(SELECTED_QIDS, start=1):
        r = results[qid]
        print(f"[{i}/{len(SELECTED_QIDS)}] {qid}: {r['question'][:60]}...")
        evidence_chunks = retrieve_top5(r["question"], chunks_by_id)
        per_question_evidence[qid] = evidence_chunks

        reconstructed = [c["paper_id"] for c in evidence_chunks]
        stored = r["evidence_paper_ids"].split(";")
        if reconstructed != stored:
            mismatch_warnings.append((qid, stored, reconstructed))

    if mismatch_warnings:
        print(f"\nWARNING: {len(mismatch_warnings)} question(s) had reconstructed evidence that didn't match the stored CSV:")
        for qid, stored, reconstructed in mismatch_warnings:
            print(f"  {qid}: stored={stored} reconstructed={reconstructed}")
    else:
        print("\nRetrieval reconstruction check: all 9 questions' evidence matched the stored CSV exactly.")

    write_blind_file(results, per_question_evidence)
    write_judge_file(results, per_question_evidence)

    print(f"\nBlind review file written to: {BLIND_MD}")
    print(f"Judge-answers file written to: {JUDGE_MD}")


def _format_evidence_for_md(evidence_chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(evidence_chunks, start=1):
        parts.append(
            f"**Evidence {i}** -- paper_id: `{chunk['paper_id']}`, "
            f"paper_title: \"{chunk['paper_title']}\", section_title: \"{chunk['section_title']}\"\n\n"
            f"> {chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def write_blind_file(results: dict, per_question_evidence: dict):
    lines = [
        "# Human Validation Sample (Blind Review)",
        "",
        "9 questions from the Step 9c full-scale generation evaluation "
        "(data/evaluation/generation_eval_results.csv), selected to cover a "
        "range of question types and outcomes.",
        "",
        "For each question below: the question asked, the full generated "
        "answer, and the full evidence chunks the model was given -- "
        "exactly the inputs the faithfulness judge itself was given, "
        "nothing more. No judge rating, reasoning, or any other automated "
        "check result appears in this file. Form your own independent "
        "judgment of each answer's faithfulness to its evidence before "
        "opening `human_validation_judge_answers.md`.",
        "",
        "---",
    ]

    for i, qid in enumerate(SELECTED_QIDS, start=1):
        r = results[qid]
        evidence_chunks = per_question_evidence[qid]
        lines.append(f"\n## {i}. {qid} ({r['question_type']}, {r['pillar']})")
        lines.append(f"\n**Question:** {r['question']}")
        lines.append(f"\n**Generated answer:**\n\n{r['answer']}")
        lines.append(f"\n**Evidence given to the model:**\n\n{_format_evidence_for_md(evidence_chunks)}")
        lines.append("\n---")

    BLIND_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(BLIND_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_judge_file(results: dict, per_question_evidence: dict):
    lines = [
        "# Human Validation Sample -- Judge Answers (Open After Your Own Review)",
        "",
        "The faithfulness/relevancy judge's actual ratings and reasoning, "
        "plus the other automated checks, for the same 9 questions in "
        "`human_validation_sample.md`, in the same order. Open this only "
        "after forming an independent judgment from that file.",
        "",
        "---",
    ]

    for i, qid in enumerate(SELECTED_QIDS, start=1):
        r = results[qid]
        lines.append(f"\n## {i}. {qid} ({r['question_type']}, {r['pillar']})")
        lines.append(f"\n**Question:** {r['question']}")
        lines.append(f"\n**Source paper / section (ground truth):** `{r['source_paper_id']}` / \"{r['source_section']}\"")
        lines.append(f"\n**Faithfulness rating:** {r['faithfulness_rating']}")
        lines.append(f"\n**Faithfulness reasoning:** {r['faithfulness_reasoning']}")
        lines.append(f"\n**Relevancy rating:** {r['relevancy_rating']}")
        lines.append(f"\n**Relevancy reasoning:** {r['relevancy_reasoning']}")
        lines.append(f"\n**Citation format valid:** {r['citation_format_valid']}")
        if r["citation_format_issues"]:
            lines.append(f"\n**Citation format issues:** {r['citation_format_issues']}")
        lines.append(f"\n**Citation accuracy accurate:** {r['citation_accuracy_accurate']}")
        if r["citation_hallucinated"]:
            lines.append(f"\n**Hallucinated citations:** {r['citation_hallucinated']}")
        lines.append(f"\n**False refusal:** {r['false_refusal']}")
        lines.append("\n---")

    JUDGE_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
