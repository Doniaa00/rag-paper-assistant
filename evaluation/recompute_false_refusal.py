"""
Step 12: recompute false_refusal across all 24 non-negative questions using
exact_section_hit (reranked stage) as the "retrieval succeeded" criterion,
replacing the original paper-level hit_at_5 criterion -- see
check_false_refusal's docstring in evaluation/generation_judge.py for why.

check_false_refusal is False by construction wherever the model didn't
refuse at all, regardless of retrieval_succeeded (refused AND
retrieval_succeeded). So only questions where check_refusal(answer) is
True can possibly change classification when the criterion changes --
across the 24 non-negative questions, that's exactly q015 and q016; this
script recomputes all 24 rather than special-casing those two, so the
result is a genuine re-run, not an assumption.

Rewrites data/evaluation/generation_eval_results.csv's false_refusal
column in place with the corrected values.
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.generation_judge import check_false_refusal  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
GENERATION_RESULTS_CSV = DATA_DIR / "evaluation" / "generation_eval_results.csv"
RETRIEVAL_RESULTS_CSV = DATA_DIR / "evaluation" / "retrieval_eval_results.csv"


def load_reranked_stage_fields():
    with open(RETRIEVAL_RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    reranked = [r for r in rows if r["stage"] == "reranked"]
    hit_at_5 = {r["question_id"]: r["hit_at_5"] == "True" for r in reranked}
    exact_section_hit = {r["question_id"]: r["exact_section_hit"] == "True" for r in reranked}
    return hit_at_5, exact_section_hit


def main():
    hit_at_5, exact_section_hit = load_reranked_stage_fields()

    with open(GENERATION_RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    changed = []
    for r in rows:
        qid = r["question_id"]
        if r["question_type"].strip() == "negative":
            continue

        old_value = r["false_refusal"] == "True"
        new_retrieval_succeeded = exact_section_hit.get(qid, False)
        new_value = check_false_refusal(r["answer"], r["question_type"].strip(), new_retrieval_succeeded)

        if new_value != old_value:
            changed.append((qid, old_value, new_value, hit_at_5.get(qid), exact_section_hit.get(qid)))
        r["false_refusal"] = str(new_value)

    fieldnames = list(rows[0].keys())
    with open(GENERATION_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    non_negative = [r for r in rows if r["question_type"].strip() != "negative"]
    negative = [r for r in rows if r["question_type"].strip() == "negative"]
    n_false_refusal_new = sum(1 for r in non_negative if r["false_refusal"] == "True")
    n_correct_negative_refusals = sum(1 for r in negative if r["refused"] == "True")

    print(f"Recomputed false_refusal for {len(non_negative)} non-negative questions using exact_section_hit.")
    print(f"\nClassification changes ({len(changed)}):")
    for qid, old, new, old_hit5, new_section in changed:
        print(f"  {qid}: false_refusal {old} -> {new}   (hit_at_5={old_hit5}, exact_section_hit={new_section})")

    print(f"\nNew false_refusal count (non-negative, n={len(non_negative)}): {n_false_refusal_new}")
    print(f"Correct refusals on true negatives (n={len(negative)}): {n_correct_negative_refusals}")

    total = len(non_negative) + len(negative)
    combined_correct = (len(non_negative) - n_false_refusal_new) + n_correct_negative_refusals
    print(f"\nCombined two-sided refusal correctness (recomputed): {combined_correct}/{total} ({combined_correct/total:.1%})")
    print(f"(Was 28/30 = 93.3% under the old hit_at_5 criterion.)")


if __name__ == "__main__":
    main()
