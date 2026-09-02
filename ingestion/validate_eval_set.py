"""
Step 9a: Evaluation set validator.

Run this after any manual edit to data/evaluation/eval_set.csv. Checks:
  - no duplicate question_ids
  - every non-blank source_paper_id exists (arxiv_id in data/shortlist_papers.csv,
    or the documented manual-ID exception "chandola2007_anomaly_survey")
  - every non-blank source_section exists as a real section_title for that
    exact paper_id in data/chunks/section_aware_chunks.jsonl
  - pillar and question_type values are from the allowed sets
  - negative questions have blank/N/A source_paper_id and source_section
    (consistency check, not just a formatting nit)

Also reports current progress: total questions, breakdown by pillar and by
question_type, so progress toward the 30-50 target is visible at a glance.
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_SET_CSV = DATA_DIR / "evaluation" / "eval_set.csv"
SHORTLIST_CSV = DATA_DIR / "shortlist_papers.csv"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

ALLOWED_PILLARS = {"fraud_detection", "class_imbalance", "anomaly_detection", "cross_pillar", "N/A"}
ALLOWED_QUESTION_TYPES = {"factual", "comparative", "negative"}

# The one documented manual-ID exception (no arxiv_id -- see eval_set.csv's
# source_paper_id column description and docs/background_reading.md).
MANUAL_PAPER_IDS = {"chandola2007_anomaly_survey"}

BLANK_MARKERS = {"", "n/a", "na", "none"}


def is_blank(value: str) -> bool:
    return value.strip().lower() in BLANK_MARKERS


def load_eval_rows():
    with open(EVAL_SET_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_valid_paper_ids():
    with open(SHORTLIST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = {row["arxiv_id"].strip() for row in rows if row["arxiv_id"].strip()}
    return ids | MANUAL_PAPER_IDS


def load_sections_by_paper():
    sections_by_paper = {}
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            sections_by_paper.setdefault(chunk["paper_id"], set()).add(chunk["section_title"])
    return sections_by_paper


def validate(rows, valid_paper_ids, sections_by_paper):
    errors = []

    seen_ids = {}
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        qid = row["question_id"].strip()
        if qid in seen_ids:
            errors.append(f"Row {i}: duplicate question_id {qid!r} (also at row {seen_ids[qid]})")
        else:
            seen_ids[qid] = i

        pillar = row["pillar"].strip()
        if pillar not in ALLOWED_PILLARS:
            errors.append(f"Row {i} ({qid}): invalid pillar {pillar!r}, must be one of {sorted(ALLOWED_PILLARS)}")

        qtype = row["question_type"].strip()
        if qtype not in ALLOWED_QUESTION_TYPES:
            errors.append(f"Row {i} ({qid}): invalid question_type {qtype!r}, must be one of {sorted(ALLOWED_QUESTION_TYPES)}")

        source_paper_id = row["source_paper_id"].strip()
        source_section = row["source_section"].strip()

        if qtype == "negative":
            if not is_blank(source_paper_id):
                errors.append(f"Row {i} ({qid}): question_type is 'negative' but source_paper_id is set to {source_paper_id!r} (should be blank/N/A)")
            if not is_blank(source_section):
                errors.append(f"Row {i} ({qid}): question_type is 'negative' but source_section is set to {source_section!r} (should be blank/N/A)")
        elif pillar == "N/A":
            errors.append(f"Row {i} ({qid}): pillar is 'N/A' but question_type is {qtype!r}, not 'negative' -- N/A pillar is only valid for negative questions")

        if not is_blank(source_paper_id):
            if source_paper_id not in valid_paper_ids:
                errors.append(f"Row {i} ({qid}): source_paper_id {source_paper_id!r} not found in {SHORTLIST_CSV.name}")
            elif not is_blank(source_section):
                valid_sections = sections_by_paper.get(source_paper_id, set())
                if source_section not in valid_sections:
                    errors.append(
                        f"Row {i} ({qid}): source_section {source_section!r} not found for paper_id "
                        f"{source_paper_id!r} in {CHUNKS_JSONL.name}"
                    )

    return errors


def print_report(rows, errors):
    print("\n" + "=" * 70)
    print("EVALUATION SET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal questions: {len(rows)}  (target: 30-50)")

    pillar_counts = Counter(row["pillar"].strip() for row in rows)
    print("\nBreakdown by pillar:")
    if pillar_counts:
        for pillar in sorted(pillar_counts):
            print(f"  {pillar}: {pillar_counts[pillar]}")
    else:
        print("  (no rows yet)")

    type_counts = Counter(row["question_type"].strip() for row in rows)
    print("\nBreakdown by question_type:")
    if type_counts:
        for qtype in sorted(type_counts):
            print(f"  {qtype}: {type_counts[qtype]}")
    else:
        print("  (no rows yet)")

    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nNo validation errors.")

    print("=" * 70)


def main():
    rows = load_eval_rows()
    valid_paper_ids = load_valid_paper_ids()
    sections_by_paper = load_sections_by_paper()

    errors = validate(rows, valid_paper_ids, sections_by_paper)
    print_report(rows, errors)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
