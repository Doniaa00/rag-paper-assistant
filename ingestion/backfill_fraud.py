"""
Step 1a.3: Fraud detection backfill.

The curated shortlist (data/shortlist_papers_45.csv) trimmed fraud_detection
from 22 to 17 papers by dropping off-topic entries, undercutting its intended
priority. This pulls replacement candidates from the full 292-paper archive
(data/candidate_papers.csv) -- no new searches -- filtered to papers tagged
fraud_detection (including multi-pillar tags) that aren't already in the
curated 45, ranked by the same recency-adjusted relevance score, top 10 out
to data/fraud_backfill_candidates.csv.

No PDF downloads in this step -- metadata/ranking only.
"""

import csv
from datetime import datetime
from pathlib import Path

from rank_candidates import compute_score, FIELDNAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDIDATES_CSV = DATA_DIR / "candidate_papers.csv"
CURATED_CSV = DATA_DIR / "shortlist_papers_45.csv"
OUTPUT_CSV = DATA_DIR / "fraud_backfill_candidates.csv"

TARGET_PILLAR = "fraud_detection"
BACKFILL_COUNT = 10


def load_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    now = datetime.now()
    candidates = load_rows(CANDIDATES_CSV)
    curated = load_rows(CURATED_CSV)
    curated_ids = {row["arxiv_id"] for row in curated}

    eligible = []
    for row in candidates:
        pillars = row["pillar_tags"].split(";")
        if TARGET_PILLAR not in pillars:
            continue
        if row["arxiv_id"] in curated_ids:
            continue
        score = compute_score(row, now)
        eligible.append((row, score))

    eligible.sort(key=lambda pair: pair[1], reverse=True)
    top10 = eligible[:BACKFILL_COUNT]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row, score in top10:
            out_row = dict(row)
            out_row["relevance_score"] = f"{score:.4f}"
            writer.writerow(out_row)

    overlap = [row["arxiv_id"] for row, _ in top10 if row["arxiv_id"] in curated_ids]

    print("\n" + "=" * 70)
    print("FRAUD DETECTION BACKFILL REPORT")
    print("=" * 70)
    print(f"\nEligible fraud_detection candidates outside the curated 45: {len(eligible)}")
    print(f"\nTop {BACKFILL_COUNT} backfill candidates:")
    for row, score in top10:
        print(f"  [{score:7.3f}] {row['title']}  ({row['arxiv_id']})")

    if overlap:
        print(f"\nWARNING: {len(overlap)} backfill picks duplicate an arxiv_id already in the 45-paper shortlist: {overlap}")
    else:
        print("\nConfirmed: none of the 10 backfill picks duplicate an arxiv_id already in the 45-paper shortlist.")

    print(f"\nOutput written to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
