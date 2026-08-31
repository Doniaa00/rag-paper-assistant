"""
Step 1a.2: Rank & shortlist corpus candidates.

Reads data/candidate_papers.csv (the full 292-paper candidate pool from
search_corpus.py), computes a recency-adjusted relevance score per paper,
resolves multi-pillar papers to a single pillar, and takes the top N per
pillar into data/shortlist_papers.csv.

No PDF downloads in this step -- metadata/ranking only.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "candidate_papers.csv"
OUTPUT_CSV = DATA_DIR / "shortlist_papers.csv"

RECENCY_FLOOR_MONTHS = 18
RECENCY_FLOOR_SCORE = 0.5

PILLAR_QUOTAS = {
    "fraud_detection": 22,
    "class_imbalance": 18,
    "anomaly_detection": 15,
}

FIELDNAMES = [
    "arxiv_id", "title", "authors", "published_date",
    "pillar_tags", "citation_count", "abstract", "arxiv_url",
    "relevance_score",
]


@dataclass
class Paper:
    row: dict
    pillars: list
    score: float
    assigned_pillar: str = ""


def load_candidates():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_score(row: dict, now: datetime) -> float:
    citation_count = int(row["citation_count"]) if row["citation_count"] else 0
    published = datetime.strptime(row["published_date"], "%Y-%m-%d")
    years_since = (now - published).days / 365.25
    score = citation_count / max(1, years_since)
    months_since = years_since * 12
    if months_since <= RECENCY_FLOOR_MONTHS:
        score = max(score, RECENCY_FLOOR_SCORE)
    return score


def resolve_pillar_assignment(papers: list):
    """Rank each pillar's tagged papers by score, then send every multi-tagged
    paper to whichever of its pillars it ranks highest in (best relative
    standing), so it counts toward exactly one pillar's quota."""
    pillar_pools = {p: [] for p in PILLAR_QUOTAS}
    for paper in papers:
        for pillar in paper.pillars:
            pillar_pools[pillar].append(paper)

    pillar_rank = {}  # (id(paper), pillar) -> rank, 1 = best
    for pillar, pool in pillar_pools.items():
        pool.sort(key=lambda p: p.score, reverse=True)
        for rank, paper in enumerate(pool, start=1):
            pillar_rank[(id(paper), pillar)] = rank

    reassignments = []
    for paper in papers:
        if len(paper.pillars) == 1:
            paper.assigned_pillar = paper.pillars[0]
            continue
        best_pillar = min(paper.pillars, key=lambda pl: pillar_rank[(id(paper), pl)])
        paper.assigned_pillar = best_pillar
        reassignments.append((paper, best_pillar))

    return reassignments


def build_shortlist(papers: list):
    by_pillar = {p: [] for p in PILLAR_QUOTAS}
    for paper in papers:
        by_pillar[paper.assigned_pillar].append(paper)

    shortlist = {}
    for pillar, quota in PILLAR_QUOTAS.items():
        pool = sorted(by_pillar[pillar], key=lambda p: p.score, reverse=True)
        shortlist[pillar] = pool[:quota]

    return shortlist


def write_csv(shortlist: dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for pillar in PILLAR_QUOTAS:
            for paper in shortlist[pillar]:
                out_row = dict(paper.row)
                out_row["relevance_score"] = f"{paper.score:.4f}"
                writer.writerow(out_row)


def print_report(shortlist: dict, reassignments: list):
    print("\n" + "=" * 70)
    print("SHORTLIST REPORT")
    print("=" * 70)

    total = sum(len(v) for v in shortlist.values())
    print(f"\nFinal shortlist count: {total}")

    print("\nPer-pillar count achieved vs. target:")
    for pillar, quota in PILLAR_QUOTAS.items():
        achieved = len(shortlist[pillar])
        flag = "  <-- SHORT (overlap dedup reduced the eligible pool)" if achieved < quota else ""
        print(f"  {pillar}: {achieved}/{quota}{flag}")

    for pillar in PILLAR_QUOTAS:
        print(f"\nTop 5 in {pillar}:")
        for paper in shortlist[pillar][:5]:
            print(f"  [{paper.score:7.3f}] {paper.row['title']}")

    if reassignments:
        print(f"\nMulti-pillar papers resolved to a single pillar ({len(reassignments)}):")
        for paper, winner in reassignments:
            other_pillars = [p for p in paper.pillars if p != winner]
            print(f"  {paper.row['arxiv_id']} ({'/'.join(paper.pillars)}) -> counted under '{winner}', "
                  f"dropped from {other_pillars}")
    else:
        print("\nNo multi-pillar papers needed resolution.")

    print(f"\nOutput written to: {OUTPUT_CSV}")
    print("=" * 70)


def main():
    now = datetime.now()
    rows = load_candidates()

    papers = []
    for row in rows:
        pillars = row["pillar_tags"].split(";")
        score = compute_score(row, now)
        papers.append(Paper(row=row, pillars=pillars, score=score))

    reassignments = resolve_pillar_assignment(papers)
    shortlist = build_shortlist(papers)
    write_csv(shortlist)
    print_report(shortlist, reassignments)


if __name__ == "__main__":
    main()
