"""
Step 1a: Candidate paper discovery.

Queries the arXiv API across 12 predefined searches (grouped into three
pillars: fraud_detection, anomaly_detection, class_imbalance), restricted to
categories cs.LG / cs.AI / stat.ML, then cross-references each result against
Semantic Scholar to attach a citation count. Writes a deduplicated candidate
list to data/candidate_papers.csv for manual review.

No PDFs are downloaded in this step.
"""

import csv
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

ARXIV_CATEGORY_FILTER = "(cat:cs.LG OR cat:cs.AI OR cat:stat.ML)"
ARXIV_DELAY_SECONDS = 3
S2_BATCH_SIZE = 100
S2_BATCH_DELAY_SECONDS = 3
MAX_RESULTS_PER_QUERY = 30
LOW_YIELD_THRESHOLD = 5

OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "candidate_papers.csv"

# Queries are written in arXiv's search_query syntax (field-prefixed terms,
# quoted phrases, AND/OR/parentheses) rather than as free text, since the
# arXiv API requires field prefixes like all:/cat: to parse correctly.
QUERIES = [
    ("fraud_detection", 'all:"credit card fraud detection" AND (all:"deep learning" OR all:"machine learning")'),
    ("fraud_detection", 'all:"financial fraud detection" AND (all:"graph neural network" OR all:GNN)'),
    ("fraud_detection", 'all:"insurance fraud detection" AND all:"machine learning"'),
    ("fraud_detection", 'all:"fraud detection" AND all:survey AND (all:"deep learning" OR all:"machine learning")'),
    ("anomaly_detection", 'all:"anomaly detection" AND all:survey AND all:"deep learning"'),
    ("anomaly_detection", 'all:"anomaly detection" AND (all:autoencoder OR all:"one-class classification")'),
    ("anomaly_detection", 'all:"time series anomaly detection" AND all:"deep learning"'),
    ("anomaly_detection", 'all:"graph anomaly detection"'),
    ("class_imbalance", 'all:"class imbalance" AND all:classification AND all:survey'),
    ("class_imbalance", 'all:SMOTE AND (all:oversampling OR all:"imbalanced learning")'),
    ("class_imbalance", 'all:"cost-sensitive learning" AND all:classification'),
    ("class_imbalance", 'all:"imbalanced data" AND (all:"fraud detection" OR all:"anomaly detection")'),
]

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Candidate:
    arxiv_id: str
    title: str
    authors: str
    published_date: str
    pillar_tags: set = field(default_factory=set)
    citation_count: str = ""
    abstract: str = ""
    arxiv_url: str = ""


def query_arxiv(query: str, max_results: int = MAX_RESULTS_PER_QUERY, max_retries: int = 3):
    """Query the arXiv API and return a list of parsed entry dicts."""
    full_query = f"({query}) AND {ARXIV_CATEGORY_FILTER}"
    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return _parse_arxiv_response(resp.text)
        except (requests.RequestException, ET.ParseError) as exc:
            wait = 2 ** attempt
            logger.warning(
                "arXiv request failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
    logger.error("arXiv query permanently failed: %s", query)
    return None  # signals a hard failure, distinct from a genuine 0-result query


def _parse_arxiv_response(xml_text: str):
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
        arxiv_id = _extract_arxiv_id(raw_id)
        if not arxiv_id:
            continue
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        title = re.sub(r"\s+", " ", title)
        abstract = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        abstract = re.sub(r"\s+", " ", abstract)
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        published_date = published.split("T")[0] if published else ""
        authors = [
            a.findtext("atom:name", default="", namespaces=ATOM_NS)
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": "; ".join(a for a in authors if a),
            "published_date": published_date,
            "abstract": abstract,
            "arxiv_url": raw_id,
        })
    return entries


def _extract_arxiv_id(raw_id: str) -> str:
    """Pull the bare arXiv ID (no version suffix) out of an abs-page URL."""
    if not raw_id:
        return ""
    match = re.search(r"arxiv\.org/abs/([^v]+)", raw_id)
    if not match:
        return ""
    return match.group(1)


def get_citation_counts_batch(arxiv_ids: list, max_retries: int = 5):
    """Look up citation counts for a batch of arXiv IDs via Semantic Scholar's
    batch endpoint (one POST for up to S2_BATCH_SIZE papers, instead of one
    GET per paper -- this is what makes citation lookup tractable under S2's
    unauthenticated rate limits).

    Returns a dict {arxiv_id: citation_count_or_None}. IDs Semantic Scholar
    doesn't recognize come back as None (paper not indexed there).
    """
    s2_ids = [f"ARXIV:{aid}" for aid in arxiv_ids]
    result = dict.fromkeys(arxiv_ids)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                S2_BATCH_URL,
                params={"fields": "citationCount,externalIds"},
                json={"ids": s2_ids},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 10 * attempt
                logger.warning("Semantic Scholar batch rate limited. Waiting %ds.", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            for aid, entry in zip(arxiv_ids, data):
                if entry:
                    result[aid] = entry.get("citationCount")
            return result
        except (requests.RequestException, json.JSONDecodeError) as exc:
            wait = 5 * attempt
            logger.warning(
                "Semantic Scholar batch request failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
    logger.error("Semantic Scholar batch lookup permanently failed for a batch of %d papers", len(arxiv_ids))
    return result


def run_search():
    candidates = {}
    query_stats = []  # (index, pillar, query, result_count_or_None)
    s2_failures = []

    for i, (pillar, query) in enumerate(QUERIES, start=1):
        logger.info("Query %d/%d [%s]: %s", i, len(QUERIES), pillar, query)
        results = query_arxiv(query)
        query_stats.append((i, pillar, query, None if results is None else len(results)))
        for r in results or []:
            if r["arxiv_id"] in candidates:
                candidates[r["arxiv_id"]].pillar_tags.add(pillar)
            else:
                candidates[r["arxiv_id"]] = Candidate(
                    arxiv_id=r["arxiv_id"],
                    title=r["title"],
                    authors=r["authors"],
                    published_date=r["published_date"],
                    pillar_tags={pillar},
                    abstract=r["abstract"],
                    arxiv_url=r["arxiv_url"],
                )
        if i < len(QUERIES):
            time.sleep(ARXIV_DELAY_SECONDS)

    logger.info("Found %d unique candidates across all queries. Fetching citation counts...", len(candidates))

    all_ids = list(candidates.keys())
    batches = [all_ids[i:i + S2_BATCH_SIZE] for i in range(0, len(all_ids), S2_BATCH_SIZE)]
    for batch_num, batch_ids in enumerate(batches, start=1):
        logger.info("Semantic Scholar batch %d/%d (%d papers)", batch_num, len(batches), len(batch_ids))
        counts = get_citation_counts_batch(batch_ids)
        for arxiv_id, count in counts.items():
            if count is None:
                s2_failures.append(arxiv_id)
                candidates[arxiv_id].citation_count = ""
            else:
                candidates[arxiv_id].citation_count = str(count)
        if batch_num < len(batches):
            time.sleep(S2_BATCH_DELAY_SECONDS)

    return candidates, query_stats, s2_failures


def write_csv(candidates: dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arxiv_id", "title", "authors", "published_date",
            "pillar_tags", "citation_count", "abstract", "arxiv_url",
        ])
        for cand in sorted(candidates.values(), key=lambda c: c.arxiv_id):
            writer.writerow([
                cand.arxiv_id,
                cand.title,
                cand.authors,
                cand.published_date,
                ";".join(sorted(cand.pillar_tags)),
                cand.citation_count,
                cand.abstract,
                cand.arxiv_url,
            ])


def print_report(candidates, query_stats, s2_failures):
    print("\n" + "=" * 70)
    print("CANDIDATE PAPER DISCOVERY REPORT")
    print("=" * 70)
    print(f"\nTotal unique candidates: {len(candidates)}")

    pillar_counts = {}
    for cand in candidates.values():
        for tag in cand.pillar_tags:
            pillar_counts[tag] = pillar_counts.get(tag, 0) + 1
    print("\nBreakdown per pillar (a paper matching multiple queries counts in each):")
    for pillar, count in sorted(pillar_counts.items()):
        print(f"  {pillar}: {count}")

    print("\nPer-query result counts:")
    low_yield = []
    errored = []
    for i, pillar, query, n in query_stats:
        if n is None:
            print(f"  Q{i:02d} [{pillar}] (FAILED after retries): {query}")
            errored.append((i, pillar, query))
            continue
        flag = "  <-- LOW YIELD" if n < LOW_YIELD_THRESHOLD else ""
        print(f"  Q{i:02d} [{pillar}] ({n} results){flag}: {query}")
        if n < LOW_YIELD_THRESHOLD:
            low_yield.append((i, pillar, query, n))

    if low_yield:
        print(f"\nQueries with fewer than {LOW_YIELD_THRESHOLD} results (may need adjusting):")
        for i, pillar, query, n in low_yield:
            print(f"  Q{i:02d} [{pillar}]: {n} results - {query}")
    else:
        print("\nNo queries returned critically low result counts.")

    if errored:
        print(f"\nQueries that failed outright after retries ({len(errored)}):")
        for i, pillar, query in errored:
            print(f"  Q{i:02d} [{pillar}]: {query}")

    if s2_failures:
        print(f"\nSemantic Scholar citation lookups failed/not found for {len(s2_failures)} papers:")
        for aid in s2_failures[:20]:
            print(f"  {aid}")
        if len(s2_failures) > 20:
            print(f"  ... and {len(s2_failures) - 20} more")
    else:
        print("\nAll citation count lookups succeeded.")

    print(f"\nOutput written to: {OUTPUT_CSV}")
    print("=" * 70)


def main():
    candidates, query_stats, s2_failures = run_search()
    write_csv(candidates)
    print_report(candidates, query_stats, s2_failures)


if __name__ == "__main__":
    main()
