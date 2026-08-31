"""
Step 2b: Batch GROBID parsing of the full corpus.

Iterates over every PDF in data/papers/, sends each to GROBID's
/api/processFulltextDocument (same settings used in the Step 2a test:
consolidateHeader=1, includeRawCitations=1), and saves the TEI/XML to
data/parsed/{arxiv_id}.tei.xml.

A single paper's failure (timeout, malformed PDF, GROBID error) is logged
and skipped rather than halting the batch. One retry is attempted for
transient failures before giving up on a paper.

Tracks per-paper processing time, section (<div>) count, and reference
count for QA -- a paper with 0 sections or 0 references usually means the
PDF didn't parse cleanly and is worth a manual look.
"""

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from grobid_client import check_alive, process_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
PARSED_DIR = DATA_DIR / "parsed"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
MAX_ATTEMPTS = 2


@dataclass
class ParseResult:
    arxiv_id: str
    success: bool
    seconds: float = 0.0
    div_count: int = 0
    ref_count: int = 0
    reason: str = ""


def count_structure(tei_xml: str):
    root = ET.fromstring(tei_xml)
    div_count = len(root.findall(".//tei:text/tei:body/tei:div", TEI_NS))
    ref_count = len(root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS))
    return div_count, ref_count


def parse_one(pdf_path: Path) -> ParseResult:
    arxiv_id = pdf_path.stem
    dest = PARSED_DIR / f"{arxiv_id}.tei.xml"
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            tei_xml = process_pdf(pdf_path)
            elapsed = time.monotonic() - start
            div_count, ref_count = count_structure(tei_xml)
            dest.write_text(tei_xml, encoding="utf-8")
            return ParseResult(arxiv_id, True, elapsed, div_count, ref_count)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - start
            if attempt < MAX_ATTEMPTS:
                logger.warning("Parse failed for %s (attempt %d/%d, %.1fs): %s. Retrying...",
                                arxiv_id, attempt, MAX_ATTEMPTS, elapsed, last_error)

    return ParseResult(arxiv_id, False, reason=last_error)


def main():
    if not check_alive():
        raise SystemExit("GROBID is not reachable at http://localhost:8070. Is grobid-service running?")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdf_paths), PAPERS_DIR)

    results = []
    batch_start = time.monotonic()
    for i, pdf_path in enumerate(pdf_paths, start=1):
        logger.info("Processing %d/%d: %s", i, len(pdf_paths), pdf_path.name)
        result = parse_one(pdf_path)
        results.append(result)
        if result.success:
            logger.info("  OK in %.1fs -- %d sections, %d references",
                        result.seconds, result.div_count, result.ref_count)
        else:
            logger.error("  FAILED -- %s", result.reason)

    batch_seconds = time.monotonic() - batch_start
    print_report(results, batch_seconds)


def print_report(results, batch_seconds):
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print("\n" + "=" * 70)
    print("BATCH GROBID PARSING REPORT")
    print("=" * 70)

    print(f"\nTotal PDFs: {len(results)}")
    print(f"Succeeded: {len(successes)}")
    print(f"Failed: {len(failures)}")
    print(f"Total batch time: {batch_seconds:.1f}s ({batch_seconds/60:.1f} min)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  {r.arxiv_id}: {r.reason}")

    if successes:
        div_counts = [r.div_count for r in successes]
        ref_counts = [r.ref_count for r in successes]
        times = [r.seconds for r in successes]

        def stats(values):
            return (sum(values) / len(values), min(values), max(values))

        div_avg, div_min, div_max = stats(div_counts)
        ref_avg, ref_min, ref_max = stats(ref_counts)
        time_avg, time_min, time_max = stats(times)

        print("\nSection (<div>) count distribution:")
        print(f"  avg={div_avg:.1f}  min={div_min}  max={div_max}")
        print("\nReference count distribution:")
        print(f"  avg={ref_avg:.1f}  min={ref_min}  max={ref_max}")
        print("\nPer-paper processing time distribution:")
        print(f"  avg={time_avg:.1f}s  min={time_min:.1f}s  max={time_max:.1f}s")

        outliers = [r for r in successes if r.div_count == 0 or r.ref_count == 0]
        if outliers:
            print(f"\nOUTLIERS -- 0 sections or 0 references ({len(outliers)}), needs manual review:")
            for r in outliers:
                print(f"  {r.arxiv_id}: {r.div_count} sections, {r.ref_count} references")
        else:
            print("\nNo outliers with 0 sections or 0 references.")

    print(f"\nOutput written to: {PARSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
