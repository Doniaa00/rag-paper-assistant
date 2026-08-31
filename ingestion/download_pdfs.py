"""
Step 2: Bulk PDF acquisition.

Reads data/shortlist_papers.csv, downloads the arXiv PDF for every row that
has an arxiv_id, into data/papers/{arxiv_id}.pdf. Rows without an arxiv_id
(e.g. the Chandola survey, sourced elsewhere) are skipped, as are rows whose
PDF is already present on disk (e.g. SMOTE, downloaded during Step 1b).

Each download is validated: PDF magic bytes, page count > 1, and a check
that it isn't an HTML error page saved with a .pdf extension. One retry on
transient failure; failures are logged and the batch continues.
"""

import csv
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHORTLIST_CSV = DATA_DIR / "shortlist_papers.csv"
PAPERS_DIR = DATA_DIR / "papers"

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_DELAY_SECONDS = 3
REQUEST_TIMEOUT = 60


def load_targets():
    with open(SHORTLIST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["arxiv_id"].strip()]


def validate_pdf(path: Path):
    """Returns (is_valid, reason)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return False, f"could not read file: {exc}"

    if not data.startswith(b"%PDF-"):
        return False, "missing %PDF- magic bytes (likely an HTML error page or corrupt download)"

    if b"<html" in data[:2000].lower() or b"<!doctype html" in data[:2000].lower():
        return False, "HTML content detected despite .pdf extension"

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception:
        # Fall back to a crude structural count if pypdf isn't available /
        # can't parse it; a PDF with a readable /Type/Pages /Count is enough.
        import re
        match = re.search(rb"/Type\s*/Pages.{0,200}?/Count\s+(\d+)", data, re.DOTALL)
        if match:
            page_count = int(match.group(1))
        else:
            return False, "could not determine page count (unparseable PDF structure)"

    if page_count <= 1:
        return False, f"page count is {page_count} (expected > 1)"

    return True, page_count


def download_one(arxiv_id: str, dest: Path, max_attempts: int = 2):
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "rag-paper-assistant/1.0"})
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            valid, info = validate_pdf(dest)
            if valid:
                return True, info
            last_error = info
            dest.unlink(missing_ok=True)
        except requests.RequestException as exc:
            last_error = str(exc)

        if attempt < max_attempts:
            logger.warning("Download failed for %s (attempt %d/%d): %s. Retrying...",
                            arxiv_id, attempt, max_attempts, last_error)
            time.sleep(ARXIV_DELAY_SECONDS)

    return False, last_error


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()

    already_present = []
    to_download = []
    for row in targets:
        dest = PAPERS_DIR / f"{row['arxiv_id']}.pdf"
        if dest.exists():
            valid, info = validate_pdf(dest)
            already_present.append((row["arxiv_id"], row["title"], valid, info))
        else:
            to_download.append(row)

    logger.info("%d rows with arxiv_id total; %d already on disk; %d to download",
                len(targets), len(already_present), len(to_download))

    successes = []
    failures = []

    for i, row in enumerate(to_download, start=1):
        arxiv_id = row["arxiv_id"]
        dest = PAPERS_DIR / f"{arxiv_id}.pdf"
        logger.info("Downloading %d/%d: %s (%s)", i, len(to_download), arxiv_id, row["title"])
        ok, info = download_one(arxiv_id, dest)
        if ok:
            successes.append((arxiv_id, row["title"], info))
        else:
            failures.append((arxiv_id, row["title"], info))
            logger.error("Permanently failed: %s -- %s", arxiv_id, info)

        if i < len(to_download):
            time.sleep(ARXIV_DELAY_SECONDS)

    print_report(already_present, successes, failures)


def print_report(already_present, successes, failures):
    print("\n" + "=" * 70)
    print("BULK PDF ACQUISITION REPORT")
    print("=" * 70)

    print(f"\nAlready present on disk (skipped download): {len(already_present)}")
    for aid, title, valid, info in already_present:
        status = f"valid, {info} pages" if valid else f"INVALID -- {info}"
        print(f"  {aid}: {status}  ({title})")

    print(f"\nNewly downloaded successfully: {len(successes)}")
    for aid, title, pages in successes:
        print(f"  {aid}: {pages} pages  ({title})")

    print(f"\nFailed downloads: {len(failures)}")
    for aid, title, reason in failures:
        print(f"  {aid}: {reason}  ({title})")

    total_valid = len(successes) + sum(1 for *_ , valid, _ in already_present if valid)
    print(f"\nTotal valid PDFs in data/papers/: {total_valid}")
    print(f"Total failures: {len(failures)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
