"""
Step 2a: GROBID client.

Sends a single PDF to a running GROBID service's /api/processFulltextDocument
endpoint and saves the returned TEI/XML. Assumes GROBID is already running
locally (e.g. `docker run -d -p 8070:8070 grobid/grobid:<tag>`).

This is a one-PDF-at-a-time client for validating the GROBID setup before
batch-processing the full corpus.
"""

import argparse
import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GROBID_BASE_URL = "http://localhost:8070"
PROCESS_FULLTEXT_ENDPOINT = f"{GROBID_BASE_URL}/api/processFulltextDocument"
ALIVE_ENDPOINT = f"{GROBID_BASE_URL}/api/isalive"

REQUEST_TIMEOUT = 300  # GROBID fulltext parsing can be slow on long PDFs


def check_alive() -> bool:
    """Ping GROBID's health check endpoint."""
    try:
        resp = requests.get(ALIVE_ENDPOINT, timeout=10)
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.error("GROBID health check failed: %s", exc)
        return False


def process_pdf(pdf_path: Path) -> str:
    """Send a PDF to GROBID's processFulltextDocument endpoint and return the
    TEI/XML response body as text."""
    with open(pdf_path, "rb") as f:
        files = {"input": (pdf_path.name, f, "application/pdf")}
        data = {
            "consolidateHeader": "1",
            "consolidateCitations": "0",
            "includeRawCitations": "1",
        }
        resp = requests.post(
            PROCESS_FULLTEXT_ENDPOINT,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
    resp.raise_for_status()
    return resp.text


def process_to_file(pdf_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tei_xml = process_pdf(pdf_path)
    output_path.write_text(tei_xml, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Send one PDF to GROBID and save the TEI/XML output.")
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF")
    parser.add_argument("output_path", type=Path, help="Path to write the TEI/XML output")
    args = parser.parse_args()

    if not check_alive():
        raise SystemExit(f"GROBID is not reachable at {GROBID_BASE_URL}. Is the container running?")

    logger.info("GROBID is alive. Processing %s ...", args.pdf_path)
    out = process_to_file(args.pdf_path, args.output_path)
    logger.info("Saved TEI/XML to %s", out)


if __name__ == "__main__":
    main()
