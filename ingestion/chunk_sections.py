"""
Step 3: Section-aware chunking (baseline arm of Experiment 1).

Reads the 51 GROBID-parsed TEI/XML files in data/parsed/, splits each into
retrieval-ready chunks along section (<div>) boundaries, and writes them to
data/chunks/section_aware_chunks.jsonl (one JSON object per line).

Rules:
  - Each body <div> is one section. Sections <= ~450 tokens become a single
    chunk, even if short -- never merged with a neighboring section.
  - Sections > ~450 tokens are split into ~450-token chunks with ~65-token
    (~15%) overlap between consecutive chunks *within that section only*.
  - No chunk ever crosses a section boundary.
  - Figure/table captions are excluded (GROBID represents those via
    <figDesc>, not <p>, so only extracting <p> text already excludes them).
  - <listBibl>/references are excluded by construction: we only walk
    <text>/<body>/<div>, never <back>/<listBibl>.

Tokenization: tiktoken's cl100k_base encoding. This is an approximation
standing in for the BGE-M3 tokenizer (per the task's explicit allowance) --
actual sub-word boundaries will differ somewhat from BGE-M3's SentencePiece
vocabulary, but token *counts* are close enough for chunk-sizing purposes at
this baseline stage.

No embedding or indexing happens in this step.
"""

import csv
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import tiktoken

# Windows consoles default to cp1252, which can't encode some characters
# GROBID pulls out of math-heavy PDFs (e.g. stray U+FFFD replacement chars).
# Reports are diagnostic output, not data, so degrade gracefully instead of
# crashing the whole run over a single unprintable heading.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARSED_DIR = DATA_DIR / "parsed"
SHORTLIST_CSV = DATA_DIR / "shortlist_papers.csv"
OUTPUT_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

CHUNK_SIZE_TOKENS = 450
OVERLAP_TOKENS = 65  # ~15% of 450

ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Section:
    title: str
    text: str


def load_paper_metadata():
    """Returns {paper_id: {"title": ..., "pillar_tags": ...}}, matching TEI
    filename stems -- arxiv_id for 50 papers, or the filename-derived manual
    ID for the one row (Chandola) with no arxiv_id."""
    with open(SHORTLIST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_id = {}
    no_id_rows = []
    for row in rows:
        if row["arxiv_id"].strip():
            by_id[row["arxiv_id"].strip()] = row
        else:
            no_id_rows.append(row)

    tei_stems = {re.sub(r"\.tei\.xml$", "", p.name) for p in PARSED_DIR.glob("*.tei.xml")}
    unmatched_stems = tei_stems - set(by_id.keys())

    if len(unmatched_stems) != len(no_id_rows):
        logger.warning(
            "Metadata match mismatch: %d TEI files have no arxiv_id-based match, "
            "but %d shortlist rows have no arxiv_id. Manual review needed.",
            len(unmatched_stems), len(no_id_rows),
        )

    for stem, row in zip(sorted(unmatched_stems), no_id_rows):
        by_id[stem] = row

    return by_id


def extract_sections(tei_path: Path):
    root = ET.parse(tei_path).getroot()
    body_divs = root.findall("./tei:text/tei:body/tei:div", TEI_NS)

    sections = []
    for div in body_divs:
        head = div.find("tei:head", TEI_NS)
        title = (head.text or "").strip() if head is not None and head.text else "(untitled section)"

        paragraphs = div.findall(".//tei:p", TEI_NS)
        text_parts = []
        for p in paragraphs:
            p_text = "".join(p.itertext()).strip()
            p_text = re.sub(r"\s+", " ", p_text)
            if p_text:
                text_parts.append(p_text)

        text = "\n\n".join(text_parts)
        if text:
            sections.append(Section(title=title, text=text))

    return sections


def chunk_section_text(text: str):
    """Split token-encoded text into ~CHUNK_SIZE_TOKENS chunks with
    ~OVERLAP_TOKENS overlap. Returns a list of decoded text chunks."""
    tokens = ENCODING.encode(text)
    if len(tokens) <= CHUNK_SIZE_TOKENS:
        return [text]

    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
    chunks = []
    start = 0
    while start < len(tokens):
        window = tokens[start:start + CHUNK_SIZE_TOKENS]
        chunks.append(ENCODING.decode(window))
        if start + CHUNK_SIZE_TOKENS >= len(tokens):
            break
        start += step

    return chunks


def build_chunks_for_paper(paper_id: str, meta: dict, sections: list):
    chunks = []
    for section in sections:
        piece_texts = chunk_section_text(section.text)
        total = len(piece_texts)
        for i, piece_text in enumerate(piece_texts, start=1):
            chunks.append({
                "paper_id": paper_id,
                "paper_title": meta["title"],
                "section_title": section.title,
                "chunk_index_in_section": f"{i} of {total}",
                "pillar_tags": meta["pillar_tags"],
                "chunk_text": piece_text,
            })
    return chunks


def main():
    metadata = load_paper_metadata()
    tei_paths = sorted(PARSED_DIR.glob("*.tei.xml"))
    logger.info("Found %d TEI files, %d metadata rows", len(tei_paths), len(metadata))

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    per_paper_stats = []
    per_section_stats = []

    for tei_path in tei_paths:
        paper_id = re.sub(r"\.tei\.xml$", "", tei_path.name)
        meta = metadata.get(paper_id)
        if meta is None:
            logger.error("No shortlist metadata found for %s -- skipping", paper_id)
            continue

        sections = extract_sections(tei_path)
        chunks = build_chunks_for_paper(paper_id, meta, sections)
        all_chunks.extend(chunks)

        per_paper_stats.append((paper_id, meta["title"], len(sections), len(chunks)))
        for section in sections:
            n_pieces = len(chunk_section_text(section.text))
            per_section_stats.append((paper_id, section.title, n_pieces))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print_report(all_chunks, per_paper_stats, per_section_stats)


def print_report(all_chunks, per_paper_stats, per_section_stats):
    print("\n" + "=" * 70)
    print("SECTION-AWARE CHUNKING REPORT")
    print("=" * 70)

    token_counts = [len(ENCODING.encode(c["chunk_text"])) for c in all_chunks]
    print(f"\nTokenizer used: tiktoken cl100k_base (approximation for BGE-M3)")
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Total papers: {len(per_paper_stats)}")
    print(f"Average chunks per paper: {len(all_chunks) / len(per_paper_stats):.1f}")

    print(f"\nChunk size (tokens): avg={sum(token_counts)/len(token_counts):.1f}  "
          f"min={min(token_counts)}  max={max(token_counts)}")

    single = sum(1 for *_, n in per_section_stats if n == 1)
    split = sum(1 for *_, n in per_section_stats if n > 1)
    print(f"\nSections: {len(per_section_stats)} total -- {single} single-chunk, {split} sub-split")

    print("\nPer-paper chunk counts:")
    for paper_id, title, n_sections, n_chunks in per_paper_stats:
        print(f"  {paper_id}: {n_sections} sections -> {n_chunks} chunks  ({title[:60]})")

    avg_chunks = len(all_chunks) / len(per_paper_stats)
    low_outliers = [p for p in per_paper_stats if p[3] < avg_chunks * 0.4]
    if low_outliers:
        print(f"\nFLAG -- papers with suspiciously few chunks (< 40% of average, {avg_chunks*0.4:.1f}):")
        for paper_id, title, n_sections, n_chunks in low_outliers:
            print(f"  {paper_id}: only {n_chunks} chunks from {n_sections} sections  ({title[:60]})")
    else:
        print("\nNo papers flagged for suspiciously few chunks.")

    heavy_split = [s for s in per_section_stats if s[2] >= 5]
    if heavy_split:
        print(f"\nFLAG -- sections that needed 5+ sub-chunks:")
        for paper_id, section_title, n in sorted(heavy_split, key=lambda s: -s[2]):
            print(f"  {paper_id} / \"{section_title}\": {n} sub-chunks")
    else:
        print("\nNo sections needed 5+ sub-chunks.")

    print(f"\nOutput written to: {OUTPUT_JSONL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
