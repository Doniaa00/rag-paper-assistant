"""
Step 5 (sparse arm): Build the BM25 index.

Reads all 1,834 chunks' chunk_text from data/chunks/section_aware_chunks.jsonl
and builds a rank_bm25 (BM25Okapi) index over them.

Tokenization: lowercase, then split on a simple word-character regex
(\\w+ -- letters, digits, underscore). This is intentionally basic (no
stemming, no stopword removal) to keep the sparse arm a straightforward,
inspectable baseline alongside the dense arm.

Saves:
  - data/indexes/bm25_index.pkl   : the pickled BM25Okapi object
  - data/indexes/bm25_mapping.jsonl : one line per index position, in the
    same order as the corpus fed to BM25Okapi, {position, chunk_id} -- lets
    a BM25 result (a corpus position) be traced back to its chunk_id.
"""

import json
import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_JSONL = DATA_DIR / "chunks" / "section_aware_chunks.jsonl"
INDEXES_DIR = DATA_DIR / "indexes"
BM25_INDEX_PKL = INDEXES_DIR / "bm25_index.pkl"
BM25_MAPPING_JSONL = INDEXES_DIR / "bm25_mapping.jsonl"

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str):
    """Lowercase, then split on \\w+ (word characters: letters/digits/_)."""
    return TOKEN_PATTERN.findall(text.lower())


def main():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    logger.info("Loaded %d chunks from %s", len(chunks), CHUNKS_JSONL)

    tokenized_corpus = [tokenize(c["chunk_text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("Built BM25Okapi index over %d documents", len(tokenized_corpus))

    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PKL, "wb") as f:
        pickle.dump(bm25, f)

    with open(BM25_MAPPING_JSONL, "w", encoding="utf-8") as f:
        for position, chunk in enumerate(chunks):
            f.write(json.dumps({"position": position, "chunk_id": chunk["chunk_id"]}, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("BM25 INDEX BUILD REPORT")
    print("=" * 70)
    print(f"\nChunks indexed: {len(chunks)}")
    print(f"Tokenization: lowercase + \\w+ regex (no stemming, no stopword removal)")
    print(f"Index saved to: {BM25_INDEX_PKL}")
    print(f"Mapping saved to: {BM25_MAPPING_JSONL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
