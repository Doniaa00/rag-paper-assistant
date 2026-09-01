# Experiment Notes

Findings about retrieval *behavior* worth testing formally against the
architecture doc's experiment plan. Distinct from `known_limitations.md`,
which tracks data-quality defects (bad parses, garbled text) rather than
retrieval-design questions.

## RRF (k=60) can dilute a single strong dense hit

On the validation query "How does SMOTE generate synthetic samples?", dense
search's best result (cosine 0.7413, chunk `2402.17398_006`, "How SMOTE
Works" -- about as precise a match as this corpus has) fell from **dense
rank #1 to fused rank #10** after RRF (k=60) with BM25. It fell because BM25
never surfaced this chunk in its own top-20, so it received only a single
retriever's contribution (1/61 ≈ 0.0164), while several chunks that ranked
moderately in *both* lists (e.g. dense #5 + sparse #3) scored higher by
summing two partial contributions (≈0.0313).

This is RRF working as designed -- it rewards cross-retriever consensus over
a single strong signal from one retriever -- not a bug in `rrf_fuse`. But it
is a real trade-off: on this query, fusion actively made the top result
worse than dense alone. See [[bm25_generate_vocabulary_mismatch]] for why
BM25 missed the chunk in the first place.

**How to apply:** don't treat RRF's default k=60 as free of downside. Worth
weighing against dense-only or a dense-weighted fusion once a real
evaluation set exists (see below) -- not something to retune from a single
query's outcome.

## BM25 vocabulary-mismatch root cause {#bm25_generate_vocabulary_mismatch}

Investigated why BM25 ranked the best dense chunk (`2402.17398_006`) #43
instead of top-20. Two compounding, structural causes, both confirmed with
concrete token-level evidence (not from the same single query in isolation --
each was checked against the chunk's actual tokenized text and against the
BM25 index's term-frequency data for the competing top-ranked chunks):

1. **Inflection mismatch, no stemming.** The chunk uses "generat**ing**"
   twice but never the bare form "generate." The query tokenizes to
   `generate`. With no stemming in the BM25 tokenizer (lowercase + `\w+`
   only), these are entirely distinct vocabulary terms. All three of BM25's
   actual top-3 results for this query literally contain the token
   `generate` at least once -- this chunk does not.
2. **Low literal "SMOTE" term frequency in a well-paraphrased chunk.** The
   chunk names "SMOTE" explicitly only once, then refers to it indirectly
   ("the minority class is augmented by...") for the rest of the passage --
   despite being unambiguously about SMOTE throughout. BM25's competing
   top-ranked chunks repeat the literal word "smote" 3-6 times. BM25's
   term-frequency component has no way to credit a well-written paraphrase.

Both are structural properties of a simple no-stemming, pure keyword-
frequency BM25 baseline -- not implementation bugs, and not fixed here.

**How to apply:** worth testing formally rather than patching ad hoc off one
query:
- **Experiment 2 (hybrid retrieval):** does stemming/lemmatizing BM25's
  tokenizer (or swapping in a proper analyzer) measurably close this gap
  across the hand-written evaluation set, without regressing precision
  elsewhere?
- **Experiment 4 (query rewriting):** would expanding the query with
  inflected/synonym forms (generate → generating/generates; SMOTE → its
  expansion "Synthetic Minority Over-sampling Technique") recover chunks
  like this one without a BM25 tokenizer change?

Hold off on tuning k, the tokenizer, or fusion weights until the
hand-written evaluation set exists -- these findings are motivation for
what to test, not conclusions to act on from a single query.
