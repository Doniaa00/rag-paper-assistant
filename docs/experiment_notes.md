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

## Generation: faithful-but-lossy paraphrase, not hallucination

Step 8's first real generation test (query: "How does SMOTE generate
synthetic samples?", `qwen2.5:3b-instruct`, grounded on the reranked top-5)
produced a correctly-cited, well-grounded answer. Both citations pointed to
the chunk that actually supports the claim, and most of the answer was a
close paraphrase of the source.

One deviation worth flagging precisely: the source (`2402.17398_006`) says a
*subset* of the K (or five) nearest neighbors is randomly chosen depending
on the oversampling percentage (e.g., 3 of 5). The model's answer simplified
this to "SMOTE randomly selects K neighbors from its set of closest
neighbors" -- true in spirit, imprecise in detail. This is not a
hallucination (it invents no fact, cites correctly, stays within the
evidence's topic) -- it's a faithfulness/precision distinction: the claim is
grounded but not exact.

**How to apply:** this is a concrete, real example of a failure mode the
Step 9 evaluation framework needs a way to *distinguish*, not just detect.
A binary "grounded / hallucinated" check would score this answer as fully
grounded and miss the simplification entirely. Worth designing a rubric
dimension (or a "faithful but imprecise" category, separate from
"unsupported claim") that can catch this kind of drift, since a 3B model
doing this on a simple mechanical description suggests it'll happen more on
harder questions.

## Methodology reminder: Ollama cold-start vs. steady-state timing

The first `qwen2.5:3b-instruct` call after `ollama pull` took 126s total
(61.7s model load + 50.4s prompt eval for 196 tokens) -- alarmingly slow on
its face. A second call on the warm model completed in under a second
(0.34s prompt eval, ~50 tokens/sec generation), confirming the first call's
latency was CUDA context/kernel warm-up plus disk load, not steady-state
performance.

**How to apply:** Step 9's latency measurements must warm up the model with
a throwaway call before timing anything, and cold-start latency should be
reported separately (if at all) rather than folded into averaged latency
stats -- a single cold call would blow up any mean/percentile figure by two
orders of magnitude and misrepresent real query latency.

## Retrieval eval n=12 -> n=17: dense's apparent perfection was phrasing-dependent

The first retrieval eval run (12 questions, all from earlier batches phrased
close to their source section titles) showed dense at 100% Hit Rate@5/@10.
Growing the eval set to 17 -- adding questions deliberately paraphrased away
from source vocabulary -- dropped dense to **94.1%**, entirely due to one
question (`q015`, source `2503.13195`): a comparative question about
traditional-vs-deep-learning anomaly detection paradigms, phrased with no
lexical overlap with the source section's language. Dense missed it
completely -- not just outside top-5, absent from its entire top-20
(`reciprocal_rank = 0.0`).

This confirms, with a concrete before/after number, what was previously
only a design concern: **the n=12 dense numbers were inflated by phrasing
proximity to source text**, not a true measure of dense retrieval's
semantic-matching ceiling. Growing the eval set with intentionally-distant
phrasing is doing its job -- it's surfacing a real weakness that easier
questions hid.

**q015 also inverts the corpus's usual retrieval story.** BM25 (sparse)
found the correct paper at rank 4 on the same query where dense found
nothing at all. Every prior finding in this file was "dense is strong,
sparse is the weak link" -- q015 is the first concrete case of the reverse,
and it's exactly the scenario hybrid retrieval is supposed to justify:
BM25's literal term-matching caught something dense's semantic embedding
missed. This is no longer a theoretical argument for keeping sparse in the
pipeline -- it's one real, reproducible example.

**RRF fusion dilution is now reproduced 3x independently**: the original
SMOTE validation query (dense #1 -> fused #10), `q006` (fused rank 9), and
now `q001` (fused rank 6). Three separate questions, same failure shape --
this has crossed the line from "one query's fluke" to a real, repeatable
characteristic of RRF at k=60 on this corpus.

**How to apply:**
- Don't treat any single-run retrieval metric (especially dense's) as
  settled until the eval set is phrasing-diverse -- rerun this comparison
  again as the set grows toward 30-50 to see if the trend continues.
- The q015 inversion is concrete evidence (not just justification-by-design)
  that the hybrid dense+sparse approach earns its complexity -- worth citing
  directly when someone asks "why not just use dense retrieval alone."
- RRF dilution at k=60 is now well-evidenced enough to formally test
  alternatives (smaller k, dense-weighted fusion) in Experiment 2/3, once
  the full eval set and generation-quality metrics exist -- not before.
- Sample size caveat still applies (n=17, one dense miss) -- but three
  independently-reproduced instances of the same fusion-dilution shape, plus
  a clean directional confirmation of the phrasing-proximity hypothesis, is
  meaningfully stronger evidence than the single-query anecdotes these notes
  started from.

## Final retrieval baseline (n=24): reranking is load-bearing for comparative questions

This is the final retrieval baseline snapshot, run against the complete
30-question hand-written eval set (24 non-negative) -- distinct from the
earlier n=12 and n=17 exploratory runs, which were run mid-construction to
sanity-check the pipeline and catch phrasing-proximity bias as the set grew.
This run's numbers are what should feed Table 2/3.

**Headline finding: comparative questions cause severe degradation in every
retrieval-only stage, and reranking fully compensates for it.**

| | dense HR@5 | sparse HR@5 | fused HR@5 | reranked HR@5 |
|---|---|---|---|---|
| factual (n=15) | 100.0% | 93.3% | 93.3% | 93.3% |
| comparative (n=9) | 77.8% | 66.7% | 55.6% | **100.0%** |

Every retrieval-only stage drops sharply on comparative questions relative
to factual (dense -22.2pp, sparse -26.6pp, fused -37.7pp -- fused is hit
hardest, worse than either individual retriever it's built from). Reranking
not only recovers all of this, it scores *higher* on comparative (100%) than
on factual (93.3%) -- the only stage where that's true.

This is not "reranking generally helps a bit" -- reranking is **specifically
load-bearing for comparative-question performance**, compensating for a
failure mode that's near-universal across dense, sparse, and fused
retrieval on this question category. This gives the architecture doc's
Experiment 3 rationale (justifying the reranking stage) a concrete number to
cite instead of a general precision-improvement argument: without
reranking, comparative-question retrieval would be badly degraded on this
corpus; with it, comparative questions are actually the *strongest*
category.

**How to apply:** cite this table directly when writing up why reranking
is in the pipeline, not just "it improves precision." If Experiment 3 ever
considers dropping or downgrading the reranker to save latency/VRAM, this is
the concrete cost -- a ~44pp Hit Rate@5 collapse on comparative questions
specifically (fused 55.6% -> would-be-necessary reranked 100%), not a
generic quality regression.

## Known evaluation-methodology limitation: gold-label ambiguity (q002)

While investigating `q002`'s reranked-stage miss (dense #2, sparse #2, fused
#1 all found the labeled paper `1611.06439`; reranker's own re-scoring of
the same 10 candidates dropped it out of its top-10 entirely), the cause
turned out not to be a reranker defect. A *different* paper in the corpus,
`2208.10943`, has a section literally titled "Challenges of Credit Card
Fraud Detection" with genuinely substantive, on-topic content:

> "Credit card fraud detection is extremely difficult due to its volume,
> the adaptive and unique nature of each fraud and the need for real time
> or near real time assessments... constraints such as lack of real data
> sets due to sensitivity, confidentiality and privacy concerns and the
> massively imbalanced highly skewed nature of the data makes the problem
> challenging..."

This is near-synonymous with the labeled source's own section title
("Difficulties of Credit Card Fraud Detection"). The reranker promoted this
second paper's chunk to rank #4 of its own ordering -- a defensible,
arguably correct call, not an error. **The corpus contains at least two
papers that legitimately answer this question, but the eval set's
single-`source_paper_id` scheme can only credit one of them.**

**How to apply:** this is a known constraint of the evaluation methodology,
not a system defect -- note it in the final write-up's limitations section
rather than reporting it as a retrieval or reranking failure. Don't
"fix" this by re-labeling `q002` to the second paper (that would just move
the same problem elsewhere); the honest framing is that single-gold-paper
labeling under-credits genuinely correct retrievals whenever topical overlap
exists across papers in the corpus, and a small number of scored misses
should be read with that caveat rather than taken as proof of a retrieval
weakness.
