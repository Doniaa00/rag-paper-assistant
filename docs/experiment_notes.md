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

## Generation-quality judge pilot (Step 9c): three findings before trusting it at scale

A 6-question pilot (2 factual, 2 comparative, 2 negative) was run before any
full-scale generation-quality evaluation, specifically because judge output
can't be trusted blindly -- and the pilot justified that caution.

**1. Faithfulness judge false positives, fixed by teaching paraphrase
tolerance, not leniency.** Round 1's judge prompt rated all 4 non-negative
pilot answers "unfaithful," but manual review showed at least 3 of those 4
were false positives -- the judge's own reasoning text quoted evidence that
directly supported the claim it had just rejected, apparently penalizing
any rewording rather than checking whether the *meaning* was supported.
Round 2 revised the prompt to explicitly instruct that paraphrasing is not
a faithfulness violation, with contrastive examples anchoring
`fully_faithful` (accurate paraphrase) vs. `unfaithful` (fabrication). All 3
false positives flipped to `fully_faithful`. Critically, this was not just
loosened grading: the 4th case (`q027`) was correctly *downgraded* from
`unfaithful` to `faithful_but_imprecise`, with reasoning that specifically
identified an emphasis-shift/precision-loss the answer actually made --
matching independent manual review exactly. **The revision taught the judge
to discriminate between fabrication and imprecision, not to rubber-stamp
everything** -- a real distinction to check for whenever this rubric is
revised again, since a judge that just gets more lenient overall would be as
useless as one that's falsely strict.

**2. Citation formatting is a capability ceiling, not a prompting problem --
evidence for Experiment 5, not more prompt iteration.** Three rounds of
prompt work on `generation/local_ollama.py`'s citation instruction --
including explicit right/wrong examples added directly in the prompt after
round 2's findings -- only reliably fixed 1 of the 4 pilot cases' citation
formatting, and that "fix" surfaced a worse problem (see finding 3). The
other citations kept reproducing the exact malformed patterns the prompt
now explicitly labels as WRONG (leaked `paper_id=`/`paper_title=` fields,
bare `[Evidence N]` references), with no improvement between rounds 2 and 3
despite the added examples. **This looks like `qwen2.5:3b-instruct` hitting
a genuine instruction-following ceiling on this level of formatting
precision, not a prompt-wording gap** -- diminishing returns from a third
iteration is the signal, not a reason to try a fourth. This is concrete
evidence for the architecture doc's Experiment 5 (local vs. hosted model
comparison): if citation-format compliance matters for the final system, a
larger/hosted model is a more promising lever than further local-prompt
engineering. In the meantime, a programmatic citation validator/repair step
is more likely to help than continued prompting.

**3. `check_citation_accuracy`'s catch could not have been caught by either
other check -- the three checks are complementary, not redundant.** While
verifying the citation-format fix, `q017`'s regenerated answer cited
`[1901.03407, Autoencoders]` -- a real paper, correctly formatted, that was
never in `q017`'s own evidence set (it's `q009`'s source paper, about
autoencoders, for a question about cost-sensitive learning). This is a
genuine fabricated source attribution:
- `check_citation_format` passed it -- the marker is syntactically clean,
  exactly two comma-separated bare fields, no leaked labels.
- `judge_faithfulness` rated the answer `fully_faithful` -- it checks
  whether the answer's *prose claims* are grounded in evidence, and the
  prose claims here were fine; it has no mechanism to check whether a
  citation marker's paper_id is real.
- Only `check_citation_accuracy` -- built specifically to cross-reference
  each citation's claimed paper_id against the evidence_chunks actually
  passed to the model for that query -- caught it, correctly, while also
  correctly clearing two other pilot answers whose citations were
  badly-formatted (`[Evidence N]` style) but pointed to genuinely correct
  papers once resolved.

**How to apply:** run all three checks (faithfulness, format, accuracy) on
every generation-eval question going forward, not a subset -- each one
catches a failure mode invisible to the other two. A format-valid citation
is not evidence of a real citation; a content-faithful answer is not
evidence of correct attribution. Don't assume a full-scale run only needs
the faithfulness judge because it's the most expensive-sounding check.

## Full-scale generation evaluation (n=24 non-negative + 6 negative)

The full pipeline (retrieve -> rerank -> generate) plus all four
generation-quality checks were run across the complete 30-question eval
set. Aggregate numbers below are the corrected ones -- see the tooling-bug
note at the end before trusting any citation-related figure computed
before this run.

**Aggregate numbers:**
- Faithfulness: 79.2% fully_faithful, 12.5% faithful_but_imprecise, 8.3%
  unfaithful (2/24).
- Relevancy: 50.0% yes, 41.7% partial, 8.3% no (2/24).
- Citation format: 12.5% pass (3/24) -- the failures are overwhelmingly
  cosmetic (back-to-back duplicate citations, leaked `[Evidence N]` labels),
  consistent with the Step 9c pilot's capability-ceiling finding rather than
  a new problem.
- Citation accuracy: 87.5% pass (21/24) -- 3 confirmed hallucinated
  citations.
- Refusal accuracy on true negatives: 100% (6/6) -- held steady from the
  pilot, no regression.

**Citation hallucination taxonomy -- 3 confirmed variants, not one:**
1. **Wrong-paper-entirely** (`q017`, reproduced from the pilot): cites a
   real paper that is simply the wrong one -- q009's autoencoders source,
   for a question about cost-sensitive learning. No overlap with the
   correct source at all.
2. **Correct-section-wrong-paper_id** (`q018`, new): the cited
   `section_title` is verbatim correct (it's really the true source's
   section), but the `paper_id` attached to it belongs to a different
   paper entirely. **This is the most concerning variant** -- a reader
   checking the citation would see a real, correctly-worded section title
   and have no surface-level reason to doubt the (wrong) paper_id next to
   it. Wrong-paper-entirely and malformed-reference-copying are easier to
   notice as broken; this one is the closest to being genuinely misleading.
3. **Malformed-reference-copying** (`q014`, new): the model copies an
   in-text reference number from the evidence's own citation style (e.g.
   "[76]") and garbles it together with a large quoted span, producing a
   citation that is both unparseable and inaccurate at once.

**How to apply:** when reporting citation hallucination rates, report the
taxonomy, not just a pass/fail rate -- variant 2 deserves separate,
higher-priority attention than variants 1 and 3 precisely because it is
hardest for a reader to catch unaided.

**False refusals -- a new, symmetric failure mode to hallucination, and a
real gap in current evaluation coverage.** 2/24 (8.3%) non-negative
questions (`q015`, `q016`) were incorrectly declined ("The provided
evidence does not contain sufficient information...") despite adequate
evidence being present in retrieval:
- `q015`: the real source (`2503.13195`) was in the retrieved evidence: the
  model refused anyway. The faithfulness judge correctly caught this --
  "the evidence does contain information about both traditional and deep
  learning anomaly detection methods and their strengths and weaknesses."
- `q016`: the source paper (`2009.13807`) appeared **four times** in the
  retrieved evidence -- about as strong a retrieval signal as this eval set
  produces -- and the model still refused. The likely trigger: the question
  asks what "criticism" the paper raises, but the evidence's own framing
  never uses that word, instead discussing dataset "flaws" directly. This
  looks like the same phrasing-proximity sensitivity documented earlier in
  these notes (the n=12->n=17 retrieval findings), now showing up at the
  generation stage instead of retrieval.

This is a real, user-facing risk distinct from -- and arguably more costly
than -- an imprecise answer: a `faithful_but_imprecise` answer still gives
the user something to work with and evaluate, while a false refusal
silently withholds a correct answer the system actually had. A user has no
way to distinguish "the corpus genuinely doesn't cover this" from "the
system had the answer and declined anyway."

**Methodology gap worth naming explicitly:** `check_refusal` as currently
built only measures whether the system correctly refuses on true negative
questions (where refusal is the correct behavior) -- it has no counterpart
that measures false refusals on true positive questions (where refusal is
a failure). The 100% refusal accuracy figure above is therefore only half
the picture: it says nothing about q015/q016-style over-refusal, which was
only caught here because judge_faithfulness happened to flag the refusal
claim as unsupported. A dedicated check (e.g., flag any non-negative
question whose answer matches the refusal pattern, cross-referenced against
whether real evidence was actually retrieved) would make this failure mode
visible without depending on the faithfulness judge catching it
incidentally.

**Methodology note: a tooling bug was self-caught and fixed mid-analysis.**
The first pass of `check_citation_format`/`check_citation_accuracy` treated
*any* `[...]` bracket as a citation attempt. Evidence chunks are excerpts
from real papers and routinely contain their own in-text reference numbers
(e.g. "[11]", "[76]") and bracketed math notation (e.g. "[x_i]"), which the
model sometimes quotes verbatim while paraphrasing -- both were being
misflagged as malformed or hallucinated citations. This inflated the raw
citation-format failure rate and the hallucination count (6 flagged
citations, only 3 genuine) before the fix. Both functions now require a
bracket to contain a comma, an `Evidence N`/`paper_id` marker, or an
arxiv-ID-shaped token before treating it as a citation attempt at all,
verified against the known false-positive cases before recomputing. **The
numbers in this entry are the corrected, post-fix numbers** -- any
citation-format or citation-accuracy figure computed before this fix should
be treated as unreliable.
