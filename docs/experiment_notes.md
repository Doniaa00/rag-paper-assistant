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

## Systems metrics (Step 9d): retrieval latency is question-type-invariant

Full stage-level latency and resource-footprint instrumentation across all
30 eval-set questions (`evaluation/systems_metrics.py`), warm-up-excluded
per the Step 8 cold-start protocol.

**Headline finding: retrieval-stage latency barely moves across question
type, but total response time varies a lot -- and the variation is
generation, not retrieval.** Dense search, sparse search, and reranking
each stay within ~5% of their overall average regardless of whether the
question is factual, comparative, or negative (e.g. dense search: 1517.6ms
factual vs. 1558.3ms comparative vs. 1476.3ms negative -- an average, not a
coincidence, across 15/9/6 questions respectively). All of the
question-type variation in total response time traces to generation
duration, which scales directly with how much the model actually writes:
2.41s (factual, ~130 tokens) vs. 4.01s (comparative, ~220 tokens) vs. 0.245s
(negative, ~14-token refusal). This is a clean confirmation of what Steps
9b/9c already showed from a quality angle -- comparative questions are
genuinely different downstream of retrieval, producing longer, more
effortful answers -- now with a latency number attached: comparative
questions cost roughly 1.6x a negative refusal's total response time, and
essentially all of that gap is generation, not search.

**How to apply:** if end-to-end latency ever needs optimizing, retrieval is
not the lever -- it's already type-invariant and comparatively cheap next to
generation. Generation duration is the dial that actually moves, and it
moves with answer length/complexity rather than question type per se
(comparative questions just happen to elicit longer answers).

**VRAM stability: 3.438-3.439 GB across all 30 sequential queries, zero
drift.** This matches Step 8's isolated reranker+LLM measurement almost
exactly and confirms there's no memory leak or gradual accumulation across
a full run of back-to-back queries -- the combined footprint is flat
whether it's the 1st question or the 30th.

**`q022`'s per-stage latency anomaly -- flagged, not explained away.**
`q022` (negative) recorded the single slowest dense search (2,647ms, ~1.7x
the run average) and the single slowest reranking (1,907ms, ~2.5x average)
of all 30 questions -- yet its total response time (8,315ms) wasn't extreme
enough to register on the aggregate outlier check (that check only looks at
total time and peak VRAM), and its peak VRAM was the same 3.439GB as every
other question, ruling out a memory-pressure explanation. No other question
shows the same pattern, before or after. This looks like a one-off
transient system-level contention event (OS scheduling, background
process, etc.) rather than a systemic pipeline issue, but it's reported as
an observed, unexplained, non-recurring anomaly rather than dismissed --
there isn't enough evidence here to name a specific cause, and it's worth
watching for recurrence in any future systems-metrics run rather than
assuming it was a fluke.

## Experiment 2 resolution: fusion's raw-stage weakness is absorbed by reranking

**Original finding (Step 9):** the final retrieval baseline showed raw RRF
fusion underperforming dense-only at the pre-rerank stage -- 79.2% vs. 91.7%
Hit Rate@5 -- the clearest instance yet of the fusion-dilution pattern
documented earlier in this file (RRF rewarding cross-retriever consensus
over a single strong dense signal). That baseline number was raw motivation
to formally test alternatives, not a verdict on the pipeline as shipped,
since reranking always runs afterward in production.

**Sweep result (Step 10, `retrieval/experiment2_fusion_sweep.py`):** across
5 RRF k-values (10/20/40/60/100), 2 weighted-fusion ratios (dense 2x/3x),
a dense-only no-fusion arm, and a 3-depth candidate-pool sweep (top-10/20/30
per retriever), **the current default (RRF k=60, depth=20) ties or beats
every single alternative on post-rerank metrics -- the only metrics that
actually reach the user**, since the production pipeline always reranks
before generation. All five k-values produced *identical* post-rerank Hit
Rate@5/@10/MRR (95.8%/95.8%/0.783). Weighted fusion and both off-default
pool depths (10 and 30) each scored measurably *worse* post-rerank than the
current default. Full comparison table in
`data/evaluation/experiment2_fusion_results.csv`.

**Verification finding -- the precise mechanism, not just the correlation
(`retrieval/verify_rrf_k_tie.py`):** spot-checking 3 questions' actual top-10
fused chunk_id sets across k=10/60/100 showed k's effect on the fused pool
is real, not illusory -- 2 of 3 questions had genuinely different chunk_id
sets at different k values. But the difference was structurally confined to
**the tail of the ranking, around rank 9-10**, a direct consequence of RRF's
`1/(k+rank)` formula flattening at higher k (k=60 and k=100 were byte-for-
byte identical to each other in every case checked; only k=10, the steepest
curve, ever diverged, and only at the bottom of the pool). That tail region
never survives the reranker's top-5 cut, which is exactly why the sweep's
post-rerank output is unaffected regardless of which k produced the pool --
the ties are real, but not because RRF's ranking is k-invariant; they hold
because k's k-dependent disagreement and the reranker's decision boundary
don't overlap at this pool depth.

**Two results flagged as close-but-not-actionable -- worth revisiting only
if the corpus/eval set grows, not acted on now:**
- **Dense-only** ties the current default's Hit Rate@5 (95.8%) and comes
  within noise on MRR (0.779 vs. 0.783, smaller than one question's rank
  shift on n=24). Dropping the sparse/BM25 arm on this evidence would risk
  overfitting to a 24-question sample that simply doesn't happen to contain
  a question needing BM25's exact-term matching -- this file already has one
  concrete counterexample (`q015`'s dense-total-miss / BM25-rank-4 inversion,
  documented above) showing that strength is real on this corpus, just not
  exercised by every question.
- **Weighted fusion** (dense 2x/3x) improves the *raw* fused-stage MRR (up
  to 0.767 vs. 0.730 unweighted) but actively hurts post-rerank quality
  (91.7% vs. 95.8% Hit Rate@5). This reproduces this file's very first
  finding -- "reranking can mask a weak fusion stage" -- in reverse: here, a
  fusion change that looks like an improvement pre-rerank changes *which*
  candidates reach the reranker in a way that measurably hurts the final
  output. A better raw score is not evidence of a better pipeline.

**Conclusion: Experiment 2 is closed with no configuration change.** The
current defaults (RRF, k=60, top-20 per-retriever depth, fused top-10 into
the reranker) are now **validated against 11 concrete alternatives, not
merely unchallenged** -- a meaningfully stronger claim than "no one has
tried anything else yet."

**How to apply:** don't re-open the k/weighting/depth question without new
evidence -- specifically, a larger or more phrasing-diverse eval set (the
dense-only and weighted-fusion caveats above are exactly where a bigger
sample could flip the verdict). Any future retrieval-stage experiment should
default to scoring the post-rerank output, not the raw fusion output, per
this entry's mechanism finding -- a raw-stage-only comparison would have
recommended weighted fusion, which is now demonstrated to be the wrong call.

## Step 11: citation-reliability layer -- repair before reject

Step 9c's `check_citation_accuracy` could only diagnose hallucinated
citations after the fact, in evaluation. Step 11
(`generation/citation_guard.py`) turns that diagnosis into a runtime guard
that repairs or strips bad citations before an answer is ever returned.

**Design: repair-before-reject.** Every citation marker is resolved
against the evidence set in a fixed priority order: (1) if `paper_id`
already matches a real evidence chunk, leave it untouched; (2) if
`paper_id` doesn't match but `section_title` matches a real chunk's
section, **repair** it by swapping in that chunk's correct `paper_id` --
this is the load-bearing case, since it directly targets the
correct-section-wrong-paper_id failure mode documented as "the most
concerning variant" in Step 9c's hallucination taxonomy (a citation that
looks completely legitimate to a reader because the section title really
is correct, only the paper_id is wrong); (3) only when *neither* matches
anything in the evidence set is the citation **stripped** and replaced
with `[source could not be verified]`. Repair is preferred over stripping
whenever there's enough information to do it correctly -- stripping is the
fallback for genuinely unrecoverable citations, not the default response
to any mismatch.

**Results on the full 24-question set: 2 repairs, 2 strips, 3/24 questions
affected, 0 false positives or false negatives against Step 9c's already-
known cases.** The guard was re-run against the exact same evidence sets
the original Step 9c answers were generated against (retrieval
reconstruction confirmed all 24 questions' evidence_paper_ids matched the
stored CSV exactly, since retrieval is deterministic). It correctly:
- **repaired** both duplicate citations in `q018` (`[1901.03407, Importance
  of credit card fraud detection]` -> `[2208.10943, ...]`), the exact
  correct-section-wrong-paper_id case the design was built around;
- **stripped** `q017`'s wrong-paper-entirely citation
  (`[1901.03407, Autoencoders]`, pointing to `q009`'s paper on a
  cost-sensitive-learning question), correctly finding no repair target
  since neither field matched anything in `q017`'s evidence;
- and (see below) correctly identified `q014`'s malformed-reference-copying
  case as unrecoverable, though the first implementation's *handling* of
  that case needed a fix before it shipped.
All 21 other questions -- confirmed clean in Step 9c -- passed through with
zero actions taken, a clean 1:1 match to the known hallucination count.

**The q014 finding: technically correct, actually harmful -- caught before
shipping.** `q014`'s known malformed citation
(`[76, "Inspection-L compares favorably against ... classifying illicit
transactions.]`) is a single unclosed bracket that happens to swallow an
entire informative sentence before the model's next stray `]` closes it.
The first implementation correctly identified this as unresolvable (neither
`76` nor the giant quoted span matches anything in evidence) and stripped
it -- but stripping the *entire matched span* meant deleting the whole
Inspection-L comparison sentence along with the malformed citation marker,
replacing genuinely useful content with a bare "[source could not be
verified]" note. The diagnosis was right; the repair action was
disproportionate to the problem. **Fix: a length-based over-match guard**
(`max_citation_words`, default 20) -- a real citation marker is a handful
of words, so a matched span far longer than that is almost certainly the
bracket regex spanning an earlier unclosed `[` through an unrelated later
`]`, not an actual citation attempt, however comma-shaped its contents
look. Over-long matches now only have a short, clearly-identifiable
citation-shaped fragment removed from either edge (here, the leading `76`)
-- the informative prose is preserved verbatim, with the unverified-source
note appended after it rather than replacing it. Re-verified against all 24
questions after the fix: `q014` now preserves its full comparison sentence
with just the note attached, and `q017`/`q018` are byte-for-byte unchanged
from before the fix -- confirming the guard is correctly scoped to the
pathological long-match case, not a change to citation-stripping behavior
generally.

**How to apply:** this is the second time in this project a stripping/
flagging mechanism's raw logic was correct but its blast radius wasn't (see
the bracket-matching false-positive bug in Step 9c) -- worth treating
"does the check correctly identify the problem" and "is the check's
response to the problem proportionate" as two separate questions to verify,
not one, whenever a new automated repair/strip mechanism is built.

**This is now a permanent runtime safeguard, not a one-time analysis.**
`generate()` and `generate_with_metrics()` in `generation/local_ollama.py`
both run every real answer through `validate_and_repair_citations` before
returning it -- this runs on every query through `OllamaBackend` in normal
operation, not just when evaluation scripts are invoked. `generate_raw()`
is deliberately excluded, since the generation-quality judge
(`evaluation/generation_judge.py`) also calls it directly for
non-citation-bearing judge prompts, where guarding would be meaningless at
best.

## Step 12: Human Validation Findings

A blind human review of 9 questions sampled from the Step 9c full-scale
generation eval (`data/evaluation/human_validation_sample.md`, judge output
withheld until after independent review) surfaced three findings the
automated checks either can't see by design or got imprecisely right.

**1. Claim-to-evidence attribution gap (`q002`, `q011`).** No current
check verifies that a specific cited chunk actually supports the specific
claim it's attached to -- `check_citation_accuracy` only checks that the
cited `paper_id` exists somewhere in the evidence set, and
`judge_faithfulness` only checks that the answer's claims are supported
somewhere in the evidence block as a whole. `q002`'s bullet-by-bullet
`[Evidence N]` citations and `q011`'s citation to the wrong chunk of the
right paper (see finding detail in `docs/known_limitations.md`) both pass
every current check while leaving the actual claim-to-chunk correspondence
unverified. Documented as a known limitation, not fixed here -- see
`docs/known_limitations.md`, "Claim-level citation attribution gap."

**2. The judge's miss on internal contradiction (`q012`).** `q012`'s
answer states unsupervised learning "does not require balanced label
data, making it suitable for credit card fraud detection" and, two lines
later, that it "is not suitable for scenarios where labeled data is
unavailable" -- a direct self-contradiction about the exact same
condition. `judge_faithfulness` rated the answer `fully_faithful`, its
reasoning explicitly noting the answer retains information "without
contradicting" the evidence -- accurate claim-by-claim, but blind to the
contradiction between the answer's own two claims. Faithfulness-to-
evidence and internal coherence are different quality dimensions; the
current judge only evaluates the former. Documented as a known limitation,
not fixed here -- see `docs/known_limitations.md`, "Judge evaluates
holistically, not clause-by-clause."

**3. Corrected understanding of `q015`: a genuine retrieval granularity
failure, not a clean false refusal.** Comparing `q015`'s reconstructed
evidence (confirmed identical to what Step 9c's judge actually saw --
same `evidence_paper_ids` sequence, same chunk content, cross-validated
against Step 10's independent reconstruction of the same question) against
the corpus shows the real comparative section (`2503.13195`, "A.
Contrasting Traditional Models with Deep Learning Models") was never
retrieved at all -- not even into the fused top-10 before reranking. Only
an unrelated section of the same paper ("D. Summary and Insights") made it
into evidence. The judge's faithfulness reasoning claimed "the evidence
does contain information about both traditional and deep learning anomaly
detection methods and their strengths and weaknesses" -- **overstated, not
fabricated**: the evidence does contain three brief, incidental comparative
clauses scattered across unrelated introductions (all noting a *weakness*
of traditional methods relative to deep learning; none noting a *strength*
of traditional methods), which the judge is not wrong to have noticed, but
which don't add up to the structured "strengths and weaknesses" treatment
the judge's wording implies. The model's refusal was closer to defensible
than either the original false-refusal flag or the judge's own reasoning
suggested.

**Consequence: `check_false_refusal`'s retrieval-succeeded criterion was
too coarse, and has been corrected.** The original criterion (paper-level
`hit_at_5`) counted `q015` as "retrieval succeeded" because *a* chunk from
the correct paper appeared in the top-5 -- exactly the granularity this
finding shows is misleading. Updated `check_false_refusal` (see its
docstring in `evaluation/generation_judge.py`) to require
`exact_section_hit` instead, and recomputed
`false_refusal` for all 24 non-negative questions
(`evaluation/recompute_false_refusal.py`). Only questions where the model
actually refused can change classification under a stricter criterion --
across the 24, that's exactly `q015` and `q016`, and both flip from
`True` to `False`: neither had its exact source section retrieved
(`exact_section_hit=False` for both at the reranked stage, despite
`hit_at_5=True`), so neither can be confidently called a *false* refusal
under the corrected definition. The false-refusal count for the 24
non-negative questions drops from 2 to 0, and the combined two-sided
refusal correctness figure (Table 2's Experiment 0 row) recomputes from
28/30 (93.3%) to 30/30 (100%) under the corrected criterion.

**How to apply:** this is not evidence the system's refusal behavior
quietly got better -- nothing about the pipeline or the model changed. It's
evidence the original 93.3% figure was measuring false refusals at the
wrong granularity, crediting retrieval success whenever the right paper
showed up anywhere in top-5 regardless of section. Table 2 currently still
shows 93.3% and needs updating to reflect this correction in a future
pass. More generally: whenever a paper-level retrieval metric is used to
gate a downstream correctness judgment (as `check_false_refusal` does),
check whether paper-level is actually the right granularity for that
judgment -- here it wasn't, and the human validation sample is what
surfaced it, not the automated metrics themselves.
