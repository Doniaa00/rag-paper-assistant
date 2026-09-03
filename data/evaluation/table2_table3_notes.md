# Notes on Table 2 and Table 3 — Sample Sizes

## Table 2 (`table2_experiment_log.csv`) — Experiment 0 (Baseline)

- **n=24** (`n_quality`): Hit Rate@5, MRR, Faithfulness distribution, Relevancy
  distribution, and Citation Accuracy are all scored on the 24 non-negative
  questions only (15 factual + 9 comparative). These metrics require a known
  `source_paper_id`/`source_section` to check against, which negative
  questions don't have by design.
- **n=30** (`n_refusal`): Refusal Correctness (combined two-sided) is scored
  on all 30 questions — the 6 negative questions (refusal-on-negatives) plus
  the 24 non-negative questions (false-refusal check). It's the only metric
  in this table that spans the full eval set.
- **Two Refusal Correctness columns, not one** (`refusal_correctness_paper_level_pct`
  = 93.3% / 28/30, `refusal_correctness_section_level_pct` = 100% / 30/30,
  as of Step 12): these come from two different definitions of "did
  retrieval actually succeed" for the false-refusal half of this metric,
  and they are logically ordered, not independent estimates — because an
  exact-section hit always implies a paper-level hit (if the correct
  *section*'s chunk is in the top-5, some chunk of that *paper* is too,
  but not vice versa), the section-level criterion can only flag a false
  refusal in cases the paper-level criterion would also flag, never more.
  So `refusal_correctness_paper_level_pct` is always the lower (more
  conservative) figure and `refusal_correctness_section_level_pct` is
  always the higher (more generous) one — confirmed here (93.3% ≤ 100%),
  not a coincidence of this particular run.
  - **Paper-level (`hit_at_5`) is the stricter measure of the system's
    refusal behavior**, precisely because it's the *more lenient* judge of
    "did retrieval succeed": crediting success whenever any chunk from the
    correct paper appears in the top-5 is an easy bar to clear, so it can
    end up crediting retrieval as adequate in cases like `q015` where only
    an unrelated section of the right paper actually reached the model —
    and then counts the resulting refusal as false, even though the model
    arguably didn't have what it needed. This is the pessimistic,
    lower-bound reading: some of what it flags as "false" may not really
    be.
  - **Section-level (`exact_section_hit`) is the more lenient measure of
    the system's refusal behavior**, because it's the *stricter* judge of
    "did retrieval succeed": it only credits success when the exact
    labeled section is retrieved, so it can let a genuinely-avoidable
    false refusal off the hook whenever adequate evidence existed
    elsewhere without an exact section match. It also inherits whatever
    imprecision is in the eval set's own `source_section` ground truth —
    Step 12's human validation review found that ground truth itself has
    some ambiguity (the `q002` gold-label case, where a second paper
    legitimately answers the same question; the claim-level citation
    attribution gaps in `docs/known_limitations.md`). This is the
    optimistic, upper-bound reading.
  - Neither number alone is "the" refusal correctness rate — **the true
    rate is bounded between the two** (93.3% pessimistic lower bound,
    100% optimistic upper bound), not equal to either in isolation.

## Table 3 (`table3_quality_latency_vram.csv`) — Retrieval Stage Comparison

- **n=24** (`n_quality`): Hit Rate@5 per stage is scored on the 24
  non-negative questions only, same reasoning as Table 2.
- **n=30** (`n_latency`): Stage-only latency, cumulative latency, and peak
  VRAM are measured across all 30 questions, including negatives — a
  refusal response still runs the full retrieval pipeline and still costs
  real wall-clock time and GPU memory, so Step 9d deliberately included them.

In short: **quality metrics are n=24, cost metrics (latency/VRAM) are n=30**,
across both tables. Each CSV also carries this as explicit `n_quality`/
`n_latency`/`n_refusal` columns so the distinction travels with the data,
not just this note.
