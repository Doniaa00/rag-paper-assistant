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
