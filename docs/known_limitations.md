# Known Limitations

Non-blocking issues discovered during ingestion, evaluation, and
generation, kept here so they aren't re-discovered (or re-investigated)
later.

## 2509.19032 — incomplete reference extraction

2509.19032's reference list failed GROBID segmentation due to Word-export
text-layer artifacts (irregular kerning/spacing). Body content, sections, and
abstract parsed correctly and are unaffected — only the bibliography is
incomplete. Not fixed, since reference data isn't used downstream in
retrieval or evaluation. Full bibliography is recoverable via direct PDF text
extraction if ever needed.

See the Step 2b.1 investigation: `consolidateCitations=1` and the isolated
`/api/processReferences` endpoint were both tried and neither fixed the
segmentation, confirming it's a GROBID limitation on this specific PDF's
text layer rather than a request-parameter issue.

## Garbled section headers from font-encoding/ligature artifacts

A handful of `<div>` headers extracted by GROBID contain corrupted text --
e.g. "G D z" (in 2012.02364) and "Tokeniza�on (TOK)" (in 2207.03820,
where a "ti" ligature glyph didn't map to a valid character). This traces
back to the source PDF's font encoding, not a GROBID or chunking bug --
GROBID (and any other text-extraction tool) can only read the character
mapping the PDF itself provides. Flagged, not fixed: these are isolated
section-title cosmetic issues (the section body text and chunking around
them are unaffected), not worth engineering around at this stage.

## Claim-level citation attribution gap (generation evaluation)

Discovered during Step 12's human-vs-judge validation review. Neither
automated check verifies that a specific cited evidence chunk actually
supports the specific claim it's attached to: `check_citation_accuracy`
only confirms the cited `paper_id` exists *somewhere* in the evidence set
(an existence check), and `judge_faithfulness` only confirms the answer's
claims are supported *somewhere* in the evidence block as a whole (a
holistic check). Neither cross-references claim N against the actual
content of the specific chunk claim N is cited to.

Concretely: `q002`'s answer attributes distinct bullet-point claims to
specific `[Evidence N]` markers, and `q011`'s citation points to
`2106.07178`'s "ANOMALOUS NODE DETECTION (ANOS ND)" chunk for a graph
definition that the eval set's own `source_section` label says actually
comes from that paper's "PRELIMINARIES" section instead. Both pass every
current check -- right paper, plausible claim -- but neither check actually
confirms the claim's substance lives in the chunk it's attributed to.

Not fixed: closing this gap would require a new claim-level check (segment
the answer into individual claims, resolve each claim's own citation, and
verify chunk-level support per claim) -- meaningfully more work than the
current chunk-existence and whole-answer approaches, and out of scope for
now. Documented as a real gap in what "citation accuracy" and
"faithfulness" currently mean in this system, same treatment as the
gold-label ambiguity limitation (see `docs/experiment_notes.md`,
"Known evaluation-methodology limitation: gold-label ambiguity (q002)") --
not something to silently patch or claim is covered.

## Judge evaluates holistically, not clause-by-clause (misses internal contradiction)

Discovered during Step 12's human-vs-judge validation review.
`judge_faithfulness` rates an entire answer against the entire evidence
block in one pass; it has no mechanism to check the answer's own internal
logical consistency between its own claims.

Concretely: `q012`'s answer states, under "Unsupervised Learning --
Advantages," that unsupervised learning "does not require balanced label
data, making it suitable for credit card fraud detection," then two lines
later under "Disadvantages" states unsupervised learning "is not suitable
for scenarios where labeled data is unavailable" -- a direct
self-contradiction about whether unsupervised learning is suitable
precisely when labels are unavailable. The judge rated this answer
`fully_faithful`, with reasoning explicitly stating the answer retains
information "without contradicting" the evidence -- true of each claim's
relationship to the evidence individually, but the judge's own reasoning
never surfaced the contradiction *between* the answer's two claims.

Not fixed: faithfulness-to-evidence and internal coherence are different
quality dimensions, and no current check evaluates the latter. Documented
as a known blind spot in the current generation-quality evaluation
methodology, not something the existing faithfulness/relevancy checks
should be assumed to catch.
