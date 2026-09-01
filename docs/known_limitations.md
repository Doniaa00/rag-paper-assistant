# Known Limitations

Non-blocking issues discovered during ingestion, kept here so they aren't
re-discovered (or re-investigated) later.

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
