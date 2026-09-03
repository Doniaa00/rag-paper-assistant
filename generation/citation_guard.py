"""
Step 11: deterministic citation-reliability layer.

Step 9c's check_citation_accuracy diagnosed hallucinated citations (3/24
confirmed cases in the full-scale generation eval, including the
correct-section-wrong-paper_id variant -- see docs/experiment_notes.md).
This module turns that diagnosis into a runtime guard that repairs or
strips bad citations before an answer is returned to a user, rather than
only flagging them after the fact in evaluation.

This is the canonical home for the citation-marker parsing primitives
(bracket detection, citation-attempt filtering, paper_id extraction)
originally built for evaluation/generation_judge.py's check_citation_format
and check_citation_accuracy in Step 9c. They were moved here rather than
reused via import from generation_judge, because generation_judge.py
imports generation.local_ollama.OllamaBackend (for the judge backend), and
this guard is wired into local_ollama.py itself (see generate() /
generate_with_metrics()) -- importing generation_judge from local_ollama
would create a circular import. generation_judge.py now imports these
primitives from here instead, so the parsing logic still has exactly one
implementation, just relocated to the module both callers can safely
depend on.

For every citation marker in a generated answer:
  - paper_id resolves to a real evidence chunk -> leave untouched.
  - paper_id doesn't resolve, but section_title matches a real evidence
    chunk's section_title -> REPAIR: swap in that chunk's correct paper_id.
  - neither resolves -> STRIP the citation marker and replace it with a
    neutral "[source could not be verified]" note.
"""

import re

STRIPPED_NOTE = "[source could not be verified]"

# Our generation prompt (generation/local_ollama.py) instructs a specific
# bare inline citation format: [paper_id, section_title] -- two
# comma-separated values, no field-name labels, no quotes, no reference to
# the "[Evidence N]" markers used only in the evidence block shown to the
# model.
_BRACKET_CONTENT_RE = re.compile(r"\[([^\[\]]*)\]")
_EVIDENCE_LABEL_RE = re.compile(r"\bEvidence\s+\d+\b", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")
_MANUAL_PAPER_ID_RE = re.compile(r"chandola2007_anomaly_survey", re.IGNORECASE)


def _is_citation_attempt(text: str) -> bool:
    """Heuristic: does this bracketed span look like an attempt at the
    instructed [paper_id, section_title] citation format, as opposed to
    incidental bracket content copied verbatim from the evidence text?
    Evidence chunks are excerpts from real papers and routinely contain
    their own in-text reference numbers (e.g. "[11]", "[76]") or bracketed
    math notation (e.g. "[x_i]"), which the model can end up quoting
    directly -- neither is a citation attempt, and would otherwise be
    misflagged or mis-repaired as one."""
    if "," in text:
        return True
    if _EVIDENCE_LABEL_RE.search(text):
        return True
    if "paper_id" in text.lower():
        return True
    if _ARXIV_ID_RE.search(text) or _MANUAL_PAPER_ID_RE.search(text):
        return True
    return False


def _extract_citation_paper_id(bracket_text: str, evidence_chunks: list):
    """Best-effort extraction of the paper_id a citation marker is pointing
    at, tolerant of the malformed variants seen in practice. Handles:
      - bare format: "1901.03407, Autoencoders" -> "1901.03407"
      - leaked field=value syntax: "paper_id=1901.03407, ..." -> "1901.03407"
      - "[Evidence N]" index references -> resolved via evidence_chunks[N-1],
        since that's the paper the model actually meant, even though citing
        by internal evidence-slot number is itself a format violation.
    """
    field_match = re.search(r"paper_id\s*=\s*([^\s,\]]+)", bracket_text)
    if field_match:
        return field_match.group(1).strip("\"'")

    first_field = bracket_text.split(",")[0].strip()

    evidence_index_match = re.fullmatch(r"Evidence\s+(\d+)", first_field, re.IGNORECASE)
    if evidence_index_match:
        idx = int(evidence_index_match.group(1)) - 1
        if 0 <= idx < len(evidence_chunks):
            return evidence_chunks[idx]["paper_id"]
        return None  # references an evidence slot that doesn't exist at all

    return first_field or None


def _extract_citation_section_title(bracket_text: str):
    """Best-effort extraction of the section_title half of a citation
    marker, tolerant of the same malformed variants
    _extract_citation_paper_id handles -- a citation malformed enough to
    fail check_citation_format can still carry a recognizable
    section_title, which is exactly what the repair path below needs."""
    field_match = re.search(r"section_title\s*=\s*(.+)", bracket_text, re.IGNORECASE)
    if field_match:
        section = field_match.group(1).strip().rstrip(",").strip().strip("\"'")
        return section or None

    parts = bracket_text.split(",", 1)
    if len(parts) < 2:
        return None
    section = parts[1].strip().strip("\"'")
    return section or None


DEFAULT_MAX_CITATION_WORDS = 20


def _looks_like_short_citation_fragment(text: str, max_words: int = 4) -> bool:
    """Is `text` short and shaped like an in-text reference number, an
    Evidence-slot marker, or an arxiv-style/manual paper ID -- the kind of
    small token that can end up glued to the edge of a runaway,
    over-matched bracket span (e.g. a copied reference number "76" right
    before the model launches into unrelated prose)? Used only to decide
    whether a small strippable fragment is "clearly identifiable" at the
    edge of an over-long match -- not a general citation validity check."""
    text = text.strip()
    if not text or len(text.split()) > max_words:
        return False
    return bool(
        re.fullmatch(r"\d+", text)
        or _EVIDENCE_LABEL_RE.fullmatch(text)
        or _ARXIV_ID_RE.fullmatch(text)
        or _MANUAL_PAPER_ID_RE.fullmatch(text)
    )


def validate_and_repair_citations(
    generated_answer: str, evidence_chunks: list, max_citation_words: int = DEFAULT_MAX_CITATION_WORDS,
) -> dict:
    """Parse every citation marker in generated_answer and repair or strip
    the ones that don't resolve to a real evidence chunk.

    Returns {repaired_answer, actions_taken}. actions_taken is a list of
    {action: "repaired"|"stripped", original, repaired_to (repaired only),
    reason} dicts, in the order encountered, empty if the answer was
    already clean.

    Only brackets that look like a citation attempt (_is_citation_attempt)
    are touched -- incidental brackets copied from evidence text (in-text
    reference numbers, math notation) are left exactly as written, same as
    check_citation_format/check_citation_accuracy.

    max_citation_words guards against regex over-match: a real citation
    marker is a handful of words ("paper_id, section_title"), so a matched
    span far longer than that is almost certainly the bracket regex
    accidentally spanning an earlier unclosed "[" through a later,
    unrelated "]" and swallowing real prose in between -- not an actual
    citation attempt, however comma-shaped its contents look. In that case
    the marker is never resolved against the evidence set at all: only a
    short, clearly-identifiable citation-shaped fragment at either edge
    (see _looks_like_short_citation_fragment) is removed, the rest of the
    prose is preserved verbatim, and the unverified-source note is appended
    after it -- so an over-match degrades to an unverified sentence, never
    to a deleted one.
    """
    valid_paper_ids = {chunk["paper_id"] for chunk in evidence_chunks}
    section_to_paper_id = {chunk["section_title"]: chunk["paper_id"] for chunk in evidence_chunks}

    actions_taken = []

    def _replace(match: re.Match) -> str:
        bracket_text = match.group(1)
        if not _is_citation_attempt(bracket_text):
            return match.group(0)

        word_count = len(bracket_text.split())
        if word_count > max_citation_words:
            return _handle_overmatch(match, bracket_text, word_count, max_citation_words, actions_taken)

        paper_id = _extract_citation_paper_id(bracket_text, evidence_chunks)
        if paper_id and paper_id in valid_paper_ids:
            return match.group(0)

        section_title = _extract_citation_section_title(bracket_text)
        if section_title and section_title in section_to_paper_id:
            correct_paper_id = section_to_paper_id[section_title]
            repaired_to = f"[{correct_paper_id}, {section_title}]"
            actions_taken.append({
                "action": "repaired",
                "original": match.group(0),
                "repaired_to": repaired_to,
                "reason": f"paper_id {paper_id!r} not in evidence set, but section_title matched paper {correct_paper_id!r}",
            })
            return repaired_to

        actions_taken.append({
            "action": "stripped",
            "original": match.group(0),
            "reason": f"neither paper_id {paper_id!r} nor section_title {section_title!r} matched any evidence chunk",
        })
        return STRIPPED_NOTE

    repaired_answer = _BRACKET_CONTENT_RE.sub(_replace, generated_answer)

    return {"repaired_answer": repaired_answer, "actions_taken": actions_taken}


def _handle_overmatch(match: re.Match, bracket_text: str, word_count: int, max_citation_words: int, actions_taken: list) -> str:
    """A matched span this long is a regex over-match (real prose
    accidentally captured, e.g. by an earlier unclosed "["), not an actual
    citation attempt -- see validate_and_repair_citations' docstring.
    Preserve the prose; remove only a short citation-shaped fragment at
    either edge if one is clearly identifiable, and always append the note
    rather than deleting the informative content."""
    segments = [s.strip() for s in bracket_text.split(",")]
    leading, trailing = segments[0], segments[-1]

    if len(segments) > 1 and _looks_like_short_citation_fragment(leading):
        prose = bracket_text.split(",", 1)[1].strip()
        removed = leading
    elif len(segments) > 1 and _looks_like_short_citation_fragment(trailing):
        prose = bracket_text.rsplit(",", 1)[0].strip()
        removed = trailing
    else:
        prose = bracket_text.strip()
        removed = None

    reason = f"matched span is {word_count} words (> {max_citation_words}), treated as a regex over-match, not a citation attempt"
    reason += f"; removed short citation-shaped fragment {removed!r}, rest of the prose preserved" if removed else "; no short citation-shaped fragment found at either edge, prose left fully intact"

    actions_taken.append({
        "action": "stripped",
        "original": match.group(0),
        "reason": reason,
    })
    return f"{prose} {STRIPPED_NOTE}"
