"""
Step 9c: Generation-quality judge.

Uses our own local model (Qwen2.5-3B via the existing GenerationBackend
interface) as an LLM-judge, per the architecture doc's "free and auditable"
principle -- no external API calls, no cost, and the judge prompts/outputs
are fully inspectable.

judge_faithfulness: does every claim in the generated answer actually trace
back to the evidence it was given (not the judge's own training knowledge)?

judge_relevancy: does the answer address what was asked, independent of
whether its claims are accurate?

check_refusal: for negative questions, did the system correctly decline
rather than fabricate an answer? Implemented as a rule-based keyword check
rather than a third judge-model call -- see its docstring for why.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from generation.local_ollama import OllamaBackend  # noqa: E402
from generation.citation_guard import (  # noqa: E402
    _BRACKET_CONTENT_RE, _EVIDENCE_LABEL_RE, _is_citation_attempt, _extract_citation_paper_id,
)

FAITHFULNESS_RATINGS = {"fully_faithful", "faithful_but_imprecise", "unfaithful"}
RELEVANCY_RATINGS = {"yes", "partial", "no"}

FAITHFULNESS_PROMPT = """You are a fact-checker reviewing an AI-generated answer against the evidence it was given.

Check whether the MEANING of each claim in the answer is supported by the evidence below -- NOT whether the wording matches exactly. Paraphrasing, summarizing, and synthesizing the evidence in different words is normal and expected, and must NOT be penalized on its own. Only flag a claim if its meaning -- not its phrasing -- is unsupported.

QUESTION: {question}

EVIDENCE PROVIDED TO THE AI:
{evidence_block}

AI-GENERATED ANSWER:
{answer}

Rate the answer's faithfulness to the evidence using exactly one of these three labels:

- fully_faithful: every claim is supported by the evidence, even when reworded, paraphrased, or summarized differently from the source text. Exact wording match is NOT required.
  Example: evidence says "The goal of this type of learning is to minimize the total cost of misclassification." Answer says "This approach aims to minimize the total cost of misclassification." -> fully_faithful (same meaning, different words -- this is NOT a faithfulness problem).

- faithful_but_imprecise: claims are grounded in the evidence but shift emphasis, oversimplify a specific detail, or lose some precision, without being wrong or fabricated.
  Example: evidence says a method "lacks the capability to uncover the causal nature of each case AND is easily influenced by surrounding neighbors due to the aggregation mechanism" (two distinct points). Answer says the method is "vulnerable to the aggregation mechanism," mentioning only the second point and dropping the first. -> faithful_but_imprecise (grounded, but shifts emphasis and loses part of the original claim).

- unfaithful: contains a claim that is absent from the evidence, contradicts it, or cannot reasonably be inferred from it -- i.e. it is fabricated. Do NOT use this label just because the answer's wording differs from the evidence's wording.
  Example: the evidence never mentions dataset size. Answer states "the model was trained on 10 million examples." -> unfaithful (fabricated -- not present in or inferable from the evidence).

Respond in EXACTLY this format (two lines, nothing else):
RATING: <fully_faithful|faithful_but_imprecise|unfaithful>
REASONING: <1-3 sentences identifying the specific claim(s) that drove your rating and why>"""

RELEVANCY_PROMPT = """You are evaluating whether an AI's answer actually addresses the question asked, regardless of whether the answer's facts are correct.

QUESTION: {question}

AI-GENERATED ANSWER:
{answer}

Rate whether the answer addresses the question using exactly one of these three labels:
- yes: the answer directly and substantially addresses what was asked.
- partial: the answer addresses part of the question but misses or sidesteps some of it.
- no: the answer does not address the question asked at all.

Respond in EXACTLY this format (two lines, nothing else):
RATING: <yes|partial|no>
REASONING: <1-2 sentences explaining your rating>"""

# Phrases our own generation prompt (generation/local_ollama.py) instructs
# the model to use when evidence is insufficient, plus a few common
# paraphrases a 3B model might drift into.
REFUSAL_PATTERNS = [
    r"insufficient (evidence|information)",
    r"does not contain sufficient",
    r"do(es)? not (provide|contain) (enough|sufficient)",
    r"cannot answer",
    r"unable to answer",
    r"no relevant evidence",
    r"not (enough|sufficient) information",
    r"cannot be answered",
    r"the (provided )?evidence does not",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _format_evidence_block(evidence_chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(evidence_chunks, start=1):
        parts.append(
            f"[Evidence {i}] paper_id={chunk['paper_id']}, "
            f"section_title=\"{chunk['section_title']}\"\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def _parse_rating_response(raw: str, allowed_ratings: set):
    rating_match = re.search(r"RATING:\s*(\w+)", raw, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    rating = rating_match.group(1).strip().lower() if rating_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else raw.strip()

    if rating not in allowed_ratings:
        # Judge didn't follow the format -- surface the raw output rather
        # than silently guessing, so a malformed judge response is visible.
        return {"rating": "PARSE_ERROR", "reasoning": f"Could not parse a valid rating. Raw judge output: {raw!r}"}

    return {"rating": rating, "reasoning": reasoning}


def judge_faithfulness(question: str, generated_answer: str, evidence_chunks: list, judge_backend=None) -> dict:
    judge_backend = judge_backend or OllamaBackend()
    evidence_block = _format_evidence_block(evidence_chunks)
    prompt = FAITHFULNESS_PROMPT.format(question=question, evidence_block=evidence_block, answer=generated_answer)
    raw = judge_backend.generate_raw(prompt)
    return _parse_rating_response(raw, FAITHFULNESS_RATINGS)


def judge_relevancy(question: str, generated_answer: str, judge_backend=None) -> dict:
    judge_backend = judge_backend or OllamaBackend()
    prompt = RELEVANCY_PROMPT.format(question=question, answer=generated_answer)
    raw = judge_backend.generate_raw(prompt)
    return _parse_rating_response(raw, RELEVANCY_RATINGS)


def check_refusal(generated_answer: str) -> bool:
    """Rule-based check: did the system decline rather than fabricate an
    answer for a negative (unanswerable) question?

    Chose rule-based over a third judge-model call because:
    1. Our own generation prompt (generation/local_ollama.py) explicitly
       instructs the model to use a specific, near-fixed refusal phrase
       ("The provided evidence does not contain sufficient information to
       answer this question."), so pattern matching against that
       instructed phrase (plus a few likely paraphrases) is directly
       testing prompt-compliance, which is exactly what this check is for.
    2. It's deterministic and fully transparent -- no risk of the judge
       model itself misjudging a refusal, and free to audit by eye.
    3. Zero extra latency/cost versus adding a third LLM call per question.
    Trade-off: could miss a creatively-phrased refusal outside the pattern
    list, or a hedge that isn't quite a full refusal. Worth revisiting with
    a judge-model version if the pilot run surfaces false negatives.
    """
    return bool(_REFUSAL_RE.search(generated_answer))


def check_false_refusal(generated_answer: str, question_type: str, retrieval_succeeded: bool) -> bool:
    """Mirror of check_refusal: did the system decline to answer a question
    it actually had adequate evidence for?

    check_refusal measures whether the system correctly declines on true
    negatives (unanswerable questions) -- it has no counterpart for the
    opposite failure, a false refusal on a true positive (a question with
    real evidence in hand). That gap is real: the n=24 full-scale run found
    2 such cases (q015, q016) only because judge_faithfulness happened to
    flag the refusal claim as unsupported by the evidence -- an incidental
    catch, not a repeatable measurement. This function makes it repeatable.

    Reuses check_refusal's phrase-matching for the "did it refuse" half.
    The "should it have refused" half is retrieval_succeeded, which the
    caller derives from data/evaluation/retrieval_eval_results.csv
    (stage="reranked") -- not re-derived here, since retrieval quality is
    Step 9b's concern, not this module's.

    Step 12 update: retrieval_succeeded must be derived from
    exact_section_hit, NOT paper-level hit_at_5. Paper-level Hit Rate@5
    credits retrieval as "succeeded" whenever any chunk from the correct
    paper appears in the top-5, regardless of which section -- and Step
    12's human-validation investigation showed that granularity is
    misleading: q015's reranked evidence contained the correct paper
    (2503.13195) at rank 4, but not the actual section that answers the
    question ("A. Contrasting Traditional Models with Deep Learning
    Models"), only an unrelated section of the same paper. Under
    hit_at_5, that counted as "retrieval succeeded, model refused
    anyway" -- a false refusal. It shouldn't: the model was never
    actually given the section it needed, so declining wasn't clearly
    wrong. exact_section_hit is the correct granularity for "did this
    question actually have the evidence it needed" -- hit_at_5 answers a
    coarser, paper-level question that this check needs a finer answer
    to.

    Only meaningful for non-negative questions: a negative question
    refusing is correct behavior (that's check_refusal's job), not a false
    refusal, regardless of what retrieval_succeeded says.
    """
    if question_type == "negative":
        return False
    return check_refusal(generated_answer) and retrieval_succeeded


# Our generation prompt (generation/local_ollama.py) instructs a specific
# bare inline citation format: [paper_id, section_title] -- two
# comma-separated values, no field-name labels, no quotes, no reference to
# the "[Evidence N]" markers used only in the evidence block shown to the
# model. This is a rule-based structural check (not a judge call), since
# it's testing exact prompt-format compliance, not a subjective quality.
#
# The parsing primitives (_BRACKET_CONTENT_RE, _is_citation_attempt,
# _extract_citation_paper_id) live in generation/citation_guard.py, not
# here -- Step 11 turned this diagnostic logic into a runtime repair guard
# wired into generation/local_ollama.py, and generation_judge.py already
# imports OllamaBackend from local_ollama, so keeping the canonical
# definitions here would create a circular import once local_ollama also
# needs them. Imported above; re-used as-is, not re-implemented.


def check_citation_format(generated_answer: str) -> dict:
    """Structural check of citation markers against the instructed
    '[paper_id, section_title]' format. Returns {valid, issues} -- issues
    is a list of human-readable strings, empty when valid is True.

    Only brackets that look like a citation attempt (see
    _is_citation_attempt) are checked -- incidental brackets copied from
    evidence text (in-text reference numbers, math notation) are ignored."""
    issues = []

    open_count = generated_answer.count("[")
    close_count = generated_answer.count("]")
    if open_count != close_count:
        issues.append(f"Unbalanced brackets: {open_count} '[' vs {close_count} ']'")

    all_matches = _BRACKET_CONTENT_RE.findall(generated_answer)
    matches = [text for text in all_matches if _is_citation_attempt(text)]
    if not matches:
        issues.append("No citation markers found in the answer")

    prev = None
    for text in matches:
        if "=" in text:
            issues.append(f"Leaked field-name label (expected bare 'paper_id, section_title', not 'field=value'): '[{text}]'")
        elif text.count(",") != 1:
            issues.append(f"Citation does not have exactly two comma-separated fields (paper_id, section_title): '[{text}]'")

        if "paper_title" in text.lower():
            issues.append(f"Citation includes paper_title, which is not part of the instructed [paper_id, section_title] format: '[{text}]'")

        if _EVIDENCE_LABEL_RE.search(text):
            issues.append(f"Citation references an '[Evidence N]' label instead of [paper_id, section_title]: '[{text}]'")

        if text == prev and text != "":
            issues.append(f"Duplicate citation marker repeated back-to-back: '[{text}]'")
        prev = text

    return {"valid": len(issues) == 0, "issues": issues}


def check_citation_accuracy(generated_answer: str, evidence_chunks: list) -> dict:
    """Cross-checks every citation marker's claimed paper_id against the
    paper_ids actually present in evidence_chunks (the evidence passed to
    the model for this query).

    This is deliberately independent of check_citation_format: a citation
    can be well-formatted but fabricated (points to a paper never shown to
    the model), or badly-formatted but accurate (e.g. "[Evidence 3]" instead
    of the real paper_id, when Evidence 3 genuinely is the source used).
    Only the former -- a paper_id outside the evidence set -- is a
    hallucinated citation; the latter is a formatting problem for
    check_citation_format to catch, not an accuracy problem.

    Only brackets that look like a citation attempt (see
    _is_citation_attempt) are checked -- incidental brackets copied from
    evidence text (in-text reference numbers, math notation) are ignored,
    since they were never a citation the model tried to make.

    Returns {accurate, hallucinated_citations, valid_citations}, where the
    citation lists hold {citation_text, resolved_paper_id} dicts, deduped.
    """
    valid_paper_ids = {chunk["paper_id"] for chunk in evidence_chunks}
    all_matches = _BRACKET_CONTENT_RE.findall(generated_answer)
    matches = [text for text in all_matches if _is_citation_attempt(text)]

    hallucinated = []
    valid = []
    seen = set()

    for text in matches:
        paper_id = _extract_citation_paper_id(text, evidence_chunks)
        key = (text, paper_id)
        if key in seen:
            continue
        seen.add(key)

        entry = {"citation_text": f"[{text}]", "resolved_paper_id": paper_id}
        if paper_id and paper_id in valid_paper_ids:
            valid.append(entry)
        else:
            hallucinated.append(entry)

    return {
        "accurate": len(hallucinated) == 0,
        "hallucinated_citations": hallucinated,
        "valid_citations": valid,
    }
