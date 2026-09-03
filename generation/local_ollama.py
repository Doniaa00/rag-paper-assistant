"""
Step 8: Local Ollama generation backend.

Implements GenerationBackend by calling a locally-running Ollama server's
/api/generate endpoint. This is the ONLY module that should talk to Ollama
directly -- everything else in the project goes through
generation.interface.GenerationBackend.
"""

import requests

from generation.interface import GenerationBackend

OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 300

PROMPT_TEMPLATE = """You are a research assistant answering questions about academic papers on fraud detection, anomaly detection, and class-imbalance learning.

Answer the question using ONLY the evidence provided below. Do not use any outside knowledge.

Rules:
1. Every claim you make must be supported by one of the evidence chunks below.
2. Cite the specific chunk supporting each claim inline, immediately after the claim, using ONLY this exact bare format: [paper_id, section_title]
   - Correct: [1901.03407, Autoencoders]
   - WRONG -- do not use field=value syntax: [paper_id=1901.03407, section_title="Autoencoders"]
   - WRONG -- do not include paper_title: [1901.03407, "Autoencoders Survey", Autoencoders]
   - WRONG -- do not cite the internal [Evidence N] markers below, they are for your reference only, never copy them into your answer: [Evidence 1]
   - WRONG -- never repeat the exact same citation twice in a row: [1901.03407, Autoencoders][1901.03407, Autoencoders]
   If a sentence draws on more than one chunk, cite each one once: [paper_id_1, section_title_1][paper_id_2, section_title_2].
   Example of a correctly-cited sentence: "Autoencoders flag anomalies by their high reconstruction error on out-of-distribution samples [1901.03407, Autoencoders]."
3. If the evidence below does not actually answer the question, say so explicitly ("The provided evidence does not contain sufficient information to answer this question.") rather than guessing or using outside knowledge.

EVIDENCE:
{evidence_block}

QUESTION: {query}

ANSWER:"""


def _format_evidence_block(evidence_chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(evidence_chunks, start=1):
        parts.append(
            f"[Evidence {i}] paper_id={chunk['paper_id']}, "
            f"paper_title=\"{chunk['paper_title']}\", section_title=\"{chunk['section_title']}\"\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


class OllamaBackend(GenerationBackend):
    def __init__(self, model: str = "qwen2.5:3b-instruct"):
        self.model = model

    def generate(self, query: str, evidence_chunks: list) -> str:
        evidence_block = _format_evidence_block(evidence_chunks)
        prompt = PROMPT_TEMPLATE.format(evidence_block=evidence_block, query=query)
        return self.generate_raw(prompt)

    def generate_raw(self, prompt: str) -> str:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def generate_with_metrics(self, query: str, evidence_chunks: list) -> dict:
        """Same as generate(), but also returns Ollama's own internal timing
        breakdown for Step 9d's systems metrics.

        Ollama's non-streaming response already reports nanosecond-precision
        internal timings (load_duration, prompt_eval_duration, eval_duration,
        eval_count) even with stream=False, so this doesn't need to switch to
        streaming mode to get TTFT -- time-to-first-token is, by definition,
        the time from request start until the model begins emitting tokens,
        which is exactly load_duration + prompt_eval_duration (the first
        generated token follows immediately after prompt eval completes).
        Steady-state throughput is eval_count / eval_duration, i.e. tokens
        actually generated divided by time spent generating them (excluding
        prompt processing and any model load).

        This is Ollama-specific instrumentation, kept concrete on
        OllamaBackend rather than promoted to the GenerationBackend
        interface -- a future hosted-API backend may not expose the same
        granular timing fields, and Step 9d is instrumenting this deployed
        stack specifically, not defining a portable timing contract.
        """
        evidence_block = _format_evidence_block(evidence_chunks)
        prompt = PROMPT_TEMPLATE.format(evidence_block=evidence_block, query=query)

        resp = requests.post(
            OLLAMA_URL,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        load_s = data.get("load_duration", 0) / 1e9
        prompt_eval_s = data.get("prompt_eval_duration", 0) / 1e9
        eval_s = data.get("eval_duration", 0) / 1e9
        eval_count = data.get("eval_count", 0)

        return {
            "answer": data["response"].strip(),
            "ttft_seconds": load_s + prompt_eval_s,
            "generation_seconds": eval_s,
            "tokens_per_second": (eval_count / eval_s) if eval_s > 0 else 0.0,
            "eval_count": eval_count,
            "load_seconds": load_s,
        }
