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
2. Cite the specific chunk supporting each claim inline, immediately after the claim, in this exact format: [paper_id, section_title]. If a sentence draws on more than one chunk, cite all of them, e.g. [paper_id_1, section_title_1][paper_id_2, section_title_2].
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

        resp = requests.post(
            OLLAMA_URL,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
