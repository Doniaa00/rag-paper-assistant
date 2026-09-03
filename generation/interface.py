"""
Step 8: Generation interface.

Defines the swappable abstraction every generation backend must implement.
Retrieval code (and anything else in this project) should only ever talk to
a GenerationBackend -- never to Ollama, an HTTP API, or any other backend
directly -- so a future hosted-API backend can be swapped in without
touching retrieval code.
"""

from abc import ABC, abstractmethod


class GenerationBackend(ABC):
    @abstractmethod
    def generate(self, query: str, evidence_chunks: list) -> str:
        """Generate an answer to `query`, grounded strictly in `evidence_chunks`.

        evidence_chunks: list of dicts, each expected to have at least
        paper_id, paper_title, section_title, and chunk_text.

        Returns the generated answer text (expected to include inline
        citations per the backend's prompt design).
        """
        raise NotImplementedError

    @abstractmethod
    def generate_raw(self, prompt: str) -> str:
        """Send an arbitrary prompt straight to the model, with no RAG
        answer-generation template applied. Used by non-RAG consumers of the
        backend -- e.g. the Step 9c LLM-judge, which needs its own judging
        prompt shape rather than the query+evidence_chunks answer format.

        Still goes through the backend abstraction (not a direct API call),
        so a future hosted-API backend swap-in only needs to implement this
        once here, not have every judge/caller special-cased.
        """
        raise NotImplementedError
