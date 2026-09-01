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
