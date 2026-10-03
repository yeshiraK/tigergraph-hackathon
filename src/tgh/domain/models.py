"""Domain entities and value objects for the Agentic GraphRAG system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Deterministic chunk representation for retrieval and graph loading.

    Attributes:
        document_id: Unique identifier of the source document.
        chunk_id: Deterministic identifier derived from document_id and chunk_index.
        chunk_index: 0-indexed sequential position of the chunk in the document.
        source_start: Character offset start in the original document text.
        source_end: Character offset end in the original document text.
        text: Exact, unmodified slice of source text.
        token_count: Number of tokens computed for this chunk.
    """

    document_id: str
    chunk_id: str
    chunk_index: int
    source_start: int
    source_end: int
    text: str
    token_count: int
