"""Base interface and protocol for vector embedding providers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for vector embedding providers.

    Decouples the retrieval and graph layers from specific embedding libraries,
    enabling pluggable local open-weight models.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for a batch of texts."""
        ...

    def format_query(self, query: str) -> str:
        """Format a search query according to the model's instruction requirements."""
        ...

    def format_document(self, document: str) -> str:
        """Format document text for indexing according to model specifications."""
        ...
