"""Layer 1: Embedding providers and retrieval evaluation metrics."""

from tgh.embeddings.base import EmbeddingProvider
from tgh.embeddings.gemini import GeminiEmbeddingProvider
from tgh.embeddings.gemma import GemmaEmbeddingProvider
from tgh.embeddings.metrics import (
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)
from tgh.embeddings.nomic import NomicEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "GemmaEmbeddingProvider",
    "NomicEmbeddingProvider",
    "compute_first_gold_rank",
    "compute_mrr",
    "compute_recalls",
    "is_hit_at_k",
]
