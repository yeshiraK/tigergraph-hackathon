"""Nomic Embed Text v1.5 local CPU provider using Hugging Face Transformers.

Implements mean pooling, LayerNorm, and unit L2 normalization for cosine search.
Supports dynamic context up to 2048 tokens and native 768 dimensions.
"""

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tgh.embeddings.base import EmbeddingProvider

DEFAULT_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_TARGET_DIM = 768
DEFAULT_MAX_LENGTH = 2048
QUERY_PREFIX = "search_query: {query}"
DOC_PREFIX = "search_document: {document}"


def mean_pooling(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute attention-weighted mean pooling across non-padded tokens."""
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class NomicEmbeddingProvider(EmbeddingProvider):
    """Local CPU embedding provider using nomic-ai/nomic-embed-text-v1.5."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        dimension: int = DEFAULT_TARGET_DIM,
        max_length: int = DEFAULT_MAX_LENGTH,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the Nomic embedding provider.

        Args:
            model_id: HuggingFace model hub identifier.
            dimension: Target vector dimensionality (768).
            max_length: Maximum token length for truncation (2048).
            model: Optional pre-loaded or mocked model instance.
            tokenizer: Optional pre-loaded or mocked tokenizer instance.
            device: Target device for execution ('cpu').
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        if dimension > 768:
            raise ValueError(
                f"Dimension {dimension} exceeds native model dimension 768"
            )

        self._model_id = model_id
        self._dimension = dimension
        self._max_length = max_length
        self._device = device

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        if model is not None:
            self._model = model
        else:
            self._model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self._model.eval()
            self._model.to(self._device)

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def format_query(self, query: str) -> str:
        """Format user question according to Nomic search_query prefix."""
        return QUERY_PREFIX.format(query=query)

    def format_document(self, document: str) -> str:
        """Format passage according to Nomic search_document prefix."""
        return DOC_PREFIX.format(document=document)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 1,
    ) -> list[list[float]]:
        """Generate normalized 768D embeddings for a batch of texts on CPU."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                out = self._model(**encoded)
                pooled = mean_pooling(out[0], encoded["attention_mask"])
                ln = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
                # Truncate to matryoshka dimension if dimension < 768
                if self._dimension < 768:
                    ln = ln[:, : self._dimension]
                normed = F.normalize(ln, p=2, dim=1)

            all_vectors.extend(normed.cpu().tolist())

        return all_vectors
