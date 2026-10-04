"""EmbeddingGemma-300M local CPU provider using Hugging Face Transformers.

Implements mean pooling, 2-stage dense projection to 768 dimensions,
and unit L2 normalization for cosine similarity search.
"""

import math
from typing import Any

import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from tgh.embeddings.base import EmbeddingProvider

DEFAULT_MODEL_ID = "unsloth/embeddinggemma-300m"
DEFAULT_TARGET_DIM = 768
DEFAULT_MAX_LENGTH = 2048
QUERY_PREFIX = "task: search result | query: {query}"
DOC_TEMPLATE = "title: {title} | text: {text}"


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


class GemmaEmbeddingProvider(EmbeddingProvider):
    """Local CPU embedding provider using EmbeddingGemma-300M."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        dimension: int = DEFAULT_TARGET_DIM,
        max_length: int = DEFAULT_MAX_LENGTH,
        model: Any | None = None,
        tokenizer: Any | None = None,
        w_proj: torch.Tensor | None = None,
    ) -> None:
        """Initialize the Gemma embedding provider.

        Args:
            model_id: HuggingFace model hub identifier.
            dimension: Target vector dimensionality (768).
            max_length: Maximum token length for truncation (2048).
            model: Optional pre-loaded or mocked model instance.
            tokenizer: Optional pre-loaded or mocked tokenizer instance.
            w_proj: Optional pre-computed (768, 768) projection matrix.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._model_name = model_id
        self._dimension = dimension
        self._max_length = max_length

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, padding_side="right"
            )

        if model is not None:
            self._model = model
        else:
            self._model = AutoModel.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
            self._model.eval()

        if w_proj is not None:
            self._w_proj = w_proj
        else:
            # Download and combine dense projection layers
            p2 = hf_hub_download(model_id, "2_Dense/model.safetensors")
            w2 = safetensors.torch.load_file(p2)["linear.weight"]  # (3072, 768)
            p3 = hf_hub_download(model_id, "3_Dense/model.safetensors")
            w3 = safetensors.torch.load_file(p3)["linear.weight"]  # (768, 3072)
            self._w_proj = torch.mm(w3, w2).T  # (768, 768)

    @property
    def model_name(self) -> str:
        """Identifier of the underlying embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dimension

    def format_query(self, query: str) -> str:
        """Format query using task instruction."""
        return QUERY_PREFIX.format(query=query.strip())

    def format_document(self, document: str, title: str = "none") -> str:
        """Format document text with title prefix."""
        clean_title = title.strip() if title and title.strip() else "none"
        return DOC_TEMPLATE.format(title=clean_title, text=document.strip())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for a batch of texts.

        Processes items unpadded (batch_size=1) for maximal CPU throughput
        and minimal memory overhead.

        Args:
            texts: List of formatted text strings to embed.

        Returns:
            List of 768-dimensional float lists with unit L2 norm.
        """
        if not texts:
            return []

        result: list[list[float]] = []
        for i, text in enumerate(texts):
            inputs = self._tokenizer(
                [text],
                max_length=self._max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.inference_mode():
                outputs = self._model(**inputs)
                pooled = mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                projected = torch.mm(pooled, self._w_proj)
                if self._dimension < projected.shape[-1]:
                    projected = projected[:, : self._dimension]
                normalized = F.normalize(projected, p=2, dim=-1)

            vec = normalized[0].tolist()
            if not all(math.isfinite(x) for x in vec):
                raise ValueError(
                    f"Non-finite value encountered in vector at item {i}"
                )
            result.append(vec)

        return result

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a single text."""
        return self.embed_batch([text])[0]
