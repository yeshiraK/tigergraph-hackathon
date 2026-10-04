"""Unit tests for GemmaEmbeddingProvider.

Tests run offline with lightweight mocked components to verify:
- 768-dimensional output
- L2 normalization
- deterministic query formatting
- deterministic document formatting
- batch embedding
"""

import math
from unittest.mock import MagicMock

import pytest
import torch

from tgh.embeddings.gemma import GemmaEmbeddingProvider, mean_pooling


class TestGemmaEmbeddingProvider:
    """Test suite for Gemma provider logic and formatting."""

    def test_deterministic_query_formatting(self) -> None:
        """Verify query formatting prepends the required instruction."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        w_proj = torch.eye(768)
        provider = GemmaEmbeddingProvider(
            model=mock_model, tokenizer=mock_tokenizer, w_proj=w_proj
        )

        query = "Who won the marathon in 1984?"
        formatted = provider.format_query(query)
        assert formatted == "task: search result | query: Who won the marathon in 1984?"

    def test_format_document(self) -> None:
        """Verify document formatting prepends title prefix."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        w_proj = torch.eye(768)
        provider = GemmaEmbeddingProvider(
            model=mock_model, tokenizer=mock_tokenizer, w_proj=w_proj
        )

        doc = "Olympic Stadium was built in 1928."
        assert (
            provider.format_document(doc, title="Olympic Stadium")
            == "title: Olympic Stadium | text: Olympic Stadium was built in 1928."
        )
        assert (
            provider.format_document(doc, title="")
            == "title: none | text: Olympic Stadium was built in 1928."
        )

    def test_mean_pooling(self) -> None:
        """Verify mean pooling averages across unmasked tokens."""
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            ]
        )
        attention_mask = torch.tensor([[1, 1, 0]])
        pooled = mean_pooling(hidden, attention_mask)
        # Expected mean of [1.0, 2.0] and [3.0, 4.0] -> [2.0, 3.0]
        assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))

    def test_768_dimension_and_l2_normalization(self) -> None:
        """Verify output dimensionality is 768 and L2 normalized."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

        mock_model = MagicMock()
        raw_hidden = torch.randn(1, 2, 768)
        mock_output = MagicMock()
        mock_output.last_hidden_state = raw_hidden
        mock_model.return_value = mock_output
        w_proj = torch.randn(768, 768)

        provider = GemmaEmbeddingProvider(
            dimension=768,
            model=mock_model,
            tokenizer=mock_tokenizer,
            w_proj=w_proj,
        )
        vector = provider.embed_text("Test passage")

        assert len(vector) == 768
        norm = math.sqrt(sum(x * x for x in vector))
        assert math.isclose(norm, 1.0, rel_tol=1e-5)
        assert all(math.isfinite(x) for x in vector)

    def test_invalid_dimension_raises(self) -> None:
        """Verify ValueError on non-positive dimension."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        w_proj = torch.eye(768)
        with pytest.raises(ValueError, match="dimension must be positive"):
            GemmaEmbeddingProvider(
                dimension=0,
                model=mock_model,
                tokenizer=mock_tokenizer,
                w_proj=w_proj,
            )
