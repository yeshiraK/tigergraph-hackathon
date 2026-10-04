"""Unit tests for GeminiEmbeddingProvider."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tgh.embeddings.gemini import GeminiEmbeddingProvider, parse_retry_after


class TestGeminiEmbeddingProvider:
    """Test suite for Gemini provider logic and formatting."""

    def test_deterministic_query_formatting(self) -> None:
        """Verify query formatting."""
        provider = GeminiEmbeddingProvider(api_key="test-key")
        query = "Who won the marathon in 1984?"
        assert provider.format_query(query) == "Who won the marathon in 1984?"

    def test_format_document(self) -> None:
        """Verify document formatting prepends title prefix."""
        provider = GeminiEmbeddingProvider(api_key="test-key")
        doc = "Olympic Stadium was built in 1928."
        assert (
            provider.format_document(doc, title="Olympic Stadium")
            == "title: Olympic Stadium | text: Olympic Stadium was built in 1928."
        )
        assert (
            provider.format_document(doc, title="")
            == "title: none | text: Olympic Stadium was built in 1928."
        )

    def test_parse_retry_after_header(self) -> None:
        """Verify Retry-After header parsing."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.headers = {"Retry-After": "12.5"}
        assert parse_retry_after(mock_resp) == 12.5

    def test_parse_retry_after_details(self) -> None:
        """Verify retryDelay in error details parsing."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "error": {"details": [{"retryDelay": "25.0s"}]}
        }
        assert parse_retry_after(mock_resp) == 25.0

    @patch("requests.post")
    def test_embed_batch_success(self, mock_post: MagicMock) -> None:
        """Verify batch embedding with mocked API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Return two 768D vectors
        vec1 = [0.0] * 768
        vec1[0] = 1.0
        vec2 = [0.0] * 768
        vec2[1] = 1.0
        mock_resp.json.return_value = {
            "embeddings": [{"values": vec1}, {"values": vec2}]
        }
        mock_post.return_value = mock_resp

        provider = GeminiEmbeddingProvider(api_key="test-key", default_batch_size=25)
        vectors = provider.embed_batch(["text1", "text2"])

        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        assert len(vectors[1]) == 768
        assert provider.stats["successful_batches"] == 1
        assert provider.stats["texts_embedded"] == 2

    def test_invalid_dimension_raises(self) -> None:
        """Verify ValueError on non-positive dimension."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            GeminiEmbeddingProvider(api_key="test-key", dimension=0)
