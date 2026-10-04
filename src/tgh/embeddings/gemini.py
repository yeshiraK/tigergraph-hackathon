"""Gemini Embedding 2 provider using Google Generative Language API.

Implements batchEmbedContents and embedContent with automatic retry,
exponential backoff, 768-dimensional output, and unit L2 normalization.
"""

import os
import random
import time
from pathlib import Path

import requests

from tgh.embeddings.base import EmbeddingProvider

DEFAULT_MODEL_NAME = "gemini-embedding-2"
DEFAULT_TARGET_DIM = 768
DOC_TEMPLATE = "title: {title} | text: {text}"
QUERY_TEMPLATE = "{query}"


def parse_retry_after(resp: requests.Response) -> float | None:
    """Parse Retry-After header or Gemini error details for backoff delay."""
    retry_header = resp.headers.get("Retry-After")
    if retry_header:
        try:
            return float(retry_header)
        except ValueError:
            pass

    try:
        data = resp.json()
        details = data.get("error", {}).get("details", [])
        for item in details:
            if "retryDelay" in item:
                delay_str = str(item["retryDelay"]).rstrip("s")
                return float(delay_str)
    except Exception:
        pass
    return None


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Remote API embedding provider using Gemini Embedding 2."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_TARGET_DIM,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        default_batch_size: int = 25,
        inter_batch_pause: float = 2.0,
        max_retries: int = 5,
    ) -> None:
        """Initialize the Gemini embedding provider."""
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self._model_name = model_name
        self._api_model = f"models/{model_name}"
        self._dimension = dimension
        self._base_url = base_url
        self._default_batch_size = default_batch_size
        self._inter_batch_pause = inter_batch_pause
        self._max_retries = max_retries

        if api_key:
            self._api_key = api_key
        else:
            self._api_key = self._load_api_key()

        # Telemetry counters
        self.stats = {
            "http_requests": 0,
            "texts_embedded": 0,
            "retries": 0,
            "429_count": 0,
            "5xx_count": 0,
            "successful_batches": 0,
        }

    def _load_api_key(self) -> str:
        """Read GEMINI_API_KEY from environment or project .env."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key.strip().strip("\"'")

        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        env_file = repo_root / ".env"
        if env_file.is_file():
            with env_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip("\"'").strip()
                        if val:
                            return val
        raise ValueError("GEMINI_API_KEY not found in environment or .env")

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Target output dimensionality."""
        return self._dimension

    def format_query(self, query: str) -> str:
        """Format query text."""
        return query.strip()

    def format_document(self, document: str, title: str = "none") -> str:
        """Format document text with title prefix."""
        clean_title = title.strip() if title and title.strip() else "none"
        return DOC_TEMPLATE.format(title=clean_title, text=document.strip())

    def _call_batch_api(self, texts: list[str]) -> list[list[float]]:
        """Execute single batchEmbedContents call with jittered backoff."""
        url = f"{self._base_url}/{self._api_model}:batchEmbedContents"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        payload = {
            "requests": [
                {
                    "model": self._api_model,
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self._dimension,
                }
                for t in texts
            ]
        }

        for attempt in range(self._max_retries):
            self.stats["http_requests"] += 1
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=60
                )
            except requests.RequestException as e:
                self.stats["retries"] += 1
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"Network error after {self._max_retries} attempts: {e}"
                    ) from e
                wait = 2.0 * (2**attempt) + random.uniform(0.1, 1.0)
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                data = resp.json()
                raw_embs = data.get("embeddings", [])
                embs = [item["values"] for item in raw_embs]
                self.stats["texts_embedded"] += len(embs)
                self.stats["successful_batches"] += 1
                return embs

            elif resp.status_code == 429:
                self.stats["429_count"] += 1
                self.stats["retries"] += 1
                retry_after = parse_retry_after(resp)
                backoff = (
                    retry_after
                    if retry_after is not None
                    else (10.0 * (attempt + 1) + random.uniform(0.5, 2.0))
                )
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"Rate limit exceeded (429) after {self._max_retries} retries"
                    )
                time.sleep(backoff)

            elif resp.status_code >= 500:
                self.stats["5xx_count"] += 1
                self.stats["retries"] += 1
                backoff = 2.0 * (2**attempt) + random.uniform(0.1, 1.0)
                if attempt == self._max_retries - 1:
                    raise RuntimeError(
                        f"Server error ({resp.status_code}) after "
                        f"{self._max_retries} retries: {resp.text}"
                    )
                time.sleep(backoff)

            else:
                sanitized = resp.text.replace(self._api_key, "[REDACTED]")
                raise RuntimeError(
                    f"Gemini API error (HTTP {resp.status_code}): {sanitized}"
                )

        raise RuntimeError("Exhausted retries without response")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for a batch of texts."""
        if not texts:
            return []

        all_embs: list[list[float]] = []
        bsz = self._default_batch_size

        for i in range(0, len(texts), bsz):
            chunk_slice = texts[i : i + bsz]
            embs = self._call_batch_api(chunk_slice)
            all_embs.extend(embs)
            if self._inter_batch_pause > 0 and i + bsz < len(texts):
                time.sleep(self._inter_batch_pause)

        return all_embs

    def embed_text(self, text: str) -> list[float]:
        """Generate normalized embedding vector for single text."""
        return self.embed_batch([text])[0]
