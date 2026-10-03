"""Tokenizer interface and deterministic fallback token counters.

Isolates token estimation from chunking logic to support configurable tokenizers
when embedding models or production tokenizers are integrated.
"""

import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for counting tokens in a text span."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        ...


class ApproximateCharTokenizer:
    """Deterministic character-based token estimator for structural testing.

    This is an approximate character-based tokenizer defaulting to 4.0 characters
    per token (matching the corpus approx_tokens reference statistic).

    It is NOT an exact embedding-model tokenizer. The production embedding
    tokenizer will be supplied through the Tokenizer protocol once the Qwen
    embedding model is selected. The current approximation is for deterministic
    structural chunking and offline testing only.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        """Count approximate tokens from character length."""
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)
