"""Unit tests for generic retrieval evaluation metrics.

Tests run offline without external dependencies to verify:
- first gold rank discovery
- hit@k calculation (Recall@K)
- macro-average Recall@K calculation
- MRR (Mean Reciprocal Rank) calculation
"""

import math

from tgh.embeddings.metrics import (
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)


class TestRetrievalMetrics:
    """Test suite for retrieval evaluation metrics."""

    def test_compute_first_gold_rank(self) -> None:
        """Verify first gold rank discovery."""
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
        gold = {"doc_C", "doc_E"}

        rank = compute_first_gold_rank(retrieved, gold)
        assert rank == 3  # doc_C is at index 2 (1-indexed rank 3)

        # No match
        assert compute_first_gold_rank(retrieved, {"doc_Z"}) is None

    def test_is_hit_at_k(self) -> None:
        """Verify hit@k checks."""
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D"]
        gold = {"doc_C"}

        assert not is_hit_at_k(retrieved, gold, k=1)
        assert not is_hit_at_k(retrieved, gold, k=2)
        assert is_hit_at_k(retrieved, gold, k=3)
        assert is_hit_at_k(retrieved, gold, k=5)
        assert not is_hit_at_k(retrieved, gold, k=0)

    def test_compute_recalls(self) -> None:
        """Verify Recall@K macro-average computation across queries."""
        # Query 1: gold hit at rank 1
        # Query 2: gold hit at rank 3
        # Query 3: no gold hit in top 5 (hit at rank 8)
        retrieved_list = [
            ["doc_1", "doc_2", "doc_3"],
            ["doc_X", "doc_Y", "doc_2"],
            ["a", "b", "c", "d", "e", "f", "g", "doc_3"],
        ]
        gold_list = [{"doc_1"}, {"doc_2"}, {"doc_3"}]

        recalls = compute_recalls(
            retrieved_list, gold_list, k_list=(1, 2, 5, 10)
        )
        # Recall@1: query 1 hits -> 1/3 = 0.3333
        assert math.isclose(recalls[1], 1 / 3, rel_tol=1e-4)
        # Recall@2: query 1 hits -> 1/3
        assert math.isclose(recalls[2], 1 / 3, rel_tol=1e-4)
        # Recall@5: query 1 and query 2 hit -> 2/3 = 0.6667
        assert math.isclose(recalls[5], 2 / 3, rel_tol=1e-4)
        # Recall@10: query 1, 2, and 3 hit -> 3/3 = 1.0
        assert math.isclose(recalls[10], 1.0, rel_tol=1e-4)

    def test_compute_mrr(self) -> None:
        """Verify MRR calculation."""
        # Query 1: rank 1 -> RR = 1.0
        # Query 2: rank 2 -> RR = 0.5
        # Query 3: rank 4 -> RR = 0.25
        # Query 4: no match -> RR = 0.0
        retrieved_list = [
            ["gold_1"],
            ["miss", "gold_2"],
            ["miss", "miss", "miss", "gold_3"],
            ["miss", "miss"],
        ]
        gold_list = [{"gold_1"}, {"gold_2"}, {"gold_3"}, {"gold_4"}]

        mrr = compute_mrr(retrieved_list, gold_list)
        # Expected: (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 0.4375
        assert math.isclose(mrr, 0.4375, rel_tol=1e-5)
