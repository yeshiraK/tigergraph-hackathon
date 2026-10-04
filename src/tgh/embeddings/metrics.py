"""Retrieval evaluation metrics for benchmark assessment."""

from collections.abc import Sequence


def compute_first_gold_rank(
    retrieved_doc_ids: Sequence[str],
    gold_doc_ids: set[str],
) -> int | None:
    """Find the 1-indexed rank of the first retrieved document in gold_doc_ids.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document identifiers.
        gold_doc_ids: Set of acceptable ground-truth document identifiers.

    Returns:
        1-indexed integer rank if a hit is found, else None.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return rank
    return None


def is_hit_at_k(
    retrieved_doc_ids: Sequence[str],
    gold_doc_ids: set[str],
    k: int,
) -> bool:
    """Determine whether at least one gold document is retrieved in the top K.

    Args:
        retrieved_doc_ids: Ordered list of retrieved document identifiers.
        gold_doc_ids: Set of acceptable ground-truth document identifiers.
        k: Cutoff rank (e.g., 1, 5, 10, 20).

    Returns:
        True if at least one doc in retrieved_doc_ids[:k] is in gold_doc_ids.
    """
    if k <= 0 or not gold_doc_ids:
        return False
    return any(doc_id in gold_doc_ids for doc_id in retrieved_doc_ids[:k])


def compute_recalls(
    retrieved_doc_ids_list: Sequence[Sequence[str]],
    gold_doc_ids_list: Sequence[set[str]],
    k_list: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[int, float]:
    """Compute Recall@K (hit rate) across multiple evaluation questions.

    Args:
        retrieved_doc_ids_list: List of retrieved doc ID lists per question.
        gold_doc_ids_list: List of gold doc ID sets per question.
        k_list: Cutoff ranks to evaluate.

    Returns:
        Dictionary mapping each K to its macro-average Recall@K score (0.0 to 1.0).
    """
    total = len(retrieved_doc_ids_list)
    if total == 0:
        return dict.fromkeys(k_list, 0.0)

    results: dict[int, float] = {}
    for k in k_list:
        hits = sum(
            1
            for ret, gold in zip(
                retrieved_doc_ids_list, gold_doc_ids_list, strict=True
            )
            if is_hit_at_k(ret, gold, k)
        )
        results[k] = hits / total
    return results


def compute_mrr(
    retrieved_doc_ids_list: Sequence[Sequence[str]],
    gold_doc_ids_list: Sequence[set[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across all evaluation questions.

    Args:
        retrieved_doc_ids_list: List of retrieved doc ID lists per question.
        gold_doc_ids_list: List of gold doc ID sets per question.

    Returns:
        Mean Reciprocal Rank float value (0.0 to 1.0).
    """
    total = len(retrieved_doc_ids_list)
    if total == 0:
        return 0.0

    reciprocal_ranks: list[float] = []
    for ret, gold in zip(retrieved_doc_ids_list, gold_doc_ids_list, strict=True):
        rank = compute_first_gold_rank(ret, gold)
        if rank is not None and rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / total
