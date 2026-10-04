"""Generic offline retrieval evaluation benchmark for EmbeddingProvider candidates.

Evaluates document-level retrieval metrics (Recall@1, 5, 10, 20 and MRR)
against the 100 public benchmark questions in data/benchmarks/eval_public.jsonl.
Strictly offline and ₹0 budget. Never touches hidden evaluation benchmark.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tgh.embeddings.base import EmbeddingProvider
from tgh.embeddings.metrics import (
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)
from tgh.ingestion.chunker import SemanticChunker


def load_public_benchmark(benchmark_path: Path) -> list[dict]:
    """Load and validate the 100 public evaluation questions."""
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    records = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            records.append(rec)

    if len(records) != 100:
        raise ValueError(
            f"Expected exactly 100 public questions, found {len(records)}"
        )

    # Verify required schema fields
    required_fields = {"qid", "question", "qtype", "gold_doc_ids"}
    for r in records:
        missing = required_fields - set(r.keys())
        if missing:
            raise ValueError(
                f"Record {r.get('qid')} missing required fields: {missing}"
            )

    return records


def load_corpus_chunks(
    corpus_path: Path, max_chunks: int | None = None
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Extract production chunks from corpus using SemanticChunker.

    Returns:
        chunk_ids: list of chunk IDs
        chunk_doc_ids: list of parent document IDs
        chunk_texts: list of chunk text slices
        doc_titles: mapping of doc_id -> title
    """
    chunker = SemanticChunker(target_tokens=768)
    chunk_ids: list[str] = []
    chunk_doc_ids: list[str] = []
    chunk_texts: list[str] = []
    doc_titles: dict[str, str] = {}

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            doc_id = rec["doc_id"]
            title = rec.get("title", "")
            doc_titles[doc_id] = title
            chunks = chunker.chunk_document(doc_id, rec["text"])

            for c in chunks:
                chunk_ids.append(c.chunk_id)
                chunk_doc_ids.append(c.document_id)
                chunk_texts.append(c.text)
                if max_chunks and len(chunk_ids) >= max_chunks:
                    return chunk_ids, chunk_doc_ids, chunk_texts, doc_titles

    return chunk_ids, chunk_doc_ids, chunk_texts, doc_titles


def run_benchmark(
    provider: EmbeddingProvider,
    max_chunks: int | None = None,
    batch_size: int = 1,
    cache_dir: Path | None = None,
) -> dict:
    """Execute the offline retrieval evaluation for an EmbeddingProvider candidate."""
    repo_root = Path(__file__).resolve().parent.parent
    bench_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"
    corpus_file = repo_root / "data" / "raw" / "corpus.jsonl"

    if cache_dir is None:
        cache_dir = repo_root / "experiments" / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = provider.model_name.replace("/", "_").replace(" ", "_")
    cache_file = cache_dir / f"{slug}_chunk_embeddings_cache.npz"

    print("=== Step 1: Loading Public Benchmark ===")
    questions = load_public_benchmark(bench_file)
    print(f"Verified {len(questions)} public benchmark questions.")

    print(f"\n=== Step 2: Evaluating Provider: {provider.model_name} ===")
    print(f"Provider dimension: {provider.dimension}")

    print("\n=== Step 3: Generating Query Embeddings ===")
    t0 = time.perf_counter()
    query_vectors: list[list[float]] = []
    for q in questions:
        formatted = provider.format_query(q["question"])
        vec = provider.embed_text(formatted)
        query_vectors.append(vec)
    query_time = time.perf_counter() - t0
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32)
    print(
        f"Encoded {len(query_vectors)} queries in {query_time:.2f}s "
        f"({query_time / len(query_vectors) * 1000:.1f} ms/query)"
    )

    print("\n=== Step 4: Loading Corpus Chunks ===")
    t0 = time.perf_counter()
    chunk_ids, chunk_doc_ids, chunk_texts, doc_titles = load_corpus_chunks(
        corpus_file, max_chunks=max_chunks
    )
    total_chunks = len(chunk_ids)
    print(
        f"Loaded {total_chunks} production chunks from corpus in "
        f"{time.perf_counter() - t0:.2f}s."
    )

    print("\n=== Step 5: Verification Batch Test ===")
    test_batch = chunk_texts[: min(16, total_chunks)]
    t_test0 = time.perf_counter()
    test_vecs = provider.embed_batch(test_batch)
    t_test_dur = time.perf_counter() - t_test0

    test_tensor = torch.tensor(test_vecs, dtype=torch.float32)
    assert test_tensor.shape == (
        len(test_batch),
        provider.dimension,
    ), f"Bad shape: {test_tensor.shape}, expected (*, {provider.dimension})"
    assert torch.isfinite(test_tensor).all().item(), "Encountered non-finite values"
    assert (test_tensor != 0).any().item(), "Encountered all-zero vectors"
    norms = torch.norm(test_tensor, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Not normalized"
    ms_per_chunk = (t_test_dur / len(test_batch)) * 1000
    print(
        f"Verification batch verified: shape={test_tensor.shape}, all finite, "
        f"unit normalized. Latency: {t_test_dur:.2f}s ({ms_per_chunk:.1f} ms/chunk)"
    )

    print("\n=== Step 6: Embedding Corpus Chunks ===")
    cached_vecs: list[list[float]] = []
    if cache_file.is_file():
        try:
            cached_data = np.load(cache_file)
            if (
                "embeddings" in cached_data
                and len(cached_data["embeddings"]) == total_chunks
            ):
                print(
                    f"Loading {total_chunks} cached chunk embeddings "
                    f"from {cache_file}..."
                )
                cached_vecs = cached_data["embeddings"].tolist()
        except Exception as e:
            print(f"Cache load skipped: {e}")

    if not cached_vecs:
        t_embed_start = time.perf_counter()
        chunk_vectors: list[list[float]] = []
        for idx in range(0, total_chunks, batch_size):
            batch = chunk_texts[idx : idx + batch_size]
            vecs = provider.embed_batch(batch)
            chunk_vectors.extend(vecs)
            curr = idx + len(batch)
            if curr % 50 == 0 or curr == total_chunks:
                elapsed = time.perf_counter() - t_embed_start
                rate = curr / elapsed
                rem = (total_chunks - curr) / rate if rate > 0 else 0
                print(
                    f"Progress: {curr}/{total_chunks} chunks "
                    f"({curr / total_chunks * 100:.1f}%) | "
                    f"{rate:.2f} chunks/s | ETA: {rem / 60:.1f}m",
                    end="\r",
                    flush=True,
                )
        print()
        total_embed_time = time.perf_counter() - t_embed_start
        print(
            f"Embedded {total_chunks} chunks in {total_embed_time:.2f}s "
            f"({total_embed_time / 60:.1f}m)."
        )

        try:
            arr = np.array(chunk_vectors, dtype=np.float32)
            np.savez_compressed(cache_file, embeddings=arr)
            print(f"Saved chunk embeddings cache to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    else:
        chunk_vectors = cached_vecs
        total_embed_time = 0.0

    chunk_tensor = torch.tensor(chunk_vectors, dtype=torch.float32)

    print("\n=== Step 7: Computing Cosine Similarities & Retrieval ===")
    t0 = time.perf_counter()
    sim_matrix = torch.mm(query_tensor, chunk_tensor.T)
    print(f"Similarity matrix computed in {(time.perf_counter() - t0)*1000:.2f} ms.")

    retrieved_doc_ids_all: list[list[str]] = []
    gold_doc_ids_all: list[set[str]] = []
    diagnostic_rows: list[dict] = []

    for q_idx, q in enumerate(questions):
        scores = sim_matrix[q_idx].tolist()
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        seen_docs: set[str] = set()
        retrieved_docs: list[str] = []
        top_chunks_meta: list[dict] = []

        for c_idx in sorted_indices:
            doc_id = chunk_doc_ids[c_idx]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                retrieved_docs.append(doc_id)
                if len(top_chunks_meta) < 5:
                    top_chunks_meta.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": chunk_ids[c_idx],
                            "score": round(scores[c_idx], 4),
                            "title": doc_titles.get(doc_id, ""),
                        }
                    )
            if len(retrieved_docs) >= 50:
                break

        gold_set = set(q["gold_doc_ids"])
        retrieved_doc_ids_all.append(retrieved_docs)
        gold_doc_ids_all.append(gold_set)

        first_gold = compute_first_gold_rank(retrieved_docs, gold_set)
        diagnostic_rows.append(
            {
                "qid": q["qid"],
                "qtype": q["qtype"],
                "question": q["question"],
                "gold_doc_ids": list(gold_set),
                "gold_titles": [doc_titles.get(d, d) for d in gold_set],
                "top5_doc_ids": retrieved_docs[:5],
                "top5_meta": top_chunks_meta,
                "first_gold_rank": first_gold,
                "hit@1": is_hit_at_k(retrieved_docs, gold_set, 1),
                "hit@5": is_hit_at_k(retrieved_docs, gold_set, 5),
                "hit@10": is_hit_at_k(retrieved_docs, gold_set, 10),
                "hit@20": is_hit_at_k(retrieved_docs, gold_set, 20),
            }
        )

    print("\n=== Step 8: Calculating Retrieval Metrics ===")
    recalls = compute_recalls(
        retrieved_doc_ids_all, gold_doc_ids_all, k_list=(1, 5, 10, 20)
    )
    overall_mrr = compute_mrr(retrieved_doc_ids_all, gold_doc_ids_all)
    total_failures = sum(1 for row in diagnostic_rows if not row["hit@20"])

    qid_to_idx = {q["qid"]: i for i, q in enumerate(questions)}
    qtype_groups: dict[str, list[dict]] = {}
    for row in diagnostic_rows:
        qtype_groups.setdefault(row["qtype"], []).append(row)

    qtype_breakdown: dict[str, dict] = {}
    for qtype, rows in sorted(qtype_groups.items()):
        full_ret_sub = [
            retrieved_doc_ids_all[qid_to_idx[r["qid"]]] for r in rows
        ]
        gold_sub = [set(r["gold_doc_ids"]) for r in rows]
        sub_rec = compute_recalls(full_ret_sub, gold_sub, k_list=(1, 5, 10, 20))
        sub_mrr = compute_mrr(full_ret_sub, gold_sub)
        qtype_breakdown[qtype] = {
            "count": len(rows),
            "Recall@1": round(sub_rec[1], 4),
            "Recall@5": round(sub_rec[5], 4),
            "Recall@10": round(sub_rec[10], 4),
            "Recall@20": round(sub_rec[20], 4),
            "MRR": round(sub_mrr, 4),
        }

    summary = {
        "model_id": provider.model_name,
        "dimension": provider.dimension,
        "total_questions_evaluated": len(questions),
        "total_chunks_indexed": total_chunks,
        "total_embedding_time_s": round(total_embed_time, 2),
        "Recall@1": round(recalls[1], 4),
        "Recall@5": round(recalls[5], 4),
        "Recall@10": round(recalls[10], 4),
        "Recall@20": round(recalls[20], 4),
        "MRR": round(overall_mrr, 4),
        "retrieval_failures_at_20": total_failures,
        "qtype_breakdown": qtype_breakdown,
        "diagnostics": diagnostic_rows,
    }

    results_file = cache_dir / f"{slug}_retrieval_benchmark_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Overall Retrieval Benchmark Results ---")
    print(f"Model ID: {provider.model_name}")
    print(f"Questions Evaluated: {len(questions)}")
    print(f"Chunks Indexed: {total_chunks}")
    print(f"Recall@1:  {recalls[1]*100:.1f}%")
    print(f"Recall@5:  {recalls[5]*100:.1f}%")
    print(f"Recall@10: {recalls[10]*100:.1f}%")
    print(f"Recall@20: {recalls[20]*100:.1f}%")
    print(f"MRR:       {overall_mrr:.4f}")
    print(f"Failures (no gold doc in top 20): {total_failures}/{len(questions)}")

    print("\n--- Breakdown by Question Type ---")
    for qtype, metrics in qtype_breakdown.items():
        print(
            f"  {qtype:<14} (n={metrics['count']:02d}): "
            f"R@1={metrics['Recall@1']*100:5.1f}% | "
            f"R@5={metrics['Recall@5']*100:5.1f}% | "
            f"R@10={metrics['Recall@10']*100:5.1f}% | "
            f"R@20={metrics['Recall@20']*100:5.1f}% | "
            f"MRR={metrics['MRR']:.4f}"
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run offline retrieval benchmark for an EmbeddingProvider."
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of corpus chunks for fast test run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )
    args = parser.parse_args()
    print(
        "No active embedding provider configured. Instantiate an EmbeddingProvider "
        "and pass it to run_benchmark(provider)."
    )
