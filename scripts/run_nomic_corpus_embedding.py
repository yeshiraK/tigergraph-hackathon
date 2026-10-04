"""Production corpus embedding pipeline using nomic-ai/nomic-embed-text-v1.5.

Generates 768-dimensional L2-normalized embeddings for the 9,348 production
chunks extracted from data/raw/corpus.jsonl on local CPU.
Supports resumable atomic checkpointing under experiments/runs/.
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tgh.embeddings.nomic import DEFAULT_MODEL_ID, DEFAULT_TARGET_DIM, mean_pooling
from tgh.ingestion.chunker import SemanticChunker

DOC_PREFIX = "search_document: {document}"


def get_peak_rss_mb() -> float:
    """Return peak resident set size in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def extract_production_corpus_chunks(
    corpus_path: Path,
) -> tuple[list[dict], dict[str, str]]:
    """Extract production chunks and doc titles deterministically from raw corpus."""
    chunker = SemanticChunker(target_tokens=768)
    chunk_records: list[dict] = []
    doc_titles: dict[str, str] = {}

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            doc_id = rec["doc_id"]
            title = rec.get("title", "") or ""
            doc_titles[doc_id] = title

            chunks = chunker.chunk_document(doc_id, rec["text"])
            for c in chunks:
                chunk_records.append(
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "text": c.text,
                        "title": title,
                    }
                )

    return chunk_records, doc_titles


def load_checkpoint(
    checkpoint_npz: Path, checkpoint_meta: Path
) -> tuple[list[str], list[str], list[list[float]]]:
    """Load existing checkpoint if available and valid."""
    if checkpoint_npz.is_file() and checkpoint_meta.is_file():
        try:
            with checkpoint_meta.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            data = np.load(checkpoint_npz)
            arr = data["embeddings"]
            chunk_ids = meta.get("chunk_ids", [])
            doc_ids = meta.get("doc_ids", [])

            if (
                len(chunk_ids) == len(arr)
                and len(doc_ids) == len(arr)
                and arr.ndim == 2
                and arr.shape[1] == DEFAULT_TARGET_DIM
            ):
                print(
                    f"Loaded existing checkpoint: {len(chunk_ids)} chunks "
                    f"from {checkpoint_npz.name}"
                )
                return chunk_ids, doc_ids, arr.tolist()
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh.")
    return [], [], []


def validate_checkpoint_data(
    chunk_ids: list[str],
    doc_ids: list[str],
    arr: np.ndarray,
    expected_dim: int = 768,
) -> None:
    """Validate data integrity before saving checkpoint."""
    n = len(chunk_ids)
    if len(doc_ids) != n:
        raise ValueError(f"doc_ids length ({len(doc_ids)}) != chunk_ids length ({n})")
    if arr.shape != (n, expected_dim):
        raise ValueError(f"Array shape {arr.shape} != expected ({n}, {expected_dim})")
    if arr.dtype != np.float32:
        raise ValueError(f"Array dtype {arr.dtype} != expected float32")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values detected in embedding array")
    if len(set(chunk_ids)) != n:
        raise ValueError(f"Duplicate chunk IDs found in {n} chunks")

    # Verify L2 normalization
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"Embeddings are not unit L2 normalized: min={norms.min():.4f}, "
            f"max={norms.max():.4f}"
        )


def save_checkpoint(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    chunk_ids: list[str],
    doc_ids: list[str],
    vectors: list[list[float]],
    model_name: str,
    dimension: int,
    elapsed_seconds: float,
) -> None:
    """Atomically and safely save checkpoint to disk."""
    temp_npz = checkpoint_npz.with_suffix(".tmp.npz")
    temp_meta = checkpoint_meta.with_suffix(".tmp.json")

    arr = np.array(vectors, dtype=np.float32)
    validate_checkpoint_data(chunk_ids, doc_ids, arr, expected_dim=dimension)

    np.savez_compressed(temp_npz, embeddings=arr)

    meta = {
        "model_name": model_name,
        "dimension": dimension,
        "normalization": "unit_l2",
        "completed_chunks": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "elapsed_seconds": elapsed_seconds,
        "timestamp": time.time(),
    }
    with temp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    os.replace(temp_npz, checkpoint_npz)
    os.replace(temp_meta, checkpoint_meta)


def run_embedding_job(
    checkpoint_interval: int = 100,
    limit: int | None = None,
) -> None:
    """Run production Nomic corpus embedding job with resumable checkpointing."""
    repo_root = Path(__file__).resolve().parent.parent
    corpus_file = repo_root / "data" / "raw" / "corpus.jsonl"
    runs_dir = repo_root / "experiments" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_npz = runs_dir / "nomic_corpus_embeddings.npz"
    checkpoint_meta = runs_dir / "nomic_corpus_embeddings_meta.json"

    print("=" * 70)
    print("NOMIC V1.5 PRODUCTION CORPUS EMBEDDING PIPELINE")
    print("=" * 70)

    # Step 1: Chunk corpus deterministically
    print("\n=== Step 1: Loading & Chunking Production Corpus ===")
    t0 = time.perf_counter()
    chunks, _ = extract_production_corpus_chunks(corpus_file)
    if limit is not None:
        chunks = chunks[:limit]
    total_chunks = len(chunks)
    print(
        f"Corpus processed in {time.perf_counter() - t0:.2f}s: "
        f"{total_chunks} total production chunks."
    )

    # Step 2: Check for existing checkpoint
    print("\n=== Step 2: Checking for Existing Checkpoint ===")
    done_chunk_ids, done_doc_ids, done_vectors = load_checkpoint(
        checkpoint_npz, checkpoint_meta
    )
    done_set = set(done_chunk_ids)
    print(f"Resuming: {len(done_set)} / {total_chunks} chunks already embedded.")

    if len(done_set) == total_chunks:
        print("\nAll 9,348 chunks already embedded! Checkpoint is complete.")
        perform_final_validation(checkpoint_npz, checkpoint_meta, chunks)
        return

    # Step 3: Load Nomic model
    print("\n=== Step 3: Loading Nomic Embed Text v1.5 Model ===")
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    load_time = time.perf_counter() - t_load0
    print(
        f"Model loaded in {load_time:.2f}s | Device: CPU | "
        f"Peak RSS: {get_peak_rss_mb():.1f} MB"
    )

    # Step 4: Embedding loop
    print("\n=== Step 4: Starting Embedding Process ===")
    embedded_chunk_ids = list(done_chunk_ids)
    embedded_doc_ids = list(done_doc_ids)
    embedded_vectors = list(done_vectors)

    total_tokens_processed = 0
    t_run_start = time.perf_counter()
    newly_embedded = 0

    for _idx, c in enumerate(chunks):
        cid = c["chunk_id"]
        if cid in done_set:
            continue

        doc_text = "search_document: " + c["text"]
        encoded = tokenizer(
            [doc_text],
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        input_tokens = int(encoded["attention_mask"].sum().item())
        total_tokens_processed += input_tokens

        with torch.inference_mode():
            out = model(**encoded)
            pooled = mean_pooling(out[0], encoded["attention_mask"])
            ln = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
            normed = F.normalize(ln, p=2, dim=1)

        vec = normed[0].cpu().tolist()
        embedded_chunk_ids.append(cid)
        embedded_doc_ids.append(c["document_id"])
        embedded_vectors.append(vec)
        newly_embedded += 1

        # Checkpoint condition
        if (
            newly_embedded % checkpoint_interval == 0
            or len(embedded_chunk_ids) == total_chunks
        ):
            elapsed = time.perf_counter() - t_run_start
            save_checkpoint(
                checkpoint_npz=checkpoint_npz,
                checkpoint_meta=checkpoint_meta,
                chunk_ids=embedded_chunk_ids,
                doc_ids=embedded_doc_ids,
                vectors=embedded_vectors,
                model_name=DEFAULT_MODEL_ID,
                dimension=DEFAULT_TARGET_DIM,
                elapsed_seconds=elapsed,
            )

            current_count = len(embedded_chunk_ids)
            pct = (current_count / total_chunks) * 100.0
            sec_per_chk = elapsed / newly_embedded
            remaining_chunks = total_chunks - current_count
            eta_sec = remaining_chunks * sec_per_chk
            eta_min = eta_sec / 60.0
            eta_hr = eta_sec / 3600.0
            tok_per_sec = total_tokens_processed / elapsed

            print(
                f"[{current_count:4d}/{total_chunks:4d} | {pct:5.1f}%] "
                f"Elapsed: {elapsed/60.0:5.1f}m | "
                f"ETA: {eta_hr:4.2f}h ({eta_min:5.1f}m) | "
                f"Speed: {1.0/sec_per_chk:4.2f} chk/s ({sec_per_chk:5.3f} s/chk) | "
                f"Tokens/s: {tok_per_sec:5.1f} | "
                f"RAM: {get_peak_rss_mb():.1f} MB"
            )

    total_run_time = time.perf_counter() - t_run_start
    print(
        f"\nEmbedding loop finished in {total_run_time:.2f}s "
        f"({total_run_time/3600:.2f}h)."
    )

    # Step 5: Final validation
    perform_final_validation(checkpoint_npz, checkpoint_meta, chunks)


def perform_final_validation(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    expected_chunks: list[dict],
) -> None:
    """Run comprehensive validation on final embeddings checkpoint."""
    print("\n=== Step 5: Performing Final Integrity Validation ===")
    assert checkpoint_npz.is_file(), f"Missing checkpoint: {checkpoint_npz}"
    assert checkpoint_meta.is_file(), f"Missing meta: {checkpoint_meta}"

    with checkpoint_meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    data = np.load(checkpoint_npz)
    arr = data["embeddings"]

    expected_count = len(expected_chunks)
    actual_count = len(arr)

    print(f"Total Chunks: {actual_count} / {expected_count}")
    print(f"Array Shape: {arr.shape}")
    print(f"Array Dtype: {arr.dtype}")
    print(f"Model Name: {meta.get('model_name')}")
    print(f"Dimension: {meta.get('dimension')}")

    assert actual_count == expected_count, (
        f"Chunk count mismatch: {actual_count} != {expected_count}"
    )
    assert arr.shape == (expected_count, DEFAULT_TARGET_DIM), (
        f"Shape mismatch: {arr.shape}"
    )
    assert arr.dtype == np.float32, f"Dtype mismatch: {arr.dtype}"
    assert np.isfinite(arr).all(), "Non-finite values detected in final embeddings!"

    # Norm validation
    norms = np.linalg.norm(arr, axis=1)
    min_norm, max_norm = float(norms.min()), float(norms.max())
    print(f"L2 Norms: min={min_norm:.6f}, max={max_norm:.6f}")
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit L2 normalized!"

    # Chunk ID alignment validation
    saved_ids = meta.get("chunk_ids", [])
    expected_ids = [c["chunk_id"] for c in expected_chunks]
    assert saved_ids == expected_ids, (
        "Chunk ID ordering mismatch with production chunker!"
    )

    print(
        "\n>>> ALL INTEGRITY CHECKS PASSED: "
        "9,348 CHUNKS FULLY VALIDATED AND PRESERVED <<<"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Nomic production corpus embedding"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Chunks between checkpoints",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks for testing",
    )
    args = parser.parse_args()

    run_embedding_job(
        checkpoint_interval=args.checkpoint_interval,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
