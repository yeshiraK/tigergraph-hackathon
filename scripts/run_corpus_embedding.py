"""Production corpus embedding pipeline using EmbeddingGemma-300M.

Generates 768-dimensional L2-normalized embeddings for the 9,348 production
chunks extracted from data/raw/corpus.jsonl.
Supports resumable atomic checkpointing under experiments/runs/.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from tgh.embeddings.gemma import GemmaEmbeddingProvider
from tgh.ingestion.chunker import SemanticChunker


def extract_production_corpus_chunks(
    corpus_path: Path,
) -> tuple[list[dict], dict[str, str]]:
    """Extract production chunks and document titles from raw corpus.

    Returns:
        chunk_records: list of dicts with keys (chunk_id, document_id, text, title)
        doc_titles: dict mapping doc_id -> title
    """
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
            title = rec.get("title", "none") or "none"
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
    """Load existing checkpoint if available.

    Returns:
        saved_chunk_ids: list of chunk IDs already embedded
        saved_doc_ids: list of document IDs
        saved_vectors: list of 768D float vectors
    """
    if checkpoint_npz.is_file() and checkpoint_meta.is_file():
        try:
            with checkpoint_meta.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            data = np.load(checkpoint_npz)
            vectors = data["embeddings"].tolist()
            chunk_ids = meta.get("chunk_ids", [])
            doc_ids = meta.get("doc_ids", [])
            if len(chunk_ids) == len(vectors) and len(doc_ids) == len(vectors):
                return chunk_ids, doc_ids, vectors
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh.")
    return [], [], []


def save_checkpoint(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    chunk_ids: list[str],
    doc_ids: list[str],
    vectors: list[list[float]],
    model_name: str,
    dimension: int,
) -> None:
    """Atomically save checkpoint to disk."""
    temp_npz = checkpoint_npz.with_suffix(".tmp.npz")
    temp_meta = checkpoint_meta.with_suffix(".tmp.json")

    arr = np.array(vectors, dtype=np.float32)
    np.savez_compressed(temp_npz, embeddings=arr)

    meta = {
        "model_name": model_name,
        "dimension": dimension,
        "total_embedded": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "timestamp": time.time(),
    }
    with temp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    shutil.move(temp_npz, checkpoint_npz)
    shutil.move(temp_meta, checkpoint_meta)


def run_embedding_job(
    checkpoint_interval: int = 50,
    limit: int | None = None,
) -> None:
    """Run production corpus embedding job with resumable checkpointing."""
    repo_root = Path(__file__).resolve().parent.parent
    corpus_file = repo_root / "data" / "raw" / "corpus.jsonl"
    runs_dir = repo_root / "experiments" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_npz = runs_dir / "gemma_corpus_embeddings.npz"
    checkpoint_meta = runs_dir / "gemma_corpus_embeddings_meta.json"

    print("=== Step 1: Loading & Chunking Production Corpus ===")
    t0 = time.perf_counter()
    chunks, _ = extract_production_corpus_chunks(corpus_file)
    if limit is not None:
        chunks = chunks[:limit]
    total_chunks = len(chunks)
    print(
        f"Corpus processed in {time.perf_counter() - t0:.2f}s: "
        f"{total_chunks} total production chunks."
    )

    print("\n=== Step 2: Checking for Existing Checkpoint ===")
    done_chunk_ids, done_doc_ids, done_vectors = load_checkpoint(
        checkpoint_npz, checkpoint_meta
    )
    done_set = set(done_chunk_ids)
    print(f"Resuming: {len(done_set)} / {total_chunks} chunks already embedded.")

    if len(done_set) == total_chunks:
        print("All chunks already embedded! Checkpoint is complete.")
        return

    print("\n=== Step 3: Initializing GemmaEmbeddingProvider ===")
    provider = GemmaEmbeddingProvider(dimension=768)
    print(f"Provider: {provider.model_name} (dim={provider.dimension})")

    # Filter remaining chunks to process
    remaining = [c for c in chunks if c["chunk_id"] not in done_set]
    print(f"Remaining chunks to embed: {len(remaining)}")

    embedded_chunk_ids = list(done_chunk_ids)
    embedded_doc_ids = list(done_doc_ids)
    embedded_vectors = list(done_vectors)

    t_start = time.perf_counter()
    newly_embedded = 0

    print("\n=== Step 4: Embedding Remaining Chunks ===")
    for c in remaining:
        # Document formatting: title: <title> | text: <passage>
        formatted = provider.format_document(c["text"], title=c["title"])
        vec = provider.embed_text(formatted)

        embedded_chunk_ids.append(c["chunk_id"])
        embedded_doc_ids.append(c["document_id"])
        embedded_vectors.append(vec)
        newly_embedded += 1

        curr_total = len(embedded_chunk_ids)
        if newly_embedded % checkpoint_interval == 0 or curr_total == total_chunks:
            save_checkpoint(
                checkpoint_npz,
                checkpoint_meta,
                embedded_chunk_ids,
                embedded_doc_ids,
                embedded_vectors,
                provider.model_name,
                provider.dimension,
            )
            elapsed = time.perf_counter() - t_start
            rate = newly_embedded / elapsed if elapsed > 0 else 0
            rem_chunks = total_chunks - curr_total
            eta_m = (rem_chunks / rate) / 60 if rate > 0 else 0
            pct = curr_total / total_chunks * 100
            print(
                f"[{curr_total}/{total_chunks}] ({pct:.1f}%) | "
                f"Rate: {rate:.2f} chunks/s | "
                f"Checkpointed to {checkpoint_npz.name} | "
                f"ETA: {eta_m:.1f}m"
            )

    print("\n=== Production Corpus Embedding Finished ===")
    print(f"Successfully embedded all {total_chunks} chunks to {checkpoint_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run production corpus embedding job."
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Number of chunks between checkpoint saves",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total chunks (for testing)",
    )
    args = parser.parse_args()
    run_embedding_job(
        checkpoint_interval=args.checkpoint_interval,
        limit=args.limit,
    )
