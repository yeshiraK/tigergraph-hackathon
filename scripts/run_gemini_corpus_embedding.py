"""Production corpus embedding pipeline using Gemini Embedding 2.

Generates 768-dimensional L2-normalized embeddings for the 9,348 production
chunks extracted from data/raw/corpus.jsonl via Google Generative Language API.
Supports resumable atomic checkpointing under experiments/runs/.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import requests

from tgh.embeddings.gemini import parse_retry_after
from tgh.ingestion.chunker import SemanticChunker

MODEL_NAME = "gemini-embedding-2"
API_MODEL = f"models/{MODEL_NAME}"
TARGET_DIM = 768
DEFAULT_BATCH_SIZE = 25
# 2.0s pause between successful batches to respect Free-tier RPM/TPM
INTER_BATCH_PAUSE = 2.0


def load_api_key(env_path: Path) -> str:
    """Read GEMINI_API_KEY from .env without printing it."""
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'").strip()
    raise ValueError("GEMINI_API_KEY not found in .env")


def extract_production_corpus_chunks(
    corpus_path: Path,
) -> tuple[list[dict], dict[str, str]]:
    """Extract production chunks and document titles from raw corpus."""
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
) -> tuple[list[str], list[str], list[list[float]], dict]:
    """Load existing checkpoint if available."""
    if checkpoint_npz.is_file() and checkpoint_meta.is_file():
        try:
            with checkpoint_meta.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            data = np.load(checkpoint_npz)
            vectors = data["embeddings"].tolist()
            chunk_ids = meta.get("chunk_ids", [])
            doc_ids = meta.get("doc_ids", [])
            if len(chunk_ids) == len(vectors) and len(doc_ids) == len(vectors):
                return chunk_ids, doc_ids, vectors, meta
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh.")
    return [], [], [], {}


def save_checkpoint(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    chunk_ids: list[str],
    doc_ids: list[str],
    vectors: list[list[float]],
    stats: dict,
) -> None:
    """Atomically save checkpoint and metadata to disk."""
    temp_npz = checkpoint_npz.with_suffix(".tmp.npz")
    temp_meta = checkpoint_meta.with_suffix(".tmp.json")

    arr = np.array(vectors, dtype=np.float32)
    np.savez_compressed(temp_npz, embeddings=arr)

    meta = {
        "model_name": MODEL_NAME,
        "dimension": TARGET_DIM,
        "total_embedded": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "timestamp": time.time(),
        "stats": stats,
    }
    with temp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    shutil.move(temp_npz, checkpoint_npz)
    shutil.move(temp_meta, checkpoint_meta)


def call_batch_embed_with_retry(
    texts: list[str],
    api_key: str,
    stats: dict,
    max_retries: int = 6,
) -> tuple[list[list[float]], int]:
    """Call batchEmbedContents with exponential backoff on 429/5xx."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{API_MODEL}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload = {
        "requests": [
            {
                "model": API_MODEL,
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": TARGET_DIM,
            }
            for t in texts
        ]
    }

    for attempt in range(max_retries):
        stats["http_requests"] += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            stats["retries"] += 1
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Network error after {max_retries} attempts: {e}"
                ) from e
            wait = 3.0 * (2**attempt) + random.uniform(0.5, 1.5)
            print(f"    [Network Error] Backing off {wait:.1f}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            raw_embs = data.get("embeddings", [])
            embs = [item["values"] for item in raw_embs]
            stats["successful_batches"] += 1
            stats["texts_embedded"] += len(embs)
            return embs, resp.status_code

        elif resp.status_code == 429:
            stats["429_count"] += 1
            stats["retries"] += 1
            retry_after = parse_retry_after(resp)
            backoff = (
                retry_after
                if retry_after is not None
                else (15.0 * (attempt + 1) + random.uniform(1.0, 3.0))
            )
            print(f"    [Rate limit 429] Backing off {backoff:.1f}s...")
            time.sleep(backoff)

        elif resp.status_code >= 500:
            stats["5xx_count"] += 1
            stats["retries"] += 1
            backoff = 3.0 * (2**attempt) + random.uniform(0.5, 1.5)
            print(f"    [Server {resp.status_code}] Backing off {backoff:.1f}s...")
            time.sleep(backoff)

        else:
            sanitized = resp.text.replace(api_key, "[REDACTED]")
            raise RuntimeError(
                f"Gemini API error (HTTP {resp.status_code}): {sanitized}"
            )

    raise RuntimeError(f"Failed to embed batch after {max_retries} retries")


def run_production_job(
    batch_size: int = DEFAULT_BATCH_SIZE,
    pause_s: float = INTER_BATCH_PAUSE,
    limit: int | None = None,
) -> None:
    """Execute production corpus embedding job."""
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    corpus_file = repo_root / "data" / "raw" / "corpus.jsonl"
    runs_dir = repo_root / "experiments" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_npz = runs_dir / "gemini_corpus_embeddings.npz"
    checkpoint_meta = runs_dir / "gemini_corpus_embeddings_meta.json"

    api_key = load_api_key(env_file)

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
    done_chunk_ids, done_doc_ids, done_vectors, saved_meta = load_checkpoint(
        checkpoint_npz, checkpoint_meta
    )
    done_set = set(done_chunk_ids)
    print(f"Resuming: {len(done_set)} / {total_chunks} chunks already checkpointed.")

    if len(done_set) == total_chunks:
        print("All chunks already embedded! Checkpoint is complete.")
        return

    # Initialize stats
    stats = saved_meta.get("stats", {
        "http_requests": 0,
        "texts_embedded": len(done_set),
        "tokens_processed": 0,
        "429_count": 0,
        "5xx_count": 0,
        "retries": 0,
        "successful_batches": 0,
    })

    remaining = [c for c in chunks if c["chunk_id"] not in done_set]
    print(f"Remaining chunks to embed: {len(remaining)}")

    embedded_chunk_ids = list(done_chunk_ids)
    embedded_doc_ids = list(done_doc_ids)
    embedded_vectors = list(done_vectors)

    t_start = time.perf_counter()
    newly_embedded = 0

    print(
        f"\n=== Step 3: Starting Production Batches "
        f"(BS={batch_size}, pause={pause_s}s) ==="
    )
    idx = 0
    while idx < len(remaining):
        batch_chunks = remaining[idx : idx + batch_size]
        formatted_batch = [
            f"title: {c['title']} | text: {c['text'].strip()}"
            for c in batch_chunks
        ]
        batch_tokens = sum(math.ceil(len(t) / 4.0) for t in formatted_batch)

        embs, _ = call_batch_embed_with_retry(
            formatted_batch, api_key, stats
        )

        for c, vec in zip(batch_chunks, embs, strict=True):
            embedded_chunk_ids.append(c["chunk_id"])
            embedded_doc_ids.append(c["document_id"])
            embedded_vectors.append(vec)

        newly_embedded += len(batch_chunks)
        stats["tokens_processed"] += batch_tokens
        idx += len(batch_chunks)

        # Save checkpoint after every batch for maximal safety
        save_checkpoint(
            checkpoint_npz,
            checkpoint_meta,
            embedded_chunk_ids,
            embedded_doc_ids,
            embedded_vectors,
            stats,
        )

        curr_total = len(embedded_chunk_ids)
        elapsed = time.perf_counter() - t_start
        rate = newly_embedded / elapsed if elapsed > 0 else 0
        rem_chunks = total_chunks - curr_total
        eta_m = (rem_chunks / rate) / 60 if rate > 0 else 0
        pct = curr_total / total_chunks * 100

        print(
            f"[{curr_total}/{total_chunks}] ({pct:.1f}%) | "
            f"Rate: {rate:.2f} chunks/s | "
            f"Reqs: {stats['http_requests']} | "
            f"429s: {stats['429_count']} | "
            f"Saved: {checkpoint_npz.name} | "
            f"ETA: {eta_m:.1f}m",
            flush=True,
        )

        if pause_s > 0 and idx < len(remaining):
            time.sleep(pause_s)

    print("\n=== Production Corpus Embedding Complete ===")
    print(f"Total Chunks: {len(embedded_chunk_ids)}")
    print(f"Checkpoint Saved: {checkpoint_npz}")
    print(f"Metadata Saved: {checkpoint_meta}")
    print(f"Total HTTP Requests: {stats['http_requests']}")
    print(f"Total 429 Errors: {stats['429_count']}")
    print(f"Total Retries: {stats['retries']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run production Gemini corpus embedding pipeline."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks per batchEmbedContents request",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=INTER_BATCH_PAUSE,
        help="Seconds to pause between successful batches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on total chunks (for dry validation)",
    )
    args = parser.parse_args()

    run_production_job(
        batch_size=args.batch_size,
        pause_s=args.pause,
        limit=args.limit,
    )
