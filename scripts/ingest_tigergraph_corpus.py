"""Bulk ingestion of Documents, Chunks, and HAS_CHUNK edges into TigerGraph.

Loads:
- 2,951 Document vertices
- 9,348 Chunk vertices (with 768D Nomic embeddings)
- 9,348 HAS_CHUNK edges
"""

import json
import time
from pathlib import Path

import numpy as np
import pyTigerGraph as tg

from tgh.ingestion.chunker import SemanticChunker


def load_env() -> dict[str, str]:
    env_file = Path("/Users/yeshi/Desktop/tgh/.env")
    res = {}
    if env_file.is_file():
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    res[k.strip()] = v.strip()
    return res


def get_tigergraph_connection() -> tg.TigerGraphConnection:
    env = load_env()
    conn = tg.TigerGraphConnection(
        host=env["TIGERGRAPH_HOST"],
        graphname=env["TIGERGRAPH_GRAPH_NAME"],
        gsqlSecret=env["TIGERGRAPH_SECRET"],
    )
    return conn


def main():
    repo_root = Path("/Users/yeshi/Desktop/tgh")
    corpus_file = repo_root / "data" / "raw" / "corpus.jsonl"
    npz_file = repo_root / "experiments" / "runs" / "nomic_corpus_embeddings.npz"
    meta_file = repo_root / "experiments" / "runs" / "nomic_corpus_embeddings_meta.json"

    print("=" * 70)
    print("PHASE 3: TIGERGRAPH INGESTION PIPELINE")
    print("=" * 70)

    # 1. Connect to TigerGraph
    conn = get_tigergraph_connection()
    print(f"Connected to TigerGraph graph: {conn.graphname}")

    # 2. Load corpus documents
    print("\n--- Step 1: Loading Corpus Documents ---")
    doc_records = []
    with corpus_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec["doc_id"]
            title = rec.get("title", "") or ""
            source = rec.get("source", "") or ""
            url = rec.get("url", "") or ""
            doc_records.append((doc_id, {
                "title": title,
                "source": source,
                "url": url,
            }))

    print(f"Prepared {len(doc_records)} Document vertices.")
    assert len(doc_records) == 2951, f"Expected 2951 documents, got {len(doc_records)}"

    # 3. Load embeddings and chunk metadata
    print("\n--- Step 2: Loading Chunks and Nomic 768D Embeddings ---")
    npz_data = np.load(npz_file)
    embeddings = npz_data["embeddings"]  # (9348, 768) float32
    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_chunk_ids = meta["chunk_ids"]

    # Reconstruct chunks for attributes (text, chunk_index, token_count)
    chunker = SemanticChunker(target_tokens=768)
    chunk_vertices = []
    has_chunk_edges = []

    with corpus_file.open("r", encoding="utf-8") as f:
        chunk_idx_counter = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks = chunker.chunk_document(rec["doc_id"], rec["text"])
            for c in chunks:
                expected_cid = meta_chunk_ids[chunk_idx_counter]
                assert c.chunk_id == expected_cid, (
                    f"Chunk ID mismatch at index {chunk_idx_counter}"
                )
                vec = embeddings[chunk_idx_counter].tolist()

                chunk_vertices.append((c.chunk_id, {
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                    "embedding": vec,
                }))

                has_chunk_edges.append((c.document_id, c.chunk_id, {}))
                chunk_idx_counter += 1

    print(f"Prepared {len(chunk_vertices)} Chunk vertices with 768D embeddings.")
    print(f"Prepared {len(has_chunk_edges)} HAS_CHUNK edges.")
    assert len(chunk_vertices) == 9348, (
        f"Expected 9348 chunks, got {len(chunk_vertices)}"
    )
    assert len(has_chunk_edges) == 9348, (
        f"Expected 9348 edges, got {len(has_chunk_edges)}"
    )

    # 4. Ingest Document vertices in batches
    print("\n--- Step 3: Ingesting Document Vertices ---")
    doc_batch_size = 500
    total_docs_upserted = 0
    t_docs0 = time.perf_counter()
    for i in range(0, len(doc_records), doc_batch_size):
        batch = doc_records[i : i + doc_batch_size]
        res = conn.upsertVertices("Document", batch)
        total_docs_upserted += res
        print(
            f"  Ingested Documents {i + len(batch)} / {len(doc_records)} "
            f"(Accepted: {res})"
        )
    print(
        f"Completed Document ingestion in {time.perf_counter() - t_docs0:.2f}s."
    )

    # 5. Ingest Chunk vertices in batches
    print("\n--- Step 4: Ingesting Chunk Vertices with Embeddings ---")
    chunk_batch_size = 200
    total_chunks_upserted = 0
    t_chunks0 = time.perf_counter()
    for i in range(0, len(chunk_vertices), chunk_batch_size):
        batch = chunk_vertices[i : i + chunk_batch_size]
        res = conn.upsertVertices("Chunk", batch)
        total_chunks_upserted += res
        elapsed = time.perf_counter() - t_chunks0
        rate = (i + len(batch)) / elapsed
        print(
            f"  Ingested Chunks {i + len(batch)} / {len(chunk_vertices)} "
            f"(Accepted: {res}) | Rate: {rate:.1f} chk/s"
        )
    print(
        f"Completed Chunk ingestion in {time.perf_counter() - t_chunks0:.2f}s."
    )

    # 6. Ingest HAS_CHUNK edges in batches
    print("\n--- Step 5: Ingesting HAS_CHUNK Edges ---")
    edge_batch_size = 500
    total_edges_upserted = 0
    t_edges0 = time.perf_counter()
    for i in range(0, len(has_chunk_edges), edge_batch_size):
        batch = has_chunk_edges[i : i + edge_batch_size]
        res = conn.upsertEdges(
            sourceVertexType="Document",
            edgeType="HAS_CHUNK",
            targetVertexType="Chunk",
            edges=batch,
        )
        total_edges_upserted += res
        print(
            f"  Ingested Edges {i + len(batch)} / {len(has_chunk_edges)} "
            f"(Accepted: {res})"
        )
    print(
        f"Completed HAS_CHUNK edge ingestion in {time.perf_counter() - t_edges0:.2f}s."
    )

    # 7. Verification of live graph counts
    print("\n--- Step 6: Verifying Graph Counts Directly in TigerGraph ---")
    live_docs = conn.getVertexCount("Document")
    live_chunks = conn.getVertexCount("Chunk")
    live_edges = conn.getEdgeCount("HAS_CHUNK")

    print(f"Live Document Count: {live_docs} (Expected: 2951)")
    print(f"Live Chunk Count:    {live_chunks} (Expected: 9348)")
    print(f"Live HAS_CHUNK Count: {live_edges} (Expected: 9348)")

    assert live_docs == 2951, f"Document count mismatch: {live_docs} != 2951"
    assert live_chunks == 9348, f"Chunk count mismatch: {live_chunks} != 9348"
    assert live_edges == 9348, f"HAS_CHUNK edge count mismatch: {live_edges} != 9348"

    # 8. Check Vector Index Status
    print("\n--- Step 7: Checking Vector Index Status ---")
    idx_status = conn.getVectorIndexStatus()
    print("Vector Index Status:", idx_status)

    print("\n>>> INGESTION AND COUNT VERIFICATION COMPLETED SUCCESSFULLY <<<")


if __name__ == "__main__":
    main()
