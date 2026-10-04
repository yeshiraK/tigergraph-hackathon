"""Vector retrieval interface using TigerGraph HNSW index."""

from typing import Any

from tgh.embeddings.nomic import NomicEmbeddingProvider
from tgh.retrieval.models import SeedChunk


class TigerGraphVectorRetriever:
    """Retrieves seed chunks using TigerGraph's installed HNSW vector query."""

    def __init__(
        self,
        conn: Any,
        provider: NomicEmbeddingProvider | None = None,
        top_k: int = 10,
    ) -> None:
        """Initialize the vector retriever.

        Args:
            conn: Active pyTigerGraph connection with searchChunksByVector installed.
            provider: Nomic embedding provider instance (created if None).
            top_k: Default number of top candidate chunks to retrieve.
        """
        self.conn = conn
        self.provider = provider or NomicEmbeddingProvider(dimension=768)
        self.top_k = top_k

    def retrieve_seeds(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[SeedChunk]:
        """Embed a question and query TigerGraph HNSW index for top chunks.

        Args:
            question: Natural language question.
            top_k: Optional override for the number of candidates.

        Returns:
            List of SeedChunk objects sorted by distance (highest similarity first).
        """
        k = top_k if top_k is not None else self.top_k

        # 1. Format and embed query
        formatted_q = self.provider.format_query(question)
        q_vec = self.provider.embed_text(formatted_q)

        # 2. Query TigerGraph HNSW
        res = self.conn.runInstalledQuery(
            "searchChunksByVector",
            params={"qvec": q_vec, "top_k": k},
        )
        if not res:
            return []

        tg_res = res[0]
        distances = tg_res.get("@@distances", {})
        chunk_to_doc = tg_res.get("@@chunk_to_doc", {})
        candidates = tg_res.get("candidates", [])

        # 3. Sort candidate chunks by ascending distance
        sorted_candidates = sorted(
            candidates,
            key=lambda c: distances.get(c.get("v_id", ""), 1.0),
        )

        # 4. Construct typed SeedChunk models
        seeds: list[SeedChunk] = []
        for rank, c in enumerate(sorted_candidates, start=1):
            cid = c.get("v_id", "")
            if not cid:
                continue

            # Fallback document_id from chunk_to_doc or split by '#'
            doc_id = chunk_to_doc.get(cid, "")
            if not doc_id and "#" in cid:
                doc_id = cid.split("#", 1)[0]

            dist = float(distances.get(cid, 1.0))
            # Cosine similarity for unit vectors in cosine distance space
            similarity = max(0.0, 1.0 - dist)

            attrs = c.get("attributes", {})
            text = attrs.get("candidates.text") or attrs.get("text") or ""

            seeds.append(
                SeedChunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    distance=dist,
                    similarity=similarity,
                    rank=rank,
                    text=text,
                )
            )

        return seeds
