"""Deterministic Adaptive GraphRAG retrieval coordinator."""

import re
import time
from typing import Any

from tgh.embeddings.nomic import NomicEmbeddingProvider
from tgh.retrieval.graph import GraphExpansionConfig, TigerGraphTraverser
from tgh.retrieval.models import (
    EvidenceChunk,
    GraphRAGRetrievalResult,
    SeedChunk,
)
from tgh.retrieval.vector import TigerGraphVectorRetriever

# Deterministic patterns indicating relational, multi-hop, or venue-grounded intent
RELATIONAL_PATTERNS = [
    r"\bheld at\b",
    r"\bevent held\b",
    r"\bvenue\b",
    r"\brepresented\b",
    r"\bwhich country did\b",
    r"\bwhich nation did\b",
    r"\bwho (?:won|competed|participated) in the event held\b",
    r"\bwho (?:won|competed|participated) .* held at\b",
    r"\bwho won the gold medal in the event\b",
]


class GraphRAGRetriever:
    """Adaptive GraphRAG engine executing vector search + selective graph expansion."""

    def __init__(
        self,
        conn: Any,
        embedding_provider: NomicEmbeddingProvider | None = None,
        config: GraphExpansionConfig | None = None,
    ) -> None:
        """Initialize the GraphRAG retriever.

        Args:
            conn: Active pyTigerGraph connection with searchChunksByVector installed.
            embedding_provider: Nomic provider for query embedding.
            config: Graph expansion configuration.
        """
        self.conn = conn
        self.config = config or GraphExpansionConfig()
        self.vector_retriever = TigerGraphVectorRetriever(
            conn=conn,
            provider=embedding_provider,
            top_k=self.config.seed_top_k,
        )
        self.traverser = TigerGraphTraverser(conn=conn, config=self.config)
        self._relational_regexes = [
            re.compile(p, re.IGNORECASE) for p in RELATIONAL_PATTERNS
        ]

    def should_expand_graph(
        self,
        question: str,
        seeds: list[SeedChunk],
    ) -> tuple[bool, str]:
        """Determine if graph expansion is warranted based on deterministic signals.

        Returns:
            Tuple of (should_expand: bool, reason: str).
        """
        if not self.config.adaptive_gating:
            return True, "adaptive_gating_disabled"

        # Signal 1: Strong relational or multi-hop language
        for pattern in self._relational_regexes:
            if pattern.search(question):
                return True, f"relational_language({pattern.pattern})"

        if not seeds:
            return True, "no_vector_seeds"

        # Signal 2: Vector retrieval uncertainty or narrow margin
        top_sim = seeds[0].similarity
        margin = (top_sim - seeds[1].similarity) if len(seeds) > 1 else 1.0

        if top_sim < self.config.min_vector_confidence:
            min_c = self.config.min_vector_confidence
            return (
                True,
                f"low_vector_confidence({top_sim:.4f} < {min_c:.2f})",
            )

        if margin < self.config.min_vector_margin and top_sim < 0.80:
            return (
                True,
                f"low_vector_margin({margin:.4f} < {self.config.min_vector_margin})",
            )

        # Confident non-relational query -> preserve pure vector ranking
        return False, f"confident_vector(top_sim={top_sim:.4f}, margin={margin:.4f})"

    def retrieve(
        self,
        question: str,
        top_k_seeds: int | None = None,
    ) -> GraphRAGRetrievalResult:
        """Execute adaptive GraphRAG retrieval pipeline for a natural language question.

        Args:
            question: Natural language question.
            top_k_seeds: Optional override for the number of vector seeds.

        Returns:
            Structured GraphRAGRetrievalResult containing ranked documents, evidence
            chunks, decision reasoning, and graph provenance.
        """
        t0 = time.perf_counter()
        k_seeds = top_k_seeds or self.config.seed_top_k

        # 1. Always perform vector search first
        seeds: list[SeedChunk] = self.vector_retriever.retrieve_seeds(
            question=question,
            top_k=k_seeds,
        )

        # 2. Determine whether graph expansion is warranted
        should_expand, decision_reason = self.should_expand_graph(question, seeds)

        if not should_expand:
            # Bypass graph expansion: preserve pure vector ranking directly
            evidence_chunks = [
                EvidenceChunk(
                    chunk_id=s.chunk_id,
                    doc_id=s.doc_id,
                    score=s.similarity,
                    source="seed",
                    hop_distance=0,
                    provenance=[
                        f"VectorSearch(similarity={s.similarity:.4f}, rank={s.rank})"
                    ],
                    seed_chunk_id=s.chunk_id,
                )
                for s in seeds
            ]
            ranked_doc_ids = self._rank_documents(evidence_chunks)
            total_latency_ms = (time.perf_counter() - t0) * 1000.0

            stats = {
                "num_seeds": len(seeds),
                "num_entities": 0,
                "num_domain_vertices": 0,
                "num_graph_paths": 0,
                "num_graph_derived_chunks": 0,
                "num_total_evidence_chunks": len(evidence_chunks),
                "num_ranked_docs": len(ranked_doc_ids),
            }

            return GraphRAGRetrievalResult(
                question=question,
                seed_chunks=seeds,
                discovered_entities=[],
                discovered_domain_vertices=[],
                graph_paths=[],
                evidence_chunks=evidence_chunks,
                ranked_doc_ids=ranked_doc_ids,
                retrieval_strategy="GraphRAG_Adaptive_v1",
                latency_ms=total_latency_ms,
                expansion_stats=stats,
                decision_reason=decision_reason,
                is_graph_expanded=False,
            )

        # 3. Perform bounded graph expansion for graph-worthy queries
        (
            discovered_entities,
            domain_vertices,
            graph_paths,
            graph_derived_chunks,
        ) = self.traverser.expand_seeds(seeds)

        # 4. Fuse and rank evidence chunks
        evidence_chunks = self._fuse_evidence(seeds, graph_derived_chunks)

        # 5. Deduplicate to document level
        ranked_doc_ids = self._rank_documents(evidence_chunks)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        stats = {
            "num_seeds": len(seeds),
            "num_entities": len(discovered_entities),
            "num_domain_vertices": len(domain_vertices),
            "num_graph_paths": len(graph_paths),
            "num_graph_derived_chunks": len(graph_derived_chunks),
            "num_total_evidence_chunks": len(evidence_chunks),
            "num_ranked_docs": len(ranked_doc_ids),
        }

        return GraphRAGRetrievalResult(
            question=question,
            seed_chunks=seeds,
            discovered_entities=discovered_entities,
            discovered_domain_vertices=domain_vertices,
            graph_paths=graph_paths,
            evidence_chunks=evidence_chunks,
            ranked_doc_ids=ranked_doc_ids,
            retrieval_strategy="GraphRAG_Adaptive_v1",
            latency_ms=total_latency_ms,
            expansion_stats=stats,
            decision_reason=decision_reason,
            is_graph_expanded=True,
        )

    def _fuse_evidence(
        self,
        seeds: list[SeedChunk],
        graph_chunks: list[EvidenceChunk],
    ) -> list[EvidenceChunk]:
        """Combine vector seeds and graph-derived chunks with deterministic scoring."""
        chunk_map: dict[str, EvidenceChunk] = {}

        # 1. Add vector seeds as base evidence
        for s in seeds:
            chunk_map[s.chunk_id] = EvidenceChunk(
                chunk_id=s.chunk_id,
                doc_id=s.doc_id,
                score=s.similarity,
                source="seed",
                hop_distance=0,
                provenance=[
                    f"VectorSearch(similarity={s.similarity:.4f}, rank={s.rank})"
                ],
                seed_chunk_id=s.chunk_id,
            )

        # 2. Fuse graph-derived chunks
        for gc in graph_chunks:
            cid = gc.chunk_id
            if cid in chunk_map:
                # Chunk was both a vector seed and reached via graph traversal
                existing = chunk_map[cid]
                # Reinforce seed score with conservative graph boost
                boost = self.config.reinforcement_weight * gc.score
                existing.score += boost
                existing.source = "seed+graph"
                for p in gc.provenance:
                    if p not in existing.provenance:
                        existing.provenance.append(p)
            else:
                chunk_map[cid] = gc

        # 3. Sort deterministically by descending score, tie-break by chunk_id
        return sorted(
            chunk_map.values(),
            key=lambda c: (c.score, c.chunk_id),
            reverse=True,
        )

    def _rank_documents(self, evidence_chunks: list[EvidenceChunk]) -> list[str]:
        """Deduplicate evidence chunks to document IDs in order of highest score."""
        doc_scores: dict[str, float] = {}
        for ec in evidence_chunks:
            if not ec.doc_id:
                continue
            if ec.doc_id not in doc_scores or ec.score > doc_scores[ec.doc_id]:
                doc_scores[ec.doc_id] = ec.score

        # Sort documents by descending best score, tie-break deterministically by doc_id
        sorted_docs = sorted(
            doc_scores.keys(),
            key=lambda d: (doc_scores[d], d),
            reverse=True,
        )
        return sorted_docs
