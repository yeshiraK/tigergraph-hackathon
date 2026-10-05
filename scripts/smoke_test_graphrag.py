"""Smoke test for GraphRAG retrieval on representative public questions."""

import json
from pathlib import Path

import pyTigerGraph as tg

from tgh.embeddings.nomic import NomicEmbeddingProvider
from tgh.retrieval.graph import GraphExpansionConfig
from tgh.retrieval.graphrag import GraphRAGRetriever


def load_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
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


def main():
    print("=" * 80)
    print("GRAPHRAG RETRIEVAL SMOKE TESTS (REPRESENTATIVE PUBLIC QUESTIONS)")
    print("=" * 80)

    env = load_env()
    conn = tg.TigerGraphConnection(
        host=env["TIGERGRAPH_HOST"],
        graphname=env["TIGERGRAPH_GRAPH_NAME"],
        gsqlSecret=env["TIGERGRAPH_SECRET"],
    )
    conn.getToken()
    print(f"Connected to TigerGraph graph: {conn.graphname}")

    provider = NomicEmbeddingProvider(dimension=768)
    config = GraphExpansionConfig(
        seed_top_k=5,
        max_entities_per_seed=4,
        max_domain_vertices_per_seed=4,
        max_events_per_person=4,
        max_events_per_venue=10,
        max_graph_derived_chunks=15,
        decay_per_hop=0.85,
        reinforcement_weight=0.10,
    )
    retriever = GraphRAGRetriever(
        conn=conn,
        embedding_provider=provider,
        config=config,
    )

    repo_root = Path(__file__).resolve().parent.parent
    public_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"

    selected_qids = ["pub-005", "pub-011", "pub-009", "pub-002"]
    selected_questions = []
    with public_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["qid"] in selected_qids:
                selected_questions.append(item)

    for q in selected_questions:
        qid = q["qid"]
        qtype = q["qtype"]
        question = q["question"]
        gold_docs = set(q["gold_doc_ids"])

        print("\n" + "#" * 80)
        print(f"QUESTION ID: {qid} [{qtype}]")
        print(f"Question:    {question}")
        print(f"Gold Docs:   {sorted(gold_docs)}")
        print("#" * 80)

        result = retriever.retrieve(question)

        print("\n--- 1. Vector Seed Chunks ---")
        for s in result.seed_chunks:
            is_gold = "(GOLD DOC)" if s.doc_id in gold_docs else ""
            print(
                f"  Rank {s.rank}: {s.chunk_id} | doc={s.doc_id} | "
                f"dist={s.distance:.4f} | sim={s.similarity:.4f} {is_gold}"
            )

        print(f"\n--- 2. Discovered Entities ({len(result.discovered_entities)}) ---")
        for ent in result.discovered_entities[:8]:
            print(f"  {ent.entity_id} (from seed {ent.seed_chunk_id})")
        if len(result.discovered_entities) > 8:
            print(f"  ... and {len(result.discovered_entities) - 8} more")

        print(
            f"\n--- 3. Discovered Domain Vertices "
            f"({len(result.discovered_domain_vertices)}) ---"
        )
        for dom in result.discovered_domain_vertices[:8]:
            print(
                f"  {dom.vertex_type}:{dom.vertex_id} "
                f"(resolved from {dom.resolved_from_entity_id})"
            )
        if len(result.discovered_domain_vertices) > 8:
            print(f"  ... and {len(result.discovered_domain_vertices) - 8} more")

        print(f"\n--- 4. Graph Paths Sample ({len(result.graph_paths)} total) ---")
        for p in result.graph_paths[:6]:
            print(f"  [Hop {p.hop}] {p.to_str()}")
        if len(result.graph_paths) > 6:
            print(f"  ... and {len(result.graph_paths) - 6} more")

        print(f"\n--- 5. Final Evidence Chunks ({len(result.evidence_chunks)}) ---")
        for ec in result.evidence_chunks[:8]:
            is_gold = "(GOLD DOC)" if ec.doc_id in gold_docs else ""
            prov_preview = ec.provenance[0] if ec.provenance else "None"
            print(
                f"  Chunk {ec.chunk_id} | doc={ec.doc_id} | score={ec.score:.4f} | "
                f"source={ec.source} {is_gold}"
            )
            print(f"    provenance: {prov_preview}")
        if len(result.evidence_chunks) > 8:
            print(f"  ... and {len(result.evidence_chunks) - 8} more")

        print("\n--- 6. Ranked Documents vs Gold ---")
        gold_hit_ranks = []
        for rank, d in enumerate(result.ranked_doc_ids, start=1):
            if d in gold_docs:
                gold_hit_ranks.append(rank)
                print(f"  Rank {rank:2d}: {d} <--- GOLD HIT!")
            elif rank <= 5:
                print(f"  Rank {rank:2d}: {d}")

        print(f"\nSummary for {qid}:")
        print(f"  Gold Hit Ranks: {gold_hit_ranks if gold_hit_ranks else 'NONE'}")
        n_tot = result.expansion_stats["num_total_evidence_chunks"]
        print(f"  Total Evidence Chunks: {n_tot}")
        print(f"  Total Entities Discovered: {result.expansion_stats['num_entities']}")
        print(
            f"  Total Domain Vertices: {result.expansion_stats['num_domain_vertices']}"
        )
        print(f"  Total Graph Paths: {result.expansion_stats['num_graph_paths']}")
        print(f"  End-to-End Latency: {result.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
