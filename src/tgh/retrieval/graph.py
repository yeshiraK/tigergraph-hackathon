"""Graph traversal and bounded expansion over TigerGraph Stage 1 knowledge graph."""

import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tgh.retrieval.models import (
    DiscoveredEntity,
    DomainVertex,
    EvidenceChunk,
    GraphPath,
    SeedChunk,
)


@dataclass
class GraphExpansionConfig:
    """Configuration limits for bounded graph expansion."""

    seed_top_k: int = 20
    max_seeds_to_expand: int = 5
    max_entities_per_seed: int = 5
    max_domain_vertices_per_seed: int = 5
    max_events_per_person: int = 5
    max_events_per_venue: int = 10
    max_graph_derived_chunks: int = 20
    max_hops: int = 2
    decay_per_hop: float = 0.90
    reinforcement_weight: float = 0.05
    num_workers: int = 6
    adaptive_gating: bool = True
    min_vector_confidence: float = 0.72
    min_vector_margin: float = 0.015


class TigerGraphTraverser:
    """Performs bounded deterministic graph traversal over TigerGraph Cloud."""

    def __init__(
        self,
        conn: Any,
        config: GraphExpansionConfig | None = None,
    ) -> None:
        """Initialize traverser with connection and configuration."""
        self.conn = conn
        self.config = config or GraphExpansionConfig()
        self._edge_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def get_edges(
        self,
        vertex_type: str,
        vertex_id: str,
        edge_type: str = "",
    ) -> list[dict[str, Any]]:
        """Fetch edges for a vertex with local caching and URL-safe ID encoding.

        Args:
            vertex_type: TigerGraph vertex type name.
            vertex_id: Vertex primary ID.
            edge_type: Optional edge type filter.

        Returns:
            List of edge dictionaries returned by pyTigerGraph.
        """
        cache_key = (vertex_type, vertex_id, edge_type)
        if cache_key in self._edge_cache:
            return self._edge_cache[cache_key]

        quoted_id = urllib.parse.quote(vertex_id, safe="")
        try:
            edges = self.conn.getEdges(
                vertex_type,
                quoted_id,
                edgeType=edge_type,
            )
            if not isinstance(edges, list):
                edges = []
        except Exception:
            edges = []

        self._edge_cache[cache_key] = edges
        return edges

    def expand_seeds(
        self,
        seeds: list[SeedChunk],
    ) -> tuple[
        list[DiscoveredEntity],
        list[DomainVertex],
        list[GraphPath],
        list[EvidenceChunk],
    ]:
        """Expand a set of seed chunks up to bounded depth and return evidence.

        Args:
            seeds: List of top vector seed chunks.

        Returns:
            Tuple of (discovered_entities, domain_vertices, graph_paths, graph_chunks).
        """
        if not seeds:
            return [], [], [], []

        active_seeds = seeds[: self.config.max_seeds_to_expand]

        # Use ThreadPoolExecutor to expand seeds concurrently
        with ThreadPoolExecutor(
            max_workers=min(self.config.num_workers, len(active_seeds))
        ) as executor:
            seed_expansions = list(executor.map(self._expand_single_seed, active_seeds))

        # Aggregate and deduplicate across seeds
        all_entities: dict[str, DiscoveredEntity] = {}
        all_domain_vertices: dict[tuple[str, str], DomainVertex] = {}
        all_paths: list[GraphPath] = []
        all_derived_chunks: dict[str, EvidenceChunk] = {}

        for ents, doms, paths, chunks in seed_expansions:
            all_paths.extend(paths)

            for ent in ents:
                if ent.entity_id not in all_entities:
                    all_entities[ent.entity_id] = ent

            for dom in doms:
                dom_key = (dom.vertex_type, dom.vertex_id)
                if dom_key not in all_domain_vertices:
                    all_domain_vertices[dom_key] = dom

            for chunk in chunks:
                cid = chunk.chunk_id
                if cid not in all_derived_chunks:
                    all_derived_chunks[cid] = chunk
                else:
                    # Keep the highest scoring path and merge provenance
                    existing = all_derived_chunks[cid]
                    if chunk.score > existing.score:
                        existing.score = chunk.score
                        existing.hop_distance = chunk.hop_distance
                        existing.seed_chunk_id = chunk.seed_chunk_id
                    existing.provenance.extend(
                        p for p in chunk.provenance if p not in existing.provenance
                    )

        # Sort graph-derived chunks by descending score and enforce limit
        sorted_derived = sorted(
            all_derived_chunks.values(),
            key=lambda c: (c.score, c.chunk_id),
            reverse=True,
        )[: self.config.max_graph_derived_chunks]

        return (
            list(all_entities.values()),
            list(all_domain_vertices.values()),
            all_paths,
            sorted_derived,
        )

    def _expand_single_seed(
        self,
        seed: SeedChunk,
    ) -> tuple[
        list[DiscoveredEntity],
        list[DomainVertex],
        list[GraphPath],
        list[EvidenceChunk],
    ]:
        """Perform bounded graph expansion for one seed chunk."""
        discovered_entities: list[DiscoveredEntity] = []
        domain_vertices: list[DomainVertex] = []
        paths: list[GraphPath] = []
        derived_chunks: list[EvidenceChunk] = []

        # 1. Chunk -> MENTIONS -> Entity
        mentions_edges = self.get_edges("Chunk", seed.chunk_id, "MENTIONS")
        entity_edges = mentions_edges[: self.config.max_entities_per_seed]

        for me in entity_edges:
            ent_id = me.get("to_id", "")
            if not ent_id:
                continue

            ent = DiscoveredEntity(entity_id=ent_id, seed_chunk_id=seed.chunk_id)
            discovered_entities.append(ent)

            path_hop1 = GraphPath(
                source_type="Chunk",
                source_id=seed.chunk_id,
                edge_type="MENTIONS",
                target_type="Entity",
                target_id=ent_id,
                hop=1,
            )
            paths.append(path_hop1)

            # 2. Entity -> RESOLVES_TO -> Domain Vertex
            res_edges = self.get_edges("Entity", ent_id, "RESOLVES_TO")
            for re in res_edges[: self.config.max_domain_vertices_per_seed]:
                target_type = re.get("to_type", "")
                target_id = re.get("to_id", "")
                if not target_type or not target_id:
                    continue

                dom_v = DomainVertex(
                    vertex_type=target_type,
                    vertex_id=target_id,
                    resolved_from_entity_id=ent_id,
                )
                domain_vertices.append(dom_v)

                path_res = GraphPath(
                    source_type="Entity",
                    source_id=ent_id,
                    edge_type="RESOLVES_TO",
                    target_type=target_type,
                    target_id=target_id,
                    hop=1,
                )
                paths.append(path_res)

                # 3. Domain expansion (Hop 2 beyond entity resolution)
                discovered_event_ids: list[
                    tuple[str, str]
                ] = []  # (event_id, provenance_desc)

                if target_type == "Event":
                    discovered_event_ids.append(
                        (target_id, f"{seed.chunk_id} -> {ent_id} -> Event {target_id}")
                    )
                    # Explore event context (Sport / Venue) for provenance
                    for be in self.get_edges("Event", target_id, "BELONGS_TO"):
                        paths.append(
                            GraphPath(
                                source_type="Event",
                                source_id=target_id,
                                edge_type="BELONGS_TO",
                                target_type="Sport",
                                target_id=be.get("to_id", ""),
                                hop=2,
                            )
                        )
                    for he in self.get_edges("Event", target_id, "HELD_AT"):
                        paths.append(
                            GraphPath(
                                source_type="Event",
                                source_id=target_id,
                                edge_type="HELD_AT",
                                target_type="Venue",
                                target_id=he.get("to_id", ""),
                                hop=2,
                            )
                        )

                elif target_type in ("Person", "Team"):
                    # Person/Team -> PARTICIPATED_IN -> Event
                    part_edges = self.get_edges(
                        target_type, target_id, "PARTICIPATED_IN"
                    )
                    for pe in part_edges[: self.config.max_events_per_person]:
                        ev_id = pe.get("to_id", "")
                        if ev_id:
                            prov = (
                                f"{seed.chunk_id} -> {ent_id} -> "
                                f"{target_type} {target_id} -> Event {ev_id}"
                            )
                            discovered_event_ids.append((ev_id, prov))
                            paths.append(
                                GraphPath(
                                    source_type=target_type,
                                    source_id=target_id,
                                    edge_type="PARTICIPATED_IN",
                                    target_type="Event",
                                    target_id=ev_id,
                                    hop=2,
                                )
                            )
                    # Person/Team -> REPRESENTS -> Country (for provenance context)
                    for rep in self.get_edges(target_type, target_id, "REPRESENTS"):
                        paths.append(
                            GraphPath(
                                source_type=target_type,
                                source_id=target_id,
                                edge_type="REPRESENTS",
                                target_type="Country",
                                target_id=rep.get("to_id", ""),
                                hop=2,
                            )
                        )

                elif target_type == "Venue":
                    # Venue -> reverse_HELD_AT -> Event
                    venue_edges = self.get_edges("Venue", target_id, "reverse_HELD_AT")
                    for ve in venue_edges[: self.config.max_events_per_venue]:
                        ev_id = ve.get("to_id", "")
                        if ev_id:
                            prov = (
                                f"{seed.chunk_id} -> {ent_id} -> "
                                f"Venue {target_id} -> Event {ev_id}"
                            )
                            discovered_event_ids.append((ev_id, prov))
                            paths.append(
                                GraphPath(
                                    source_type="Venue",
                                    source_id=target_id,
                                    edge_type="reverse_HELD_AT",
                                    target_type="Event",
                                    target_id=ev_id,
                                    hop=2,
                                )
                            )

                # 4. Resolve discovered Events to Evidence Chunks
                for ev_id, prov_str in discovered_event_ids:
                    # In Stage 1 schema, event entity: entity_event_<ev_id>
                    # mentions chunk <ev_id>#c0000
                    target_chunk_id = f"{ev_id}#c0000"
                    if target_chunk_id == seed.chunk_id:
                        # Skip self-loop
                        continue

                    # Decay score by hop count: 1 hop for Event, 2 hops for Person/Venue
                    hop_dist = 1 if target_type == "Event" else 2
                    decay = self.config.decay_per_hop**hop_dist
                    graph_score = seed.similarity * decay

                    prov_full = f"{prov_str} -> Chunk {target_chunk_id}"
                    derived_chunks.append(
                        EvidenceChunk(
                            chunk_id=target_chunk_id,
                            doc_id=ev_id,
                            score=graph_score,
                            source="graph",
                            hop_distance=hop_dist,
                            provenance=[prov_full],
                            seed_chunk_id=seed.chunk_id,
                        )
                    )

        return (
            discovered_entities,
            domain_vertices,
            paths,
            derived_chunks,
        )
