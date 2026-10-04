---
name: graph-reasoning
description: Executes bounded multi-hop graph traversal and deterministic constraint filtering over TigerGraph relationships with strict provenance tracking.
allowed-tools:
  - tigergraph__get_node
  - tigergraph__get_node_edges
  - tigergraph__get_edges
---

# Graph Reasoning Skill

## Purpose
This skill provides instructions for executing deterministic graph operations over the `OlympicGraphRAG` schema without hub crowding or infinite loops:
- `Venue <- reverse_HELD_AT <- Event`: Find events conducted at a specific venue.
- `Event -> BELONGS_TO -> Sport`: Validate event sport category.
- `Event <- reverse_PARTICIPATED_IN <- Person/Team`: Find medalists and participants.
- `Person/Team -> REPRESENTS -> Country`: Find countries represented.

## Constraints & Safety
1. **Never Execute Mutation**:
   - The agent must only execute read and traversal operations. Schema mutation, vertex insertion, and raw GSQL generation are strictly forbidden.
2. **Early Filtering**:
   - Apply integer `year` filtering on Event candidates immediately to prune the search space before expanding to participants.
3. **Hub Avoidance**:
   - Never expand generic hub nodes (e.g. general Olympic year or sport hubs) without intersecting with specific venue or participant constraints.
4. **Provenance Preservation**:
   - Every graph edge traversed must be captured in the provenance path format: `(VertexA:id) -[EDGE_TYPE]-> (VertexB:id)`.
