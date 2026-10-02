# TigerGraph Knowledge Store (Layer 2)

This directory houses the schema definitions, GSQL queries, loading jobs, and index definitions for TigerGraph.

## Mandatory Architectural Constraint
TigerGraph is the **mandatory graph and vector database backend** for this project.
Do not substitute Neo4j, Qdrant, Milvus, or other databases.

TigerGraph's officially supported hybrid search (combining vector similarity and graph topology traversal) is the preferred retrieval mechanism.

## Subdirectories
- `schema/`: GSQL schema files defining vertex types, edge types, attributes, and graph definitions.
- `queries/`: Parametrized GSQL query scripts for hybrid retrieval, topological expansion, and subgraph extraction.
- `indexes/`: Local index metadata and definition exports (excluded from git).
- `exports/`: Graph export archives and dump snapshots (excluded from git).
