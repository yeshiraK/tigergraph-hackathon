# Layer 5: Pipeline Policies

This layer encapsulates the three retrieval and reasoning strategies compared on the identical corpus:

1. **RAG (Baseline)**:
   - Chunk-level vector retrieval using TigerGraph vector search.
   - Direct prompt synthesis.
2. **GraphRAG**:
   - Graph-enhanced retrieval (hybrid vector search + entity traversal / subgraph retrieval).
   - Structured context synthesis.
3. **Agentic GraphRAG**:
   - Multi-step, iterative agent reasoning investigated with DeepAgents.
   - Dynamic tool calling via Layer 3 MCP tools to query, traverse, and inspect TigerGraph.
   - Iterative evidence accumulation under Layer 4 budget control.

*Note: Unimplemented in Phase 0A.*
