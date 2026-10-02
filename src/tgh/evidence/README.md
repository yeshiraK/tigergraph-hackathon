# Evidence Handling & Grounding

Responsible for managing evidence citations and preventing unsupported answers:
- Extraction of discrete evidence spans from retrieved TigerGraph documents/nodes.
- Provenance tracking (node ID, chunk ID, source document, offset).
- Answer grounding verification against accumulated evidence.
- Refusal mechanism: answers without corpus-grounded evidence are rejected or flagged as unsupported.

*Note: Unimplemented in Phase 0A.*
