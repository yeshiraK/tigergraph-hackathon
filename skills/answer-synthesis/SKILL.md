---
name: answer-synthesis
description: Synthesizes final grounded answers strictly from facts and chunks present in the authoritative Evidence Ledger with provenance attribution.
---

# Answer Synthesis Skill

## Purpose
This skill defines the contract for producing the final verified answer:
1. **Grounding Requirement**:
   - The final answer must cite ONLY facts and document/chunk IDs recorded in the Evidence Ledger.
   - Do NOT introduce names, years, medalists, or numbers from internal model memory.
2. **Provenance Attribution**:
   - Include the evidence IDs and graph provenance paths that substantiate the answer.
3. **Insufficiency Reporting**:
   - If the Evidence Ledger fails verification and repair does not recover the missing fact, the agent must report insufficiency rather than fabricating an answer.
