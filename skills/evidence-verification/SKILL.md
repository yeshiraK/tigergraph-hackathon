---
name: evidence-verification
description: Verifies candidate answers against the Evidence Ledger, detecting insufficient or contradicted evidence and executing at most one targeted repair.
allowed-tools:
  - tigergraph__run_installed_query
  - tigergraph__get_nodes
---

# Evidence Verification Skill

## Purpose
This skill governs evidence sufficiency and truthfulness:
1. Verify that the candidate answer and key question entities exist in the authoritative Evidence Ledger.
2. Confirm temporal constraints (e.g. required year) are satisfied by the retrieved chunks or graph facts.
3. Classify verification state:
   - `SUPPORTED`: All constraints and entities grounded in ledger.
   - `INSUFFICIENT`: Evidence is incomplete or silent on required aspects.
   - `CONTRADICTED`: Evidence directly refutes candidate facts.
   - `REPAIR_REQUIRED`: Targeted retrieval can recover missing supporting documents.

## One-Repair Budget Policy
- **Strict Limit**: Exactly **ONE** repair attempt is permitted.
- If verification returns `REPAIR_REQUIRED`:
  - Identify missing aspect (e.g. missing venue event context, missing temporal document).
  - Execute targeted fallback retrieval through approved retrieval/A2 tools.
  - Re-verify newly added evidence.
  - Never initiate a second repair attempt.
