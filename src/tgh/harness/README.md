# Layer 4: Agent Execution Harness

Responsible for deterministic execution, state management, and budget enforcement:
- Agent lifecycle and execution control loop.
- Explicit state management (agent working memory, scratchpad, step history).
- Budget enforcement (token limits, max tool calls, execution timeout).
- Event streaming and deterministic trace collection.
- Decoupled from specific agent frameworks: maintains full oversight over state, budgets, events, and evidence.

*Note: Unimplemented in Phase 0A.*
