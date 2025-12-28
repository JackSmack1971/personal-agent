# API Documentation: Orchestrator Graph

This document describes the LangGraph orchestrator that manages the agent's reasoning flow.

## AgentState

The state object shared across all nodes in the graph.

| Attribute | Type | Description |
|-----------|------|-------------|
| `messages` | `List[AnyMessage]` | Annotated list of conversation messages. |
| `context` | `str` | Long-term memory context from Zep. |
| `total_cost` | `float` | Accumulated economic cost (USD). |
| `recursion_count` | `int` | Number of node executions. |
| `budget_exceeded` | `bool` | Flag triggered by economic safety checks. |
| `thread_id` | `str` | Session identifier for memory retrieval. |

## Graph Nodes

### `safety` (suicide_pact_node)

- **Role**: Economic Monitoring.
- **Logic**: Increments `recursion_count`. Sets `budget_exceeded` to `True` if `total_cost > 1.00` or `recursion_count > 50`.
- **Next**: Always routes to `memory`.

### `memory` (memory_retrieval_node)

- **Role**: Context Injection.
- **Logic**: Fetches data from `MemoryManager`. Applies 'personal' domain filter if requested.
- **Next**: Routes to `agent`.

### `agent` (reasoning_node)

- **Role**: Core Intelligence.
- **Logic**: Processes context and history (Placeholder AIMessage generation).
- **Next**: Conditional routing based on `should_continue`.

## Routing Logic

- **continue**: Routes back to `safety` if the last message is not an `AIMessage` and the budget is safe.
- **end**: Terminates if `budget_exceeded` is `True` or an `AIMessage` is detected.
