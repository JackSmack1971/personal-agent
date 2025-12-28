---

# LangGraph 1.0.5 Production Ruleset

**Version:** LangGraph 1.0.5 (Python)  
**Last Updated:** December 2025  
**Compatibility:** Python 3.9+, requires `langgraph>=1.0.0,<2.0.0`

---

## 1. STATE MANAGEMENT & ARCHITECTURE

### 1.1 State Schema Definition

**MANDATORY RULES:**

```python
# ✅ CORRECT: Use TypedDict with explicit typing
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_step: str
    result: dict[str, Any]
    error_count: int

# ❌ WRONG: Using dict without typing
class State(TypedDict):
    data: dict  # Too generic, no reducer specified
```

**Key Principles:**
- **Always use `TypedDict`** for state schemas (or Pydantic models/dataclasses)
- **Specify reducers** for fields that accumulate (messages, lists, counters)
- **Keep state minimal** - only store what needs to persist across nodes
- **Prefer `MessagesState`** for chat applications to inherit built-in message handling

```python
# ✅ CORRECT: Extend MessagesState for chat applications
from langgraph.graph import MessagesState

class ChatState(MessagesState):
    documents: list[str]
    user_context: dict[str, Any]

# ❌ WRONG: Manually redefining message handling
class ChatState(TypedDict):
    messages: list  # Missing reducer annotation
    documents: list[str]
```

### 1.2 State Reducers

**CRITICAL PATTERNS:**

```python
from typing import Annotated
from operator import add

# ✅ CORRECT: Use operator.add for list accumulation
class State(TypedDict):
    logs: Annotated[list[str], add]
    
# ✅ CORRECT: Custom reducer for complex logic
def merge_dicts(left: dict, right: dict) -> dict:
    """Merge dictionaries with right precedence."""
    return {**left, **right}

class State(TypedDict):
    metadata: Annotated[dict[str, Any], merge_dicts]
```

**Built-in Reducers:**
- `add_messages` - LangChain message handling (updates by ID, appends new)
- `operator.add` - List concatenation
- Custom functions - For domain-specific merge logic

**ANTI-PATTERN: Concurrent Updates Without Reducers**

```python
# ❌ WRONG: Parallel nodes modifying same key without reducer
class State(TypedDict):
    results: list[str]  # Missing reducer!

def node_a(state: State):
    return {"results": ["a"]}  # Overwrites!

def node_b(state: State):
    return {"results": ["b"]}  # Overwrites!

# Error: INVALID_CONCURRENT_GRAPH_UPDATE
```

**SOLUTION:**

```python
# ✅ CORRECT: Use reducer for concurrent updates
class State(TypedDict):
    results: Annotated[list[str], add]

# Now parallel nodes safely append
```

### 1.3 State Immutability

**MANDATORY:**
```python
# ✅ CORRECT: Return partial state updates
def my_node(state: State) -> dict:
    # Don't mutate state directly
    return {"current_step": "processing", "count": state["count"] + 1}

# ❌ WRONG: Mutating state in-place
def bad_node(state: State):
    state["count"] += 1  # NEVER DO THIS
    return state
```

---

## 2. GRAPH CONSTRUCTION

### 2.1 StateGraph Initialization

**MANDATORY PATTERN:**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# ✅ CORRECT: Complete graph initialization
builder = StateGraph(State)

# Add nodes
builder.add_node("node_a", node_a_function)
builder.add_node("node_b", node_b_function)

# Define edges
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

# Compile with checkpointer for state persistence
checkpointer = InMemorySaver()  # Or PostgresSaver for production
graph = builder.compile(checkpointer=checkpointer)
```

**KEY RULES:**
- **Always specify `START` and `END`** - Never use string literals
- **Compile with checkpointer** for state persistence and human-in-the-loop
- **Node functions MUST match state schema** signature

### 2.2 Conditional Edges

**PATTERN:**

```python
# ✅ CORRECT: Conditional routing with type hints
def should_continue(state: State) -> Literal["continue", "end"]:
    """Route based on tool calls in last message."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# ❌ WRONG: Missing type hints and unclear routing
def bad_router(state):
    if state["messages"][-1].content:  # Fragile check
        return "next"
    return "done"  # String not in mapping!
```

**ALTERNATIVE: Command-based Dynamic Routing**

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["next_node"]]:
    """Nodes can return Command for dynamic routing."""
    if state["should_branch"]:
        return Command(
            update={"status": "branched"},
            goto="next_node"
        )
    # Default flow continues
    return {"status": "completed"}
```

### 2.3 Subgraph Integration

**PATTERN:**

```python
# ✅ CORRECT: Checkpointer propagates to subgraphs
subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node("sub_node", sub_function)
subgraph_builder.add_edge(START, "sub_node")
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(ParentState)
builder.add_node("subgraph_step", subgraph)
builder.add_edge(START, "subgraph_step")

# Checkpointer automatically applies to subgraph
graph = builder.compile(checkpointer=checkpointer)
```

---

## 3. CHECKPOINTING & PERSISTENCE

### 3.1 Checkpointer Selection

**DEVELOPMENT:**

```python
from langgraph.checkpoint.memory import InMemorySaver

# ✅ For local development only
checkpointer = InMemorySaver()
```

**PRODUCTION:**

```python
from langgraph.checkpoint.postgres import PostgresSaver

# ✅ CORRECT: Production-grade persistence
DB_URI = "postgresql://user:pass@host:5432/db?sslmode=require"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Call setup() on first deployment only
    # checkpointer.setup()
    
    graph = builder.compile(checkpointer=checkpointer)
```

**CRITICAL RULES:**
- **NEVER use InMemorySaver in production** - state lost on restart
- **Use PostgresSaver** for production (supports horizontal scaling)
- **Call `setup()` once** to initialize database schema
- **Use connection pooling** for high-throughput applications

### 3.2 Thread Configuration

**MANDATORY:**

```python
# ✅ CORRECT: Always specify thread_id for stateful conversations
config = {"configurable": {"thread_id": "user-123-session-abc"}}

# First invocation
result = graph.invoke({"messages": [user_message]}, config)

# Subsequent invocations maintain state
result = graph.invoke({"messages": [followup_message]}, config)
```

**Thread ID Best Practices:**
- Use **unique, deterministic IDs** (user_id + session_id)
- **Never share thread_id** across different conversations
- Include **timestamp or UUID** for session isolation

---

## 4. TOOL INTEGRATION & ERROR HANDLING

### 4.1 ToolNode Usage

**RECOMMENDED PATTERN:**

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

# ✅ CORRECT: Default error handling enabled
tool_node = ToolNode([search_web])

builder.add_node("tools", tool_node)
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")
```

**Error Handling Configuration:**

```python
# ✅ Default: Errors returned as ToolMessage with status='error'
tool_node = ToolNode([search_web])

# ✅ Custom error message
tool_node = ToolNode(
    [search_web],
    handle_tool_errors="Please try a different search query."
)

# ⚠️ Disable error handling (propagates exceptions)
tool_node = ToolNode([search_web], handle_tool_errors=False)

# ✅ Advanced: Custom error handler
def custom_error_handler(state, error: Exception) -> dict:
    return {
        "messages": [{
            "role": "tool",
            "content": f"Tool failed: {str(error)}. Retry with different params.",
            "tool_call_id": state["messages"][-1].tool_calls[0]["id"]
        }]
    }

tool_node = ToolNode([search_web], handle_tool_errors=custom_error_handler)
```

### 4.2 Tool Error Patterns

**CRITICAL:**

```python
# ❌ WRONG: Tool raises unhandled exception
@tool
def fragile_api(param: str) -> str:
    response = requests.get(f"https://api.com/{param}")
    return response.json()["data"]  # KeyError if 'data' missing!

# ✅ CORRECT: Tool handles errors internally or relies on ToolNode
@tool
def robust_api(param: str) -> str:
    """API call with graceful error handling."""
    try:
        response = requests.get(
            f"https://api.com/{param}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            return "Error: API returned unexpected format"
        return data["data"]
    except requests.RequestException as e:
        # If handle_tool_errors=True (default), ToolNode catches this
        raise  # Or return error string
```

### 4.3 Prebuilt Components

**tools_condition Usage:**

```python
from langgraph.prebuilt import tools_condition

# ✅ CORRECT: Automatic routing based on tool_calls
builder.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" if tool_calls present, else END
    ["tools", END]
)

# Equivalent to:
def manual_condition(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

---

## 5. HUMAN-IN-THE-LOOP & INTERRUPTS

### 5.1 Using `interrupt()` Function

**PATTERN:**

```python
from langgraph.types import interrupt

def human_review_node(state: State):
    """Pause for human approval."""
    # Present data to human
    approval_request = {
        "action": state["proposed_action"],
        "confidence": state["confidence"]
    }
    
    # Interrupt execution and wait for human input
    human_input = interrupt(approval_request)
    
    # Process human feedback
    if human_input.get("approved"):
        return {"status": "approved", "action": state["proposed_action"]}
    else:
        return {"status": "rejected", "reason": human_input.get("reason")}

builder.add_node("human_review", human_review_node)
```

**Resuming After Interrupt:**

```python
from langgraph.types import Command

# Check for pending interrupts
state = graph.get_state(config)
if state.interrupts:
    for interrupt_info in state.interrupts:
        print(f"Waiting for input: {interrupt_info.value}")

# Resume with human input
result = graph.invoke(
    Command(resume={"approved": True, "reason": "Looks good"}),
    config
)
```

### 5.2 Multiple Parallel Interrupts

**ADVANCED PATTERN:**

```python
def node_1(state: State):
    value = interrupt({"prompt": "Enter value 1"})
    return {"value_1": value}

def node_2(state: State):
    value = interrupt({"prompt": "Enter value 2"})
    return {"value_2": value}

# Add both nodes in parallel
builder.add_edge(START, "node_1")
builder.add_edge(START, "node_2")

# Resume all interrupts at once
state = graph.get_state(config)
resume_map = {
    interrupt.id: f"input_{i}"
    for i, interrupt in enumerate(state.interrupts)
}

graph.invoke(Command(resume=resume_map), config)
```

**CRITICAL RULE:**
- **Always use `thread_id`** in config when using interrupts
- **Re-execution from interrupt start** - code before `interrupt()` runs again
- **Interrupt IDs** uniquely identify each pending interrupt

---

## 6. STREAMING & ASYNC PATTERNS

### 6.1 Streaming Modes

**AVAILABLE MODES:**

```python
# ✅ stream_mode="values" - Full state after each step
for chunk in graph.stream(input_data, config, stream_mode="values"):
    print(chunk)  # Complete state dict

# ✅ stream_mode="updates" - Only state changes per step
for chunk in graph.stream(input_data, config, stream_mode="updates"):
    print(chunk)  # Partial state updates

# ✅ stream_mode="messages" - LLM tokens as generated (for MessagesState)
for chunk in graph.stream(input_data, config, stream_mode="messages"):
    print(chunk)  # Individual message tokens
```

### 6.2 Async Execution

**PATTERN:**

```python
# ✅ CORRECT: Async streaming for concurrent requests
async def process_multiple_users(user_inputs: list):
    """Handle concurrent user requests."""
    tasks = []
    for i, user_input in enumerate(user_inputs):
        config = {"configurable": {"thread_id": f"user-{i}"}}
        task = graph.ainvoke(user_input, config)
        tasks.append(task)
    
    # Process all concurrently
    results = await asyncio.gather(*tasks)
    return results

# ✅ Async streaming
async for chunk in graph.astream(input_data, config):
    print(chunk)
```

### 6.3 Batch Processing

**EFFICIENT PATTERNS:**

```python
# ✅ Sync batch (sequential)
results = graph.batch([input1, input2, input3])

# ✅ Async batch (concurrent, faster)
results = await graph.abatch([input1, input2, input3])

# ✅ Different configs per input
configs = [
    {"configurable": {"thread_id": "user-1"}},
    {"configurable": {"thread_id": "user-2"}},
    {"configurable": {"thread_id": "user-3"}}
]
results = await graph.abatch([input1, input2, input3], configs)
```

---

## 7. MESSAGE MANAGEMENT

### 7.1 Message Types

**CORRECT USAGE:**

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage
)

# ✅ CORRECT: Proper message construction
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi! How can I help?")
]

# ✅ Tool messages require tool_call_id
tool_message = ToolMessage(
    content="Search results: ...",
    tool_call_id="call_abc123"
)
```

### 7.2 Message Deletion

**PATTERN:**

```python
from langchain_core.messages import RemoveMessage

def trim_messages(state: State):
    """Remove old messages to manage context window."""
    messages = state["messages"]
    
    if len(messages) > 10:
        # Remove first 5 messages (keep system message)
        to_remove = [
            RemoveMessage(id=msg.id) 
            for msg in messages[1:6]  # Skip system message
        ]
        return {"messages": to_remove}
    
    return {}
```

**CRITICAL:**
- `RemoveMessage` only works with `add_messages` reducer
- **Preserve system messages** - don't delete them
- Messages are removed **by ID**, not position

---

## 8. PRODUCTION DEPLOYMENT

### 8.1 Deployment Types

**LangGraph Cloud:**

- **Development**: Minimal resources, non-production only
- **Production**: 
  - Up to **500 requests/second**
  - **Autoscaling** to 10 containers
  - **Metrics-based scaling**: CPU (75%), Memory (75%), Pending runs (10/container)
  - **30-minute cool-down** before scale-down

**Configuration:**

```python
# Production deployment configuration
deployment_config = {
    "type": "production",
    "graph": "path/to/graph.py:graph",
    "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "DATABASE_URL": "${DATABASE_URL}"
    }
}
```

### 8.2 Performance Optimization

**CACHING:**

```python
# Development: In-memory cache
from langgraph.cache import InMemoryCache
cache = InMemoryCache(ttl=3600)  # 1 hour TTL

# Production: Redis cache (distributed)
from langgraph.cache import RedisCache
cache = RedisCache(
    redis_url="redis://host:6379",
    ttl=3600
)

graph = builder.compile(checkpointer=checkpointer, cache=cache)
```

**DATABASE CONNECTION POOLING:**

```python
# ✅ CORRECT: Use connection pooling for PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo=DB_URI,
    min_size=5,
    max_size=20
)

checkpointer = PostgresSaver(pool)
```

### 8.3 Monitoring & Observability

**LOGGING:**

```python
import logging

# ✅ Add structured logging to nodes
logger = logging.getLogger(__name__)

def my_node(state: State):
    logger.info(
        "Node execution started",
        extra={
            "node": "my_node",
            "thread_id": state.get("thread_id"),
            "step": state.get("current_step")
        }
    )
    # ... node logic
    return result
```

**METRICS:**
- Track **execution time** per node
- Monitor **error rates** and types
- Measure **checkpoint size** growth
- Alert on **timeout thresholds**

---

## 9. SECURITY BEST PRACTICES

### 9.1 Input Validation

**MANDATORY:**

```python
from pydantic import BaseModel, Field, field_validator

class UserInput(BaseModel):
    """Validated user input schema."""
    query: str = Field(..., max_length=1000)
    context: dict = Field(default_factory=dict)
    
    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Remove potential injection attempts
        sanitized = v.strip()
        if len(sanitized) < 1:
            raise ValueError("Query cannot be empty")
        return sanitized

def entry_node(state: State):
    """Validate input before processing."""
    try:
        validated = UserInput(**state["user_input"])
        return {"validated_input": validated.model_dump()}
    except ValidationError as e:
        return {"error": str(e), "status": "invalid_input"}
```

### 9.2 API Key Management

**CRITICAL:**

```python
# ❌ WRONG: Hardcoded API keys
ANTHROPIC_API_KEY = "sk-ant-..."  # NEVER DO THIS

# ✅ CORRECT: Environment variables
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set")
```

### 9.3 Rate Limiting

**PATTERN:**

```python
from functools import wraps
import time

def rate_limit(calls: int, period: int):
    """Decorator for rate limiting."""
    calls_made = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls_made[:] = [c for c in calls_made if c > now - period]
            
            if len(calls_made) >= calls:
                sleep_time = period - (now - calls_made[0])
                time.sleep(sleep_time)
            
            calls_made.append(time.time())
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(calls=10, period=60)  # 10 calls per minute
def call_external_api(param: str):
    # API call logic
    pass
```

---

## 10. KNOWN ISSUES & MITIGATIONS

### 10.1 INVALID_CONCURRENT_GRAPH_UPDATE

**PROBLEM:**
Parallel nodes updating the same state key without a reducer.

**SOLUTION:**

```python
# ❌ CAUSES ERROR
class State(TypedDict):
    results: list[str]  # No reducer!

# ✅ FIXED
class State(TypedDict):
    results: Annotated[list[str], add]
```

### 10.2 Message ID Conflicts

**PROBLEM:**
Manually creating messages without unique IDs.

**SOLUTION:**

```python
# ✅ CORRECT: Let LangChain generate IDs
from langchain_core.messages import AIMessage

message = AIMessage(content="Response")
# ID auto-generated

# ❌ WRONG: Reusing IDs
message = AIMessage(content="Response", id="fixed-id")  # Problematic
```

### 10.3 Checkpointer Memory Leaks

**PROBLEM:**
Using InMemorySaver in long-running production applications.

**MITIGATION:**

```python
# ❌ WRONG: InMemorySaver in production (memory grows unbounded)
checkpointer = InMemorySaver()

# ✅ CORRECT: PostgresSaver with periodic cleanup
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Implement periodic checkpoint pruning
    # DELETE FROM checkpoints WHERE created_at < NOW() - INTERVAL '30 days'
    pass
```

### 10.4 Interrupt Re-execution

**PROBLEM:**
Code before `interrupt()` executes again on resume.

**AWARENESS:**

```python
def node_with_interrupt(state: State):
    # This code runs TWICE: initial execution + resume
    expensive_computation = process_data(state["input"])  # ⚠️ Runs again!
    
    user_input = interrupt({"data": expensive_computation})
    
    return {"result": combine(expensive_computation, user_input)}

# ✅ MITIGATION: Cache expensive operations in state
def better_node(state: State):
    if "cached_computation" not in state:
        state["cached_computation"] = process_data(state["input"])
    
    user_input = interrupt({"data": state["cached_computation"]})
    return {"result": combine(state["cached_computation"], user_input)}
```

---

## 11. MIGRATION FROM 0.x TO 1.0

### 11.1 Breaking Changes

**IMPORT PATHS:**

```python
# ❌ OLD (0.x)
from langgraph.graph import MessageGraph
from langgraph.checkpoint import MemorySaver

# ✅ NEW (1.0+)
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
```

**COMMAND API:**

```python
# ❌ OLD (0.x)
graph.update_state(config, {"key": "value"})

# ✅ NEW (1.0+)
graph.invoke(Command(update={"key": "value"}), config)
```

### 11.2 Deprecated Patterns

**MessageGraph → StateGraph with MessagesState:**

```python
# ❌ DEPRECATED
from langgraph.graph import MessageGraph

graph = MessageGraph()

# ✅ CURRENT
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
```

---

## 12. TESTING STRATEGIES

### 12.1 Unit Testing Nodes

```python
import pytest
from unittest.mock import Mock

def test_node_function():
    """Test individual node logic."""
    mock_state = {
        "messages": [HumanMessage(content="test")],
        "count": 0
    }
    
    result = my_node(mock_state)
    
    assert result["count"] == 1
    assert "processed" in result

def test_conditional_routing():
    """Test routing logic."""
    state_with_tools = {
        "messages": [AIMessage(content="", tool_calls=[{...}])]
    }
    
    assert should_continue(state_with_tools) == "continue"
    
    state_no_tools = {
        "messages": [AIMessage(content="Done")]
    }
    
    assert should_continue(state_no_tools) == "end"
```

### 12.2 Integration Testing

```python
@pytest.mark.asyncio
async def test_full_graph_execution():
    """Test complete graph flow."""
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-123"}}
    
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="test query")]},
        config
    )
    
    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result["messages"][-1].content  # Has response
```

---

## 13. DEBUGGING & TROUBLESHOOTING

### 13.1 Graph Visualization

```python
from IPython.display import Image, display

# Generate Mermaid diagram
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Visualization failed: {e}")
    # Alternative: Print ASCII representation
    print(graph.get_graph())
```

### 13.2 State Inspection

```python
# Get current state and history
state = graph.get_state(config)

print(f"Current values: {state.values}")
print(f"Next nodes: {state.next}")
print(f"Interrupts: {state.interrupts}")
print(f"Config: {state.config}")

# Get state history
for historical_state in graph.get_state_history(config):
    print(f"Step: {historical_state.values}")
```

### 13.3 Common Error Messages

**"No checkpointer set":**
```python
# ✅ FIX: Always compile with checkpointer for stateful graphs
graph = builder.compile(checkpointer=InMemorySaver())
```

**"Missing thread_id":**
```python
# ✅ FIX: Provide thread_id in config
config = {"configurable": {"thread_id": "unique-id"}}
```

**"Node not found":**
```python
# ✅ FIX: Ensure node names match exactly
builder.add_node("my_node", my_function)
builder.add_edge(START, "my_node")  # Exact match required
```

---

## 14. VERSION-SPECIFIC NOTES

### LangGraph 1.0.5 Highlights

**Stable Features:**
- ✅ Command API for dynamic routing
- ✅ `interrupt()` function for human-in-the-loop
- ✅ Prebuilt components (ToolNode, tools_condition, create_react_agent)
- ✅ Multiple checkpointer backends (Memory, Postgres, Async implementations)
- ✅ Async/await support throughout
- ✅ Streaming modes (values, updates, messages)

**Known Limitations:**
- ⚠️ Subgraph interrupt handling - interrupts in deeply nested subgraphs may require special handling
- ⚠️ Large state sizes - checkpoint serialization slows with states >1MB
- ⚠️ PostgreSQL version - requires PostgreSQL 12+

**Recommended Versions:**
- Python: 3.9 - 3.12
- PostgreSQL: 14+ (for PostgresSaver)
- Redis: 6+ (for caching)

---

## 15. QUICK REFERENCE

### Essential Imports

```python
# Core
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt

# Prebuilt
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent

# Messages
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, 
    ToolMessage, RemoveMessage
)
from langgraph.graph.message import add_messages

# Types
from typing import TypedDict, Annotated, Literal, Any
from typing_extensions import TypedDict  # Python 3.9
from operator import add
```

### Minimal Working Example

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def chatbot(state: State):
    # Your LLM call here
    return {"messages": [AIMessage(content="Response")]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Use
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config
)
```

---

## CRITICAL SUCCESS CHECKLIST

Before deploying to production:

- [ ] State schema uses `TypedDict` with proper reducers
- [ ] Checkpointer configured (PostgresSaver for production)
- [ ] All edges defined (`START`, `END`, conditionals)
- [ ] Error handling implemented (ToolNode or custom)
- [ ] Thread IDs unique and deterministic
- [ ] Input validation for user data
- [ ] API keys in environment variables
- [ ] Monitoring and logging configured
- [ ] Rate limiting on external calls
- [ ] Tests cover critical paths
- [ ] Graph visualization reviewed
- [ ] Async patterns used for concurrency
- [ ] Message trimming strategy implemented
- [ ] Checkpoint cleanup scheduled

---

**Documentation Source:** Official LangGraph Documentation, Community Best Practices (Dec 2025)  
**Maintained By:** LangChain Team  
**Support:** https://docs.langchain.com/oss/python/langgraph

---

This ruleset is designed for **copy-paste readiness** with production-tested patterns. Each section includes working code examples, anti-patterns to avoid, and specific version compatibility notes for LangGraph 1.0.5.