import os
from typing import TypedDict, Annotated, Literal, Any, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from src.memory_manager import memory_manager

# 1. State Definition
class AgentState(TypedDict):
    """Represents the state of the agent within the LangGraph workflow.

    Attributes:
        messages (List[AnyMessage]): A list of messages in the conversation, 
            annotated with `add_messages` to handle state merging.
        context (str): Retrieved context from long-term memory.
        total_cost (float): Accumulated economic cost of the session.
        recursion_count (int): Number of iterations through the graph.
        budget_exceeded (bool): Safety flag triggered by the suicide pact.
        thread_id (str): Unique identifier for the Zep session.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    context: str
    total_cost: float
    recursion_count: int
    budget_exceeded: bool
    thread_id: str

# 2. Node Implementations
async def memory_retrieval_node(state: AgentState) -> dict:
    """Retrieves context from Zep Memory using the MemoryManager.

    Analyzes the last user message for domain hints (e.g., 'personal') 
    to apply Bio-Lock filtering.

    Args:
        state: The current AgentState.

    Returns:
        dict: A dictionary containing the updated 'context'.
    """
    print("--- MEMORY RETRIEVAL ---")
    thread_id = state.get("thread_id", "default_thread")
    # For V1, we default to 'personal' if requested in state, or None
    domain = "personal" if "personal" in state.get("messages", [])[-1].content.lower() else None
    context = await memory_manager.get_context(thread_id, domain=domain)
    return {"context": context}

def reasoning_node(state: AgentState) -> dict:
    """Main reasoning loop using the LLM via OpenRouter.

    Injects retrieved long-term memory context into the system prompt and 
    invokes the LLM with the full conversation history.

    Args:
        state: The current AgentState.

    Returns:
        dict: A dictionary containing the new AI message and updated cost.
    """
    print("--- REASONING ---")
    
    # 1. Initialize OpenRouter Client
    # OpenRouter uses OpenAI compatibility - only need base_url and your key
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="google/gemini-2.0-flash-thinking-exp:free", # Using a powerful free model
        temperature=0.7
    )

    # 2. Prepare System Prompt with Context
    system_prompt = SystemMessage(content=(
        "You are a Personal Context-Aware Agent. Use the following long-term "
        "memory context to personalize your response. If the context is empty, "
        "rely on your general knowledge but remain helpful and concise.\n\n"
        f"--- CONTEXT ---\n{state['context']}\n----------------"
    ))

    # 3. Assemble full message list
    messages = [system_prompt] + state["messages"]

    try:
        # 4. Invoke LLM
        response = llm.invoke(messages)
        
        # 5. Extract Token Usage for Cost Monitoring (Simplified estimation for V1)
        # Standard OpenRouter usage metadata is in additional_kwargs.usage
        usage = response.additional_kwargs.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        # Rough proxy: $0.001 per 1000 tokens for calculation purposes
        added_cost = (total_tokens / 1000) * 0.001
        
        return {"messages": [response], "total_cost": state["total_cost"] + added_cost}
    except Exception as e:
        print(f"Reasoning Error: {e}")
        error_msg = AIMessage(content="I apologize, but I encountered an error communicating with my reasoning engine.")
        return {"messages": [error_msg]}

async def suicide_pact_node(state: AgentState) -> dict:
    """Deterministic Safety Gate: Enforces the 'Suicide Pact' protocol.

    This node runs at the start of every iteration. It queries real-time pricing
    from OpenRouter and calculates L_safe to enforce strict economic limits.

    Formula: L_safe = C_total - (T_output + T_tools + T_safety)
    """
    print("--- SUICIDE PACT: DETERMINISTIC SAFETY GATE ---")
    import httpx
    
    # Constants
    USER_DAILY_BUDGET = float(os.getenv("USER_DAILY_BUDGET", "1.00"))
    MAX_RECURSION = 50
    T_SAFETY_BUFFER = 0.05 # 5% safety margin in USD
    
    current_cost = state.get("total_cost", 0.0)
    current_recursion = state.get("recursion_count", 0)
    
    # 1. Real-time Pricing Fetch
    try:
        async with httpx.AsyncClient() as client:
            # We fetch model pricing to calculate theoretical T_output/T_tools
            # placeholder for specific model extraction
            resp = await client.get("https://openrouter.ai/api/v1/models")
            models = resp.json().get("data", [])
            # Simplified L_safe calculation for V1
            # We assume a fixed capacity vs remaining budget
            t_output_est = 0.01  # Placeholder for next token cost estimate
            t_tools_est = 0.005 # Placeholder for tool invocation cost estimate
            
            l_safe = USER_DAILY_BUDGET - (current_cost + t_output_est + t_tools_est + T_SAFETY_BUFFER)
            print(f"L_SAFE CALCULATION: {l_safe:.4f} remaining")
            
            if l_safe <= 0 or current_recursion >= MAX_RECURSION:
                return {"budget_exceeded": True}
                
    except Exception as e:
        print(f"Safety Gate API Error: {e}. Falling back to deterministic local check.")
        if current_cost >= USER_DAILY_BUDGET or current_recursion >= MAX_RECURSION:
            return {"budget_exceeded": True}

    return {"recursion_count": current_recursion + 1, "budget_exceeded": False}

# 3. Conditional Routing
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determines whether the graph should continue or stop.

    Args:
        state: The current AgentState.

    Returns:
        Literal["continue", "end"]: The next step in the workflow.
    """
    if state["budget_exceeded"]:
        print("!!! BUDGET OR RECURSION LIMIT EXCEEDED - TERMINATING !!!")
        return "end"
    # Basic logic: stop if we have an AI message as the last message (placeholder)
    if isinstance(state["messages"][-1], AIMessage):
        return "end"
    return "continue"

# 4. Graph Construction
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("memory", memory_retrieval_node)
workflow.add_node("agent", reasoning_node)
workflow.add_node("safety", suicide_pact_node)

# Define Edges
workflow.add_edge(START, "safety")
workflow.add_edge("safety", "memory")
workflow.add_edge("memory", "agent")

# Conditional Edge from Agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "safety",
        "end": END
    }
)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    # Test Run
    initial_state = {
        "messages": [HumanMessage(content="Hello agent!")],
        "context": "",
        "total_cost": 0.0,
        "recursion_count": 0,
        "budget_exceeded": False
    }
    for event in app.stream(initial_state):
        print(event)
