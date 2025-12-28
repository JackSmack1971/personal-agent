import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from src.orchestrator import (
    AgentState, 
    memory_retrieval_node, 
    reasoning_node, 
    suicide_pact_node, 
    should_continue,
    app as graph
)

class TestOrchestrator(unittest.TestCase):
    """Comprehensive tests for the LangGraph Orchestrator."""

    def setUp(self):
        self.initial_state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "context": "",
            "total_cost": 0.0,
            "recursion_count": 0,
            "budget_exceeded": False,
            "thread_id": "test_thread"
        }

    # ========== HAPPY PATH TESTS ==========
    def test_suicide_pact_happy_path(self):
        """Test suicide_pact_node increments recursion and stays within budget."""
        state = self.initial_state.copy()
        result = suicide_pact_node(state)
        self.assertEqual(result["recursion_count"], 1)
        self.assertFalse(result["budget_exceeded"])

    def test_should_continue_typical(self):
        """Test should_continue logic for normal flow and termination."""
        # 1. Continue if no AI message
        state = self.initial_state.copy()
        self.assertEqual(should_continue(state), "continue")
        
        # 2. End if AI message present
        state["messages"].append(AIMessage(content="Bot reply"))
        self.assertEqual(should_continue(state), "end")

    @patch('src.orchestrator.memory_manager.get_context', new_callable=AsyncMock)
    def test_memory_retrieval_happy_path(self, mock_get_context):
        """Test memory_retrieval_node fetches data."""
        mock_get_context.return_value = "Retrieved Context"
        state = self.initial_state.copy()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(memory_retrieval_node(state))
        self.assertEqual(result["context"], "Retrieved Context")
        loop.close()

    # ========== EDGE CASE TESTS ==========
    def test_suicide_pact_at_limit(self):
        """Test suicide_pact_node when exactly at budget or recursion limits."""
        # 1. Budget limit
        state = self.initial_state.copy()
        state["total_cost"] = 1.01
        result = suicide_pact_node(state)
        self.assertTrue(result["budget_exceeded"])
        
        # 2. Recursion limit
        state = self.initial_state.copy()
        state["recursion_count"] = 51
        result = suicide_pact_node(state)
        self.assertTrue(result["budget_exceeded"])

    def test_should_continue_on_budget_exceeded(self):
        """Test should_continue terminates if budget_exceeded is True."""
        state = self.initial_state.copy()
        state["budget_exceeded"] = True
        self.assertEqual(should_continue(state), "end")

    # ========== ERROR SCENARIO TESTS ==========
    @patch('src.orchestrator.ChatOpenAI')
    def test_reasoning_node_malformed_state(self, mock_chat):
        """Test reasoning_node behavior with missing state keys."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Bot reply")
        mock_chat.return_value = mock_llm
        
        state = {"messages": [HumanMessage(content="x")], "total_cost": 0.0, "context": "..."}
        result = reasoning_node(state)
        self.assertIn("messages", result)
        self.assertGreaterEqual(result["total_cost"], 0.0)

    # ========== INTEGRATION (GRAPH) TEST ==========
    @patch('src.orchestrator.ChatOpenAI')
    @patch('src.orchestrator.memory_manager.get_context', new_callable=AsyncMock)
    def test_graph_full_loop(self, mock_get_context, mock_chat):
        """Verify the full graph execution flow."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Graph AI Response")
        mock_chat.return_value = mock_llm
        
        mock_get_context.return_value = "Graph context"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the graph
        final_state = loop.run_until_complete(graph.ainvoke(self.initial_state))
        
        self.assertTrue(len(final_state["messages"]) > 1)
        self.assertIsInstance(final_state["messages"][-1], AIMessage)
        self.assertFalse(final_state["budget_exceeded"])
        loop.close()

if __name__ == '__main__':
    unittest.main()
