import unittest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.orchestrator import reasoning_node, AgentState

class TestReasoningNode(unittest.TestCase):
    """Unit tests for the LLM reasoning integration."""

    def setUp(self):
        self.state: AgentState = {
            "messages": [HumanMessage(content="What is Zep?")],
            "context": "Zep is a long-term memory service.",
            "total_cost": 0.05,
            "recursion_count": 1,
            "budget_exceeded": False,
            "thread_id": "test_thread"
        }

    # ========== HAPPY PATH TESTS ==========
    @patch('src.orchestrator.ChatOpenAI')
    def test_reasoning_node_typical_interaction(self, mock_chat):
        """Test that reasoning_node correctly uses history, context, and updates cost."""
        # Setup mock LLM response
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Zep helps agents remember.", additional_kwargs={"usage": {"total_tokens": 1000}})
        mock_llm.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm
        
        # Execute node
        result = reasoning_node(self.state)
        
        # Assertions
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].content, "Zep helps agents remember.")
        # Cost check: $0.05 (initial) + expected cost for 1000 tokens (e.g. $0.01)
        self.assertGreater(result["total_cost"], 0.05)
        
        # Verify context was in system prompt
        call_args = mock_llm.invoke.call_args[0][0]
        self.assertTrue(any("Zep is a long-term memory service" in m.content for m in call_args if isinstance(m, SystemMessage)))

    # ========== EDGE CASE TESTS ==========
    @patch('src.orchestrator.ChatOpenAI')
    def test_reasoning_node_empty_context(self, mock_chat):
        """Test reasoning_node behavior when context is empty."""
        self.state["context"] = ""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="I don't know.")
        mock_chat.return_value = mock_llm
        
        result = reasoning_node(self.state)
        self.assertEqual(result["messages"][0].content, "I don't know.")

    # ========== ERROR SCENARIO TESTS ==========
    @patch('src.orchestrator.ChatOpenAI')
    def test_reasoning_node_api_failure(self, mock_chat):
        """Test reasoning_node handles API exceptions gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("OpenRouter Down")
        mock_chat.return_value = mock_llm
        
        # Should catch exception and return a fallback message or re-raise?
        # Standard agent behavior: Return a specific error message to state
        result = reasoning_node(self.state)
        self.assertIn("error communicating with my reasoning engine", result["messages"][0].content.lower())

if __name__ == '__main__':
    unittest.main()
