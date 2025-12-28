import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import os
import asyncio
import sys

# Mock zep_cloud and zep_cloud.api
mock_zep_pkg = MagicMock()
mock_zep_api = MagicMock()
sys.modules['zep_cloud'] = mock_zep_pkg
sys.modules['zep_cloud.api'] = mock_zep_api

# Ensure mock_zep_pkg has the expected structure
mock_zep_pkg.ZepClient = MagicMock()
mock_zep_api.Message = MagicMock()

from src.memory_manager import MemoryManager

# Helper to run async code in tests
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class TestMemoryManager(unittest.TestCase):
    """Comprehensive tests for MemoryManager."""

    def setUp(self):
        self.api_key = "test_key"
        # Force mock mode by default
        with patch('src.memory_manager.ZepClient', None):
            self.mm_mock = MemoryManager(api_key=None)

    # ========== HAPPY PATH TESTS ==========
    @patch('src.memory_manager.ZepClient')
    def test_init_with_key(self, mock_zep):
        """Test MemoryManager initialization with an API key."""
        mm = MemoryManager(api_key="manual_key")
        self.assertEqual(mm.api_key, "manual_key")
        mock_zep.assert_called_once_with(api_key="manual_key")

    def test_get_context_mock_mode(self):
        """Test get_context in mock mode (no client)."""
        context = run_async(self.mm_mock.get_context("thread_1"))
        self.assertIn("Mock Context", context)

    def test_get_context_bio_lock_mock(self):
        """Test get_context Bio-Lock branch in mock mode."""
        context = run_async(self.mm_mock.get_context("thread_1", domain="personal"))
        self.assertIn("Bio-Lock active", context)

    def test_get_context_success(self):
        """Test successful context retrieval with a mocked client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.context = "Real Cloud Context"
        mock_client.thread.get_user_context = AsyncMock(return_value=mock_response)
        
        mm = MemoryManager(api_key="test")
        mm.client = mock_client
        
        context = run_async(mm.get_context("t1", domain="tech"))
        self.assertEqual(context, "Real Cloud Context")

    def test_add_interaction_success(self):
        """Test successful interaction storage."""
        mock_client = MagicMock()
        mock_client.thread.add_messages = AsyncMock(return_value=None)
        
        mm = MemoryManager(api_key="test")
# Mocking the Message class which is imported inside add_interaction if needed? 
# Wait, it's imported at top level in memory_manager.py but might fail
        mm.client = mock_client
        
        # We need to ensure the Message class used inside is mockable or handle it
        with patch('src.memory_manager.Message') as mock_msg_class:
            mock_msg_class.return_value = MagicMock()
            run_async(mm.add_interaction("t1", "u", "a"))
            mock_client.thread.add_messages.assert_called_once()

    def test_search_graph_success(self):
        """Test successful knowledge graph search."""
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_edge = MagicMock()
        mock_edge.fact = "Reality is often disappointing."
        mock_results.edges = [mock_edge]
        mock_client.graph.search = AsyncMock(return_value=mock_results)
        
        mm = MemoryManager(api_key="test")
        mm.client = mock_client
        
        results = run_async(mm.search_knowledge_graph("u1", "truth"))
        self.assertEqual(results[0], "Reality is often disappointing.")

    # ========== EDGE CASE TESTS ==========
    def test_add_interaction_no_client(self):
        """Test add_interaction does nothing in mock mode."""
        result = run_async(self.mm_mock.add_interaction("t1", "u", "a"))
        self.assertIsNone(result)

    def test_search_graph_mock_mode(self):
        """Test search_knowledge_graph returns mock fact in mock mode."""
        results = run_async(self.mm_mock.search_knowledge_graph("u1", "test"))
        self.assertEqual(len(results), 1)
        self.assertIn("Gradio 6.2.0", results[0])

    # ========== ERROR SCENARIO TESTS ==========
    def test_get_context_exception_handling(self):
        """Test get_context handles Zep API exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.thread.get_user_context = AsyncMock(side_effect=Exception("API Error"))
        
        mm = MemoryManager(api_key="test")
        mm.client = mock_client
        
        context = run_async(mm.get_context("thread_1"))
        self.assertEqual(context, "Error retrieving context from Zep.")

if __name__ == '__main__':
    unittest.main()
