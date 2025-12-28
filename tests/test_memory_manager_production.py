import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import time
from src.memory_manager import MemoryManager

# Helper to run async code in tests
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class TestMemoryManagerProduction(unittest.TestCase):
    """TDD tests for Production-Grade Zep Integration."""

    def setUp(self):
        self.api_key = "test_key"
        # We'll use a patch for ZepClient in tests that need it
        
    # ========== AUTHENTICATION VALIDATION TESTS ==========
    @patch('src.memory_manager.ZepClient')
    def test_validate_connection_success(self, mock_zep):
        """Test that validate_connection returns True on success."""
        mock_client = MagicMock()
        mock_client.user.list_all = AsyncMock(return_value=MagicMock())
        mock_zep.return_value = mock_client
        
        mm = MemoryManager(api_key=self.api_key)
        mm.client = mock_client
        
        result = run_async(mm.validate_connection())
        self.assertTrue(result)

    @patch('src.memory_manager.ZepClient')
    def test_validate_connection_failure(self, mock_zep):
        """Test that validate_connection returns False on auth failure."""
        mock_client = MagicMock()
        # Simulate unauthorized error
        mock_client.user.list_all = AsyncMock(side_effect=Exception("401 Unauthorized"))
        mock_zep.return_value = mock_client
        
        mm = MemoryManager(api_key=self.api_key)
        mm.client = mock_client
        
        result = run_async(mm.validate_connection())
        self.assertFalse(result)

    # ========== RETRY LOGIC TESTS ==========
    @patch('src.memory_manager.ZepClient')
    def test_retry_on_transient_error(self, mock_zep):
        """Test that methods retry on 429 errors."""
        mock_client = MagicMock()
        # Fail twice with 429, then succeed
        mm = MemoryManager(api_key=self.api_key)
        mm.client = mock_client
        
        with patch.object(mm.client.thread, 'get_user_context', new_callable=AsyncMock) as mock_api_call:
            mock_api_call.side_effect=[
                Exception("429 Too Many Requests"),
                Exception("429 Too Many Requests"),
                MagicMock(context="Success After Retry")
            ]
            
            context = run_async(mm.get_context("thread_1"))
            print(f"DEBUG Happy Path Context: {context}")
            self.assertEqual(context, "Success After Retry")
            self.assertEqual(mock_api_call.call_count, 3)

    @patch('src.memory_manager.ZepClient')
    def test_no_retry_on_permanent_error(self, mock_zep):
        """Test that methods do NOT retry on 404 errors."""
        mm = MemoryManager(api_key=self.api_key)
        mm.client = MagicMock()
        
        with patch.object(mm.client.thread, 'get_user_context', new_callable=AsyncMock) as mock_api_call:
            mock_api_call.side_effect=Exception("404 Not Found")
            
            context = run_async(mm.get_context("thread_1"))
            print(f"DEBUG Permanent Error Context: {context}")
            # MemoryManager returns this string for non-transient Errors (like 404)
            self.assertEqual(context, "Error retrieving context from Zep.")
            self.assertEqual(mock_api_call.call_count, 1)

if __name__ == '__main__':
    unittest.main()
