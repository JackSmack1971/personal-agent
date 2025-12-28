import unittest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import sys
import pytest
from langchain_core.messages import HumanMessage, AIMessage

class TestUIIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests for the glue logic between Gradio UI and LangGraph orchestrator."""

    def setUp(self):
        """Set up mocks before each test to prevent Gradio initialization."""
        # Create a mock gradio module to prevent real Gradio from loading
        self.mock_gradio = MagicMock()
        self.mock_gradio.Blocks = MagicMock()
        self.mock_gradio.Sidebar = MagicMock()
        self.mock_gradio.Chatbot = MagicMock()
        self.mock_gradio.Textbox = MagicMock()
        self.mock_gradio.Button = MagicMock()
        self.mock_gradio.Markdown = MagicMock()
        self.mock_gradio.Dropdown = MagicMock()
        self.mock_gradio.Accordion = MagicMock()
        self.mock_gradio.Column = MagicMock()
        self.mock_gradio.Row = MagicMock()
        self.mock_gradio.Code = MagicMock()
        self.mock_gradio.themes = MagicMock()
        
    @pytest.mark.skip(reason="Module reloading causes RuntimeError with torch functions")
    async def test_chat_interaction_updates_environment_vars(self):
        """Verify that API keys from UI are injected into environment variables."""
        # Mock all dependencies before importing
        with patch.dict('sys.modules', {'gradio': self.mock_gradio}), \
             patch('src.orchestrator.app') as mock_graph, \
             patch('os.environ', {}) as mock_env:
            
            # Mock gradio_handler
            mock_handler = MagicMock()
            mock_handler.get_logs.return_value = "Test logs"
            
            with patch('src.utils.logger.gradio_handler', mock_handler), \
                 patch('src.utils.logger.setup_logging'):
                
                # Now safe to import - Gradio is mocked
                import importlib
                if 'src.app' in sys.modules:
                    importlib.reload(sys.modules['src.app'])
                else:
                    import src.app
                
                from src.app import chat_interaction
                
                # Mock the graph.astream to return a simple event
                mock_event = {
                    "messages": [AIMessage(content="Test response")],
                    "context": "Test context",
                    "total_cost": 0.01,
                    "recursion_count": 1
                }
                mock_graph.astream.return_value.__aiter__.return_value = [mock_event]
                
                # Call chat_interaction with UI parameters
                message = "Hello"
                history = []
                zep_key = "test_zep_key"
                openrouter_key = "test_openrouter_key"
                domain_filter = "General"
                
                results = []
                async for result in chat_interaction(message, history, zep_key, openrouter_key, domain_filter):
                    results.append(result)
                
                # Verify environment variables were set
                import os
                self.assertEqual(os.environ.get("ZEP_API_KEY"), zep_key)
                self.assertEqual(os.environ.get("OPENAI_API_KEY"), openrouter_key)

    @pytest.mark.skip(reason="Module reloading causes RuntimeError with torch functions")
    async def test_chat_interaction_domain_filter_injection(self):
        """Verify that Bio-Lock domain filter is injected into the message."""
        with patch.dict('sys.modules', {'gradio': self.mock_gradio}), \
             patch('src.orchestrator.app') as mock_graph, \
             patch('os.environ', {}):
            
            mock_handler = MagicMock()
            mock_handler.get_logs.return_value = ""
            
            with patch('src.utils.logger.gradio_handler', mock_handler), \
                 patch('src.utils.logger.setup_logging'):
                
                import importlib
                if 'src.app' in sys.modules:
                    importlib.reload(sys.modules['src.app'])
                else:
                    import src.app
                    
                from src.app import chat_interaction
                
                mock_event = {
                    "messages": [AIMessage(content="Response")],
                    "context": "Context",
                    "total_cost": 0.0,
                    "recursion_count": 0
                }
                mock_graph.astream.return_value.__aiter__.return_value = [mock_event]
                
                # Test with Personal domain
                results = []
                async for result in chat_interaction("Test message", [], "", "", "Personal"):
                    results.append(result)
                
                # Verify graph.astream was called
                self.assertTrue(mock_graph.astream.called)
                
                # Get the initial_state argument
                call_args = mock_graph.astream.call_args
                initial_state = call_args[0][0]
                
                # Verify domain hint was injected
                self.assertIn("(Domain: personal)", initial_state["messages"][0].content)

    async def test_chat_interaction_yields_correct_outputs(self):
        """Verify that chat_interaction yields the correct tuple format."""
        with patch.dict('sys.modules', {'gradio': self.mock_gradio}), \
             patch('src.orchestrator.app') as mock_graph, \
             patch('os.environ', {}):
            
            mock_handler = MagicMock()
            mock_handler.get_logs.return_value = "System log output"
            
            with patch('src.utils.logger.gradio_handler', mock_handler), \
                 patch('src.utils.logger.setup_logging'):
                
                import importlib
                if 'src.app' in sys.modules:
                    importlib.reload(sys.modules['src.app'])
                else:
                    import src.app
                    
                from src.app import chat_interaction
                
                mock_event = {
                    "messages": [AIMessage(content="AI Response")],
                    "context": "Retrieved context",
                    "total_cost": 0.05,
                    "recursion_count": 2
                }
                mock_graph.astream.return_value.__aiter__.return_value = [mock_event]
                
                results = []
                async for result in chat_interaction("Hello", [], "", "", "General"):
                    results.append(result)
                
                # Verify we got results
                self.assertGreater(len(results), 0)
                
                # Check the structure of the last result
                last_result = results[-1]
                self.assertEqual(len(last_result), 4)  # (history, stats, context, logs)
                
                history, stats, context, logs = last_result
                
                # Verify types and content
                self.assertIsInstance(history, list)
                self.assertIn("Cost:", stats)
                self.assertIn("Dept:", stats)
                self.assertEqual(context, "Retrieved context")
                self.assertEqual(logs, "System log output")

if __name__ == '__main__':
    unittest.main()
