import os
from typing import List, Optional
try:
    from zep_cloud import ZepClient
    from zep_cloud.api import Message
except ImportError:
    ZepClient = None
    Message = None

class MemoryManager:
    """Manages interactions with Zep Cloud memory services.

    This class handles client initialization, context retrieval with Bio-Lock 
    filtering, interaction storage, and knowledge graph searching. It falls 
    back to a mock mode if Zep Cloud is unavailable.

    Attributes:
        api_key (str): The Zep API key retrieved from arguments or environment.
        client (ZepClient): The initialized Zep Cloud client instance.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initializes the MemoryManager with a Zep API key.

        Args:
            api_key: Optional API key. If not provided, it attempts to 
                fetch ZEP_API_KEY from environment variables.
        """
        self.api_key = api_key or os.getenv("ZEP_API_KEY")
        self.client = None
        if ZepClient and self.api_key:
            self.client = ZepClient(api_key=self.api_key)
        else:
            print("WARNING: ZepClient or API key missing. Running in mock mode.")

    async def get_context(self, thread_id: str, domain: Optional[str] = None) -> str:
        """Retrieves assembled context from Zep for the given thread.

        Applies a 'Bio-Lock' filter if the 'personal' domain is provided, 
        limiting context to the user's biographical data.

        Args:
            thread_id: The unique identifier for the conversation thread.
            domain: Optional domain filter (e.g., 'personal').

        Returns:
            str: The retrieved context string or a mock message in mock mode.
        """
        if not self.client:
            mock_msg = "Mock Context: Performance optimization and state management are priorities."
            if domain == "personal":
                mock_msg += " (Bio-Lock active: User biography only)"
            return mock_msg
        
        try:
            from zep_cloud.api import SearchFilters
            
            filters = None
            if domain:
                filters = SearchFilters(node_labels=[domain] if domain != "personal" else ["Personal"])

            context_response = await self.client.thread.get_user_context(
                thread_id, 
                search_filters=filters
            )
            return context_response.context
        except Exception as e:
            print(f"Zep Retrieval Error: {e}")
            return "Error retrieving context from Zep."

    async def add_interaction(self, thread_id: str, user_msg: str, ai_msg: str):
        """Adds a user and assistant interaction to the Zep thread.

        Args:
            thread_id: The unique identifier for the conversation thread.
            user_msg: The text of the user's message.
            ai_msg: The text of the assistant's response.
        """
        if not self.client:
            return

        try:
            messages = [
                Message(role="user", content=user_msg),
                Message(role="assistant", content=ai_msg)
            ]
            await self.client.thread.add_messages(thread_id, messages=messages)
        except Exception as e:
            print(f"Zep Storage Error: {e}")

    async def search_knowledge_graph(self, user_id: str, query: str) -> List[str]:
        """Searches the Zep Graphiti knowledge graph for relevant facts.

        Args:
            user_id: The unique identifier for the user.
            query: The search query for the knowledge graph.

        Returns:
            List[str]: A list of facts (strings) retrieved from the graph.
        """
        if not self.client:
            return ["Mock Fact: The user is using Gradio 6.2.0."]

        try:
            results = await self.client.graph.search(user_id=user_id, query=query, limit=5)
            return [edge.fact for edge in results.edges]
        except Exception as e:
            print(f"Zep Graph Search Error: {e}")
            return []

# Singleton instance
memory_manager = MemoryManager()
