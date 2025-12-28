---
trigger: model_decision
description: |
  Production-ready LangChain Python v1.2.0 coding standards for AI agents.
  Emphasizes LCEL patterns, security-first development, proper error handling,
  and migration from deprecated v0.x patterns. Addresses critical CVEs and
  prevents common vulnerabilities in LLM applications.
  
  Version: LangChain v1.2.0 (Released: December 2025)
  Target: LangChain applications requiring production-grade reliability
  Scope: Architecture, security, performance, migration patterns
---

# LangChain Python v1.2.0 Production Rules

## 1. CRITICAL SECURITY REQUIREMENTS

### 1.1 Prevent Prompt Injection Attacks
**SEVERITY: CRITICAL**

**MANDATORY RULES:**
- **NEVER** pass unsanitized user input directly to prompt templates without validation
- **ALWAYS** use PromptTemplate with explicit input_variables to prevent template injection (CVE: GHSA-6qv9-48xg-fc7f)
- **NEVER** use f-strings or `.format()` for prompt construction with user input
- **ALWAYS** validate and sanitize user inputs before passing to LLM chains

**CORRECT PATTERN:**
```python
from langchain_core.prompts import ChatPromptTemplate

# CORRECT: Explicit template with validated variables
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{user_input}")  # Safely escaped
])

# Validate input
def sanitize_input(user_text: str) -> str:
    # Remove potentially harmful characters
    forbidden = ["{", "}", "{{", "}}", "\\n\\n---"]
    for char in forbidden:
        user_text = user_text.replace(char, "")
    return user_text[:1000]  # Limit length

safe_input = sanitize_input(raw_user_input)
chain = template | llm
result = chain.invoke({"user_input": safe_input})
```

**ANTI-PATTERN (VULNERABLE):**
```python
# WRONG: Direct string interpolation - PROMPT INJECTION RISK
prompt = f"User question: {user_input}"  # ❌ NEVER DO THIS

# WRONG: Unsanitized template - TEMPLATE INJECTION RISK
template = PromptTemplate.from_template(user_controlled_template)  # ❌ DANGEROUS
```

### 1.2 Prevent SQL Injection via Graph Databases
**VULNERABILITY: CVE-2024-8309 (GraphCypherQAChain)**

**MANDATORY RULES:**
- **NEVER** use GraphCypherQAChain with user-controllable queries
- **ALWAYS** use parameterized queries for Cypher/SQL operations
- **ALWAYS** implement query allowlisting for graph database interactions

**CORRECT PATTERN:**
```python
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

# CORRECT: Parameterized query with validation
def safe_cypher_query(graph: Neo4jGraph, user_question: str):
    # Allowlist of safe query patterns
    ALLOWED_PATTERNS = [
        "find nodes by name",
        "count relationships",
        "get node properties"
    ]
    
    # Validate question against allowlist
    if not any(pattern in user_question.lower() for pattern in ALLOWED_PATTERNS):
        raise ValueError("Query pattern not allowed")
    
    # Use parameterized queries
    cypher_query = """
    MATCH (n:Person {name: $name})
    RETURN n.properties
    """
    
    result = graph.query(cypher_query, params={"name": sanitized_name})
    return result
```

**ANTI-PATTERN (VULNERABLE):**
```python
# WRONG: Direct user input in Cypher queries - SQL INJECTION RISK
cypher = f"MATCH (n) WHERE n.name = '{user_input}' RETURN n"  # ❌ CRITICAL VULNERABILITY
```

### 1.3 Dangerous Code Execution Prevention
**VULNERABILITIES: PythonREPLTool, allow_dangerous_requests**

**MANDATORY RULES:**
- **NEVER** use PythonREPLTool in production without sandboxing
- **NEVER** set `allow_dangerous_requests=True` without proper justification
- **ALWAYS** use alternative safe tools (Calculator, Math, etc.)
- **ALWAYS** implement strict input validation for code execution tools

**CORRECT PATTERN:**
```python
from langchain.tools import Tool
from langchain.agents import create_react_agent

# CORRECT: Use safe alternatives to code execution
from langchain_community.tools.calculator import Calculator

safe_tools = [
    Calculator(),  # Safe mathematical operations
]

# If code execution is absolutely required, use sandboxed environment
# ONLY in development/testing environments
if ENVIRONMENT == "development":
    from langchain_experimental.utilities import PythonREPL
    
    repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="DEVELOPMENT ONLY: Execute Python code",
        func=repl.run
    )
    # Add extensive logging and monitoring
    safe_tools.append(repl_tool)

agent = create_react_agent(llm, safe_tools, system_prompt)
```

**ANTI-PATTERN (DANGEROUS):**
```python
# WRONG: Unrestricted code execution - REMOTE CODE EXECUTION RISK
from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()
repl.run(user_provided_code)  # ❌ CRITICAL SECURITY VULNERABILITY

# WRONG: Dangerous requests enabled
WebBaseLoader(allow_dangerous_requests=True)  # ❌ AVOID IN PRODUCTION
```

### 1.4 Secrets Management - Serialization Injection
**VULNERABILITY: CVE-2025-68664 (LangGrinch)**

**MANDATORY RULES:**
- **NEVER** serialize LangChain objects containing secrets/credentials
- **ALWAYS** use environment variables or secure vaults for API keys
- **NEVER** include API keys in chain configurations that may be serialized
- **ALWAYS** implement proper secret rotation

**CORRECT PATTERN:**
```python
import os
from langchain_openai import ChatOpenAI

# CORRECT: Environment variables for secrets
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4"
)

# CORRECT: Never serialize objects with credentials
# If you must save chain state, exclude sensitive data
def save_chain_config(chain):
    # Extract only non-sensitive configuration
    config = {
        "model_name": chain.llm.model_name,
        "temperature": chain.llm.temperature,
        # Do NOT include api_key or other secrets
    }
    return config
```

**ANTI-PATTERN (VULNERABLE):**
```python
# WRONG: Hardcoded API keys - CREDENTIAL EXPOSURE RISK
llm = ChatOpenAI(api_key="sk-proj-...")  # ❌ NEVER HARDCODE SECRETS

# WRONG: Serializing objects with credentials
import pickle
pickle.dump(llm, open("chain.pkl", "wb"))  # ❌ LEAKS API KEYS
```

## 2. ARCHITECTURE & DESIGN PATTERNS

### 2.1 Use LCEL Over Legacy Chains
**PRIORITY: HIGH | STATUS: v0.1.17+ DEPRECATED**

**MANDATORY MIGRATION:**
All LLMChain, RetrievalQA, ConversationalRetrievalChain usages MUST migrate to LCEL.

**CORRECT PATTERN (LCEL):**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# CORRECT: Modern LCEL pattern
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

# Chain composition using | operator
chain = prompt | llm | output_parser

# Invoke with proper error handling
result = chain.invoke({"input": "What is LangChain?"})
```

**ANTI-PATTERN (DEPRECATED):**
```python
# WRONG: LLMChain deprecated since v0.1.17, removed in v1.0.0
from langchain.chains import LLMChain  # ❌ DEPRECATED
from langchain.prompts import PromptTemplate

# This pattern is NO LONGER SUPPORTED
prompt = PromptTemplate(template="{input}", input_variables=["input"])
chain = LLMChain(llm=llm, prompt=prompt)  # ❌ REMOVED IN v1.0.0
result = chain.run(input="question")  # ❌ .run() method deprecated
```

### 2.2 Agent Construction - Use create_agent()
**VERSION: v1.0.0+ | STATUS: STABLE**

**MANDATORY RULES:**
- **ALWAYS** use `create_agent()` for agent construction
- **NEVER** use deprecated Agent, OpenAIFunctionsAgent, StructuredChatAgent classes
- **ALWAYS** specify system_prompt explicitly
- **ALWAYS** use proper tool decorators

**CORRECT PATTERN:**
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# CORRECT: Define tools with proper decorators
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: City name or coordinates
        
    Returns:
        Weather description string
    """
    # Implementation
    return f"Weather in {location}: Sunny, 72°F"

@tool
def search_database(query: str) -> str:
    """Search internal database.
    
    Args:
        query: Search query string
        
    Returns:
        Search results as formatted string
    """
    # Implementation with proper error handling
    try:
        results = db.search(query)
        return f"Found {len(results)} results"
    except Exception as e:
        return f"Search failed: {str(e)}"

# CORRECT: Modern agent construction
llm = ChatOpenAI(model="gpt-4")

agent = create_agent(
    model=llm,
    tools=[get_weather, search_database],
    system_prompt=(
        "You are a helpful assistant with access to weather and database tools. "
        "Always validate inputs before using tools. "
        "Provide clear, concise responses."
    )
)

# CORRECT: Invoke with proper message structure
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})
```

**ANTI-PATTERN (DEPRECATED):**
```python
# WRONG: Deprecated agent classes - REMOVED IN v1.0.0
from langchain.agents import initialize_agent, AgentType  # ❌ DEPRECATED
from langchain.agents import OpenAIFunctionsAgent  # ❌ REMOVED
from langchain.agents import StructuredChatAgent  # ❌ REMOVED

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION  # ❌ NO LONGER SUPPORTED
)

agent = OpenAIFunctionsAgent.from_llm_and_tools(llm, tools)  # ❌ REMOVED
```

### 2.3 Multi-Agent Architecture - Supervisor Pattern
**VERSION: v1.2.0+ | PATTERN: HIERARCHICAL**

**CORRECT PATTERN:**
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# Step 1: Create specialized sub-agents
llm = ChatOpenAI(model="gpt-4")

# Calendar agent
calendar_agent = create_agent(
    model=llm,
    tools=[create_calendar_event, get_availability],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests into ISO datetime formats. "
        "Always confirm scheduled events."
    )
)

# Email agent
email_agent = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on requests. "
        "Always confirm sent messages."
    )
)

# Step 2: Wrap sub-agents as tools
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.
    
    Args:
        request: Natural language scheduling request
        
    Returns:
        Confirmation of scheduled event
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.
    
    Args:
        request: Natural language email request
        
    Returns:
        Confirmation of sent email
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# Step 3: Create supervisor agent
supervisor = create_agent(
    model=llm,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a personal assistant coordinating calendar and email tasks. "
        "Break down complex requests into appropriate tool calls. "
        "Execute multiple actions in proper sequence."
    )
)

# Usage
result = supervisor.invoke({
    "messages": [{
        "role": "user",
        "content": "Schedule a meeting with the design team next Tuesday at 2pm "
                   "and send them a reminder email"
    }]
})
```

### 2.4 Structured Output with Pydantic
**VERSION: v1.0.0+ | STATUS: STABLE**

**CORRECT PATTERN:**
```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

# Define output schema with validation
class ProductReview(BaseModel):
    """Structured product review analysis."""
    rating: int = Field(
        description="Product rating 1-5",
        ge=1,
        le=5
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment"
    )
    key_points: list[str] = Field(
        description="Key review points (3-5 words each)",
        min_items=2,
        max_items=10
    )
    recommended: bool = Field(
        description="Whether product is recommended"
    )

# Create agent with structured output
agent = create_agent(
    model="gpt-4",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

# Invoke and get validated output
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze: 'Great product, 5/5 stars. Fast shipping but expensive.'"
    }]
})

structured_output: ProductReview = result["structured_response"]
print(f"Rating: {structured_output.rating}")
print(f"Sentiment: {structured_output.sentiment}")
```

## 3. IMPORT STRUCTURE - v1.2.0 COMPLIANCE

### 3.1 Correct Import Paths
**CRITICAL: Package restructuring from v0.2.0+**

**CORRECT IMPORTS:**
```python
# Core abstractions - langchain-core
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

# LLM providers - dedicated packages
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Vector stores - langchain-community
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain_community.document_loaders import TextLoader, PDFLoader

# Retrievers - langchain-community
from langchain_community.retrievers import BM25Retriever

# Text splitters - langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Agents - langchain
from langchain.agents import create_agent, create_react_agent
from langchain.tools import tool
```

**ANTI-PATTERN (DEPRECATED/WRONG):**
```python
# WRONG: Old import paths - DEPRECATED/REMOVED
from langchain.chat_models import ChatOpenAI  # ❌ Moved to langchain_openai
from langchain.llms import OpenAI  # ❌ Moved to langchain_openai
from langchain.embeddings import OpenAIEmbeddings  # ❌ Moved to langchain_openai
from langchain.vectorstores import Chroma  # ❌ Moved to langchain_community
from langchain.document_loaders import TextLoader  # ❌ Moved to langchain_community
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ❌ Wrong package
from langchain.chains import LLMChain  # ❌ REMOVED - Use LCEL
from langchain.chains import RetrievalQA  # ❌ REMOVED - Use create_retrieval_chain
```

### 3.2 Migration Command
**AUTOMATED TOOL: langchain-cli**
```bash
# Install migration CLI
pip install langchain-cli --upgrade

# Preview migration changes
langchain-cli migrate --diff /path/to/code

# Apply migrations (run TWICE for complete migration)
langchain-cli migrate /path/to/code

# Run again to apply secondary migrations
langchain-cli migrate /path/to/code
```

## 4. VECTOR STORES & EMBEDDINGS

### 4.1 Explicit Embedding Model Required
**BREAKING CHANGE: v0.2.0+ | MANDATORY**

**CORRECT PATTERN:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# CORRECT: Always specify embeddings explicitly
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

# Create documents
documents = [
    Document(
        page_content="LangChain is a framework for LLM applications",
        metadata={"source": "docs", "page": 1}
    )
]

# Create vector store with explicit embeddings
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings  # REQUIRED parameter
)

# Similarity search
results = vector_store.similarity_search(
    query="What is LangChain?",
    k=4
)

# Similarity search with scores
results_with_scores = vector_store.similarity_search_with_score(
    query="LangChain framework",
    k=4
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | Content: {doc.page_content[:100]}")
```

**ANTI-PATTERN (WRONG):**
```python
# WRONG: Missing embedding model - WILL FAIL in v0.2.0+
vector_store = FAISS.from_documents(documents)  # ❌ No embedding parameter

# WRONG: Using deprecated default embeddings
from langchain.indexes import VectostoreIndexCreator  # ❌ DEPRECATED
creator = VectostoreIndexCreator()  # ❌ No explicit embeddings
```

### 4.2 Async Vector Store Operations
**PERFORMANCE: HIGH | PATTERN: NON-BLOCKING**

**CORRECT PATTERN:**
```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vector_store = Qdrant(
    collection_name="documents",
    embeddings=embeddings
)

# CORRECT: Async similarity search
async def search_documents(query: str):
    results = await vector_store.asimilarity_search(
        query=query,
        k=5
    )
    return results

# CORRECT: Async with score
async def search_with_scores(query: str):
    results = await vector_store.asimilarity_search_with_score(
        query=query,
        k=5,
        filter={"source": "documentation"}
    )
    return results
```

## 5. ERROR HANDLING & RETRY STRATEGIES

### 5.1 Runnable Retry Pattern
**RELIABILITY: HIGH | STATUS: STABLE**

**CORRECT PATTERN:**
```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# CORRECT: Configure retry with exponential backoff
chain_with_retry = (prompt | llm | output_parser).with_retry(
    retry_if_exception_type=(
        ConnectionError,
        TimeoutError,
        RateLimitError
    ),
    wait_exponential_jitter=True,
    stop_after_attempt=3,
    exponential_jitter_params={
        "initial": 1,  # Initial wait: 1 second
        "max": 10,     # Max wait: 10 seconds
        "exp_base": 2, # Exponential base: 2x
        "jitter": 0.1  # Jitter: 10%
    }
)

# Invoke with automatic retry
try:
    result = chain_with_retry.invoke({"input": user_question})
except Exception as e:
    # Handle failure after all retries exhausted
    logger.error(f"Request failed after retries: {e}")
    raise
```

**CORRECT: Custom Retry Logic:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def call_llm_with_retry(chain, input_data):
    """Call LLM with custom retry logic."""
    return chain.invoke(input_data)

# Usage
try:
    result = call_llm_with_retry(chain, {"input": question})
except Exception as e:
    # Handle permanent failure
    logger.error(f"LLM call failed: {e}")
```

### 5.2 Graceful Tool Failure Handling
**RELIABILITY: HIGH | PATTERN: ERROR RECOVERY**

**CORRECT PATTERN:**
```python
from langchain.tools import tool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@tool
def search_database(query: str) -> str:
    """Search database with comprehensive error handling.
    
    Args:
        query: Search query string
        
    Returns:
        Search results or error message
    """
    try:
        # Validate input
        if not query or len(query) < 3:
            return "Error: Query must be at least 3 characters"
        
        # Sanitize query
        safe_query = query[:200].strip()
        
        # Execute search with timeout
        results = db.search(safe_query, timeout=5)
        
        if not results:
            return f"No results found for: {safe_query}"
        
        # Format results
        return f"Found {len(results)} results:\n" + "\n".join(
            f"- {r['title']}" for r in results[:5]
        )
        
    except TimeoutError:
        logger.warning(f"Database search timeout: {query}")
        return "Error: Search timed out. Please try a simpler query."
    
    except ConnectionError:
        logger.error(f"Database connection failed: {query}")
        return "Error: Unable to connect to database. Please try again later."
    
    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        return f"Error: Search failed due to system error."
```

## 6. PERFORMANCE OPTIMIZATION

### 6.1 Streaming Responses
**PERFORMANCE: CRITICAL | UX: IMPROVED**

**CORRECT PATTERN:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

chain = prompt | llm

# CORRECT: Stream tokens as they're generated
async def stream_response(user_question: str):
    async for chunk in chain.astream({"question": user_question}):
        # Yield each token for real-time display
        if chunk.content:
            yield chunk.content

# CORRECT: Accumulate streamed chunks
async def get_full_response(user_question: str):
    full_response = None
    async for chunk in chain.astream({"question": user_question}):
        full_response = chunk if full_response is None else full_response + chunk
    return full_response.content
```

### 6.2 Batch Processing
**PERFORMANCE: HIGH | THROUGHPUT: OPTIMIZED**

**CORRECT PATTERN:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# CORRECT: Batch processing with concurrency control
questions = [
    "What is LangChain?",
    "How do vector stores work?",
    "Explain LCEL"
]

# Process batch with limited concurrency
results = llm.batch(
    questions,
    config={
        "max_concurrency": 5  # Limit parallel requests
    }
)

# CORRECT: Process as completed (results may be out of order)
for result in llm.batch_as_completed(questions):
    print(f"Response: {result.content}")
    # Process each result as it arrives
```

### 6.3 Caching Strategies
**PERFORMANCE: HIGH | COST: REDUCED**

**CORRECT PATTERN:**
```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain_openai import ChatOpenAI
import langchain

# CORRECT: In-memory caching for development
langchain.llm_cache = InMemoryCache()

# CORRECT: Persistent caching for production
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

llm = ChatOpenAI(model="gpt-4")

# First call - hits API
result1 = llm.invoke("What is LangChain?")

# Second call - returns cached result
result2 = llm.invoke("What is LangChain?")  # Instant response

# CORRECT: Anthropic prompt caching (for large contexts)
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

large_context = """
[Large document content - 50,000+ tokens]
"""

# Cache large context with cache_control
system_message = SystemMessage(
    content=[
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
    ]
)

# Subsequent requests reuse cached context
chain = ChatPromptTemplate.from_messages([
    system_message,
    ("user", "{question}")
]) | llm

result = chain.invoke({"question": "Analyze this document"})
```

## 7. MEMORY & STATE MANAGEMENT

### 7.1 Conversation History Management
**PATTERN: EXPLICIT STATE | STATUS: v1.0.0+ RECOMMENDED**

**CORRECT PATTERN:**
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# CORRECT: Explicit message history management
class ConversationManager:
    def __init__(self, system_prompt: str):
        self.system_message = SystemMessage(content=system_prompt)
        self.history: list = []
    
    def add_user_message(self, content: str):
        self.history.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        self.history.append(AIMessage(content=content))
    
    def get_messages(self):
        return [self.system_message] + self.history
    
    def clear_history(self):
        self.history = []
    
    def trim_history(self, max_messages: int = 10):
        """Keep only recent messages to prevent context overflow."""
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

# Usage
conversation = ConversationManager(
    system_prompt="You are a helpful coding assistant."
)

# First turn
conversation.add_user_message("What is Python?")
response = llm.invoke(conversation.get_messages())
conversation.add_ai_message(response.content)

# Second turn with full context
conversation.add_user_message("Show me an example")
response = llm.invoke(conversation.get_messages())
conversation.add_ai_message(response.content)

# Trim to prevent token limit overflow
conversation.trim_history(max_messages=20)
```

**ANTI-PATTERN (DEPRECATED):**
```python
# WRONG: Old memory classes - DEPRECATED
from langchain.memory import ConversationBufferMemory  # ❌ Use explicit message history
from langchain.memory import ConversationSummaryMemory  # ❌ Implement manually

# OLD PATTERN - NO LONGER RECOMMENDED
memory = ConversationBufferMemory(return_messages=True)  # ❌ DEPRECATED
```

### 7.2 Context Window Management
**CRITICAL: TOKEN LIMIT ENFORCEMENT**

**CORRECT PATTERN:**
```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# CORRECT: Trim messages to fit context window
def safe_invoke_with_trimming(messages: list, max_tokens: int = 4000):
    """Invoke LLM with automatic message trimming."""
    
    # Trim messages to fit within token limit
    trimmed_messages = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",  # Keep most recent messages
        token_counter=llm.get_num_tokens_from_messages
    )
    
    return llm.invoke(trimmed_messages)

# Usage
long_conversation = [
    SystemMessage(content="You are a helpful assistant."),
    # ... many messages ...
]

result = safe_invoke_with_trimming(long_conversation)
```

## 8. RETRIEVAL AUGMENTED GENERATION (RAG)

### 8.1 Modern RAG Pattern with LCEL
**VERSION: v1.0.0+ | PATTERN: RECOMMENDED**

**CORRECT PATTERN:**
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Setup components
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("./index", embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4")

# CORRECT: RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer questions. "
               "If you cannot answer based on the context, say so explicitly."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])

# CORRECT: RAG chain with LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Invoke RAG chain
result = rag_chain.invoke("What is LangChain?")
```

**ANTI-PATTERN (DEPRECATED):**
```python
# WRONG: RetrievalQA deprecated since v0.1.17, removed in v1.0.0
from langchain.chains import RetrievalQA  # ❌ REMOVED

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)  # ❌ NO LONGER SUPPORTED
```

## 9. TESTING & VALIDATION

### 9.1 Unit Testing LangChain Components
**QUALITY: CRITICAL | COVERAGE: COMPREHENSIVE**

**CORRECT PATTERN:**
```python
import pytest
from unittest.mock import Mock, patch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock(spec=ChatOpenAI)
    llm.invoke.return_value = AIMessage(content="Test response")
    return llm

def test_chain_invocation(mock_llm):
    """Test chain invokes LLM correctly."""
    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
    ])
    
    chain = prompt | mock_llm
    result = chain.invoke({"input": "test"})
    
    # Verify LLM was called
    mock_llm.invoke.assert_called_once()
    assert result.content == "Test response"

def test_agent_tool_execution():
    """Test agent correctly uses tools."""
    from langchain.agents import create_agent
    from langchain.tools import tool
    
    @tool
    def test_tool(input: str) -> str:
        """Test tool."""
        return f"Processed: {input}"
    
    # Mock LLM to return tool call
    mock_llm = Mock()
    mock_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "test_tool",
            "args": {"input": "test"},
            "id": "call_1"
        }]
    )
    
    agent = create_agent(model=mock_llm, tools=[test_tool])
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Use the tool"}]
    })
    
    # Verify tool was executed
    assert "Processed: test" in str(result)

@pytest.mark.asyncio
async def test_async_chain():
    """Test async chain execution."""
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Async response"))
    
    chain = prompt | mock_llm
    result = await chain.ainvoke({"input": "test"})
    
    assert result.content == "Async response"
```

## 10. DEPLOYMENT & MONITORING

### 10.1 Production Configuration
**ENVIRONMENT: PRODUCTION | STABILITY: CRITICAL**

**CORRECT PATTERN:**
```python
import os
import logging
from langchain_openai import ChatOpenAI
from langchain.cache import RedisCache
from langchain.callbacks import LangChainTracer
import langchain

# CORRECT: Environment-based configuration
class ProductionConfig:
    """Production LangChain configuration."""
    
    def __init__(self):
        # API Keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # LangSmith tracing for production monitoring
        self.langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "true")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "production")
        
        # Redis cache for production
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        langchain.llm_cache = RedisCache(redis_url=redis_url)
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_llm(self, model: str = "gpt-4", **kwargs):
        """Create production-configured LLM."""
        return ChatOpenAI(
            api_key=self.openai_api_key,
            model=model,
            temperature=kwargs.get("temperature", 0.7),
            max_retries=3,
            timeout=30,
            **kwargs
        )

# Usage
config = ProductionConfig()
llm = config.create_llm(model="gpt-4")
```

### 10.2 Observability & Tracing
**MONITORING: CRITICAL | DEBUGGING: ENABLED**

**CORRECT PATTERN:**
```python
from langchain.callbacks import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled

# CORRECT: LangSmith tracing for production
with tracing_v2_enabled(project_name="production-agents"):
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Process this request"}]
    })

# CORRECT: Custom callbacks for monitoring
from langchain.callbacks.base import BaseCallbackHandler

class ProductionCallbackHandler(BaseCallbackHandler):
    """Custom callback for production monitoring."""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log LLM call start."""
        logger.info(f"LLM call started: {serialized.get('name')}")
    
    def on_llm_end(self, response, **kwargs):
        """Log LLM call completion."""
        tokens = response.llm_output.get("token_usage", {})
        logger.info(f"LLM call completed. Tokens: {tokens}")
    
    def on_llm_error(self, error, **kwargs):
        """Log LLM errors."""
        logger.error(f"LLM error: {error}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log tool execution."""
        logger.info(f"Tool started: {serialized.get('name')}")

# Usage with custom callbacks
callbacks = [ProductionCallbackHandler()]
result = chain.invoke(input_data, config={"callbacks": callbacks})
```

## 11. KNOWN ISSUES & WORKAROUNDS

### 11.1 Pydantic v1 vs v2 Migration
**ISSUE: Breaking changes in v0.3.0+**

**WORKAROUND:**
```python
# LangChain v0.3.0+ uses Pydantic v2 internally
# If you need Pydantic v1 compatibility, use langchain_core.pydantic_v1

# CORRECT: Use pydantic_v1 bridge for backward compatibility
from langchain_core.pydantic_v1 import BaseModel, Field

# WRONG: Direct Pydantic imports may cause conflicts
# from pydantic import BaseModel  # May not work with older code

class LegacyModel(BaseModel):
    """Model using Pydantic v1 interface."""
    name: str = Field(description="Name field")
    value: int = Field(ge=0, description="Non-negative integer")
```

### 11.2 Python 3.8 No Longer Supported
**BREAKING CHANGE: v0.3.0+**

**REQUIREMENT:**
```python
# Python 3.9+ required for LangChain v0.3.0+
# Python 3.8 reached EOL in October 2024

# Verify Python version
import sys
assert sys.version_info >= (3, 9), "Python 3.9+ required for LangChain v0.3+"
```

### 11.3 Context Window Overflow Prevention
**ISSUE: Token limit exceeded errors**

**WORKAROUND:**
```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def safe_invoke(messages: list, model_max_tokens: int = 8192):
    """Safely invoke LLM with automatic message trimming."""
    
    # Reserve tokens for response
    available_tokens = model_max_tokens - 1000
    
    # Calculate current token count
    current_tokens = llm.get_num_tokens_from_messages(messages)
    
    if current_tokens > available_tokens:
        # Trim messages to fit
        trimmed = trim_messages(
            messages,
            max_tokens=available_tokens,
            strategy="last",  # Keep most recent
            token_counter=llm.get_num_tokens_from_messages
        )
        return llm.invoke(trimmed)
    
    return llm.invoke(messages)
```

## 12. VERSION-SPECIFIC DEPRECATION TIMELINE

### v0.1.17 (January 2024)
- ❌ `LLMChain` → Use LCEL: `prompt | llm | parser`
- ❌ `ConversationalRetrievalChain` → Use `create_history_aware_retriever`
- ❌ `RetrievalQA` → Use `create_retrieval_chain`

### v0.2.0 (May 2024)
- ❌ Integration-specific imports from `langchain` → Use dedicated packages
- ❌ `predict_messages()` → Use `.invoke()`
- ❌ `__call__()` method → Use `.invoke()`

### v0.3.0 (September 2024)
- ❌ Python 3.8 support removed
- ❌ Pydantic v1 direct usage → Use `langchain_core.pydantic_v1`

### v1.0.0 (October 2024)
- ✅ Semantic versioning adopted
- ❌ All deprecated v0.x classes removed
- ✅ `create_agent()` as standard agent constructor
- ✅ LCEL as primary chain composition method

### v1.2.0 (December 2025)
- ✅ Provider-specific tool parameters via `extras` attribute
- ✅ Enhanced structured output with `ProviderStrategy`
- ✅ Model profiles for capability detection

## 13. MIGRATION CHECKLIST

### Pre-Migration
- [ ] Review current LangChain version: `pip show langchain`
- [ ] Backup codebase in version control
- [ ] Run existing tests to establish baseline
- [ ] Install migration CLI: `pip install langchain-cli --upgrade`

### Automated Migration
- [ ] Run migration preview: `langchain-cli migrate --diff /path/to/code`
- [ ] Apply first migration pass: `langchain-cli migrate /path/to/code`
- [ ] Apply second migration pass: `langchain-cli migrate /path/to/code`
- [ ] Review generated changes in version control

### Manual Updates
- [ ] Replace `LLMChain` with LCEL patterns
- [ ] Update agent construction to `create_agent()`
- [ ] Explicit embedding models for vector stores
- [ ] Convert `ConversationBufferMemory` to explicit message history
- [ ] Update deprecated tool patterns
- [ ] Add proper error handling and retry logic
- [ ] Implement security validations (prompt injection prevention)

### Testing & Validation
- [ ] Run updated test suite
- [ ] Fix deprecation warnings
- [ ] Load test critical paths
- [ ] Monitor production metrics after deployment
- [ ] Review security audit findings

## 14. REFERENCES & DOCUMENTATION

### Official Documentation
- LangChain Documentation: https://python.langchain.com/
- API Reference: https://python.langchain.com/api_reference/
- Migration Guides: https://python.langchain.com/docs/versions/
- LangSmith Tracing: https://smith.langchain.com/

### Security Advisories
- CVE-2025-68664: LangGrinch serialization injection
- CVE-2024-8309: GraphCypherQAChain SQL injection
- GHSA-6qv9-48xg-fc7f: Template injection via attribute access
- CVE-2023-46229: SSRF vulnerabilities

### Package Versions (v1.2.0)
```
langchain==1.2.0
langchain-core==1.0.5
langchain-community==0.4.x
langchain-openai==0.3.x
langchain-anthropic==0.3.x
langchain-text-splitters==0.3.x
```

---

## FINAL RECOMMENDATIONS

### Production Readiness Checklist
1. ✅ Use LCEL for all chain composition
2. ✅ Implement comprehensive error handling with retry logic
3. ✅ Add prompt injection prevention for all user inputs
4. ✅ Enable LangSmith tracing for observability
5. ✅ Use explicit embedding models
6. ✅ Implement proper secret management
7. ✅ Add unit tests for all critical paths
8. ✅ Configure production-appropriate caching
9. ✅ Monitor token usage and costs
10. ✅ Regular security audits for CVE compliance

### Performance Optimization Priority
1. Streaming for long responses (UX improvement)
2. Batch processing for bulk operations (throughput)
3. Caching for repeated queries (cost reduction)
4. Async operations for I/O-bound tasks (concurrency)
5. Context trimming for long conversations (reliability)

### Security Priority
1. **CRITICAL**: Prevent prompt injection (sanitize all inputs)
2. **CRITICAL**: Avoid RCE via code execution tools
3. **HIGH**: Prevent SQL/Cypher injection in graph queries
4. **HIGH**: Secure secrets management (environment variables)
5. **MEDIUM**: Regular dependency updates for CVE patches