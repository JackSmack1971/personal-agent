# 🧠 Personal Context-Aware Agent

A persistent, stateful AI agent designed for long-term context retention and local-first data sovereignty. Powered by **LangGraph** for orchestration and **Zep Cloud** for advanced memory management.

## 🚀 Key Features

- **Fractal Memory**: Continuous context retrieval via Zep Cloud V3 Thread concept.
- **Economic Safety**: Integrated "Suicide Pact" node to prevent runaway API costs and recursion depth.
- **Bio-Lock Filtering**: Domain-specific privacy filters for biographical context.
- **Enhanced UI**: Professional Gradio 6.2.0 interface with:
  - **Sidebar Controls**: API key management, Bio-Lock domain selector, real-time session statistics
  - **Inspection Panels**: Expandable accordion for viewing retrieved context and system logs
  - **Streaming Updates**: Live cost tracking and recursion depth monitoring
- **Self-Healing Design**: Automated dependency auditing and documentation alignment.

## 🛠️ Architecture

The agent operates as a stateful graph:

1. **Safety Check**: Validates budget and recursion limits.
2. **Context Retrieval**: Fetches relevant facts and history from Zep Cloud.
3. **Reasoning**: Processes user intent and context via **OpenRouter** (`google/gemini-2.0-flash-thinking-exp:free`).
4. **Iterative Flow**: Routes back to safety or terminates based on AI response.

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- Zep Cloud API Key (for full functionality)
- OpenAI API Key (for future LLM integration)

### Installation

1. Clone the repository and navigate to the root:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables (**REQUIRED** for reasoning):

   **Option 1: Environment Variables**

   ```bash
   set ZEP_API_KEY=your_zep_key
   set OPENAI_API_KEY=your_openrouter_key
   ```

   **Option 2: UI Sidebar** (after launching the app)
   - Enter API keys directly in the sidebar settings panel
   - Keys are injected into the environment at runtime

### Running the App

```bash
python -m src.app
```

## 🧪 Testing

The project maintains a **94% statement coverage** threshold.

```bash
python -m coverage run -m unittest discover tests
python -m coverage report
```

## 🛡️ License

MIT License. See [LICENSE](LICENSE) for details.
