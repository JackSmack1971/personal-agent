# Project Blueprint: Personal Context-Aware Agent
**Version:** 1.0
**Date:** 2024-05-24
**Prepared For:** AI Agent agents (Initial Processing by @orchestrator-project-initialization)
**Human Contact:** [Placeholder: Project Lead, contact@example.com]

## 1. Introduction & Vision
### 1.1. Project Overview
The Personal Context-Aware Agent is a persistent, stateful cognitive architecture designed to move beyond ephemeral, request-response interactions by maintaining continuity across extensive temporal horizons. It utilizes a local-first Gradio interface for user data sovereignty while leveraging cloud infrastructure (Zep and OpenRouter) for high-precision memory retrieval and cost-optimized model-agnostic inference.

### 1.2. Problem Statement / Opportunity
Standard SaaS chatbots are typically stateless, losing context once a session ends and failing to differentiate between intimate personal facts and general technical documentation. This project addresses the need for a digital extension of user cognition that provides a continuous sense of "self" and "history" through cyclic workflows and temporal knowledge graphs.

### 1.3. Core Vision
To empower individuals with a secure, local-first AI companion that autonomously manages complex reasoning tasks and long-term memory while adhering to strict economic and safety boundaries.

## 2. Project Goals & Objectives
### 2.1. Strategic Goals
* Goal 1: Achieve a "local-first" philosophy where orchestration logic and data sovereignty remain under direct user control.
* Goal 2: Implement "Fractal Agent" architectures using deeply nested subgraphs without incurring prohibitive serialization latency.
* Goal 3: Ensure economic safety through proactive, real-time API pricing enforcement to prevent runaway costs.

### 2.2. Specific Objectives (V1 / Initial Release)
* Objective 1.1: Deploy a functional Gradio UI utilizing an asynchronous concurrency model to handle parallel "Map-Reduce" research tasks without interface freezing.
* Objective 1.2: Integrate Zep’s "Graphiti" engine to support hybrid search and JSONPath metadata filtering for precise recall of personal versus technical facts.
* Objective 1.3: Launch with a "Suicide Pact" safety node that utilizes OpenRouter’s dynamic pricing endpoints to terminate loops that exceed defined budget caps.

## 3. Scope
### 3.1. In Scope (Key Deliverables & Functionalities for V1)
* **Cyclic State Machine:** A LangGraph-based orchestration layer that supports iteration, error correction, and state persistence to disk at every "super-step".
* **Temporal Knowledge Graph:** A Zep-backed memory system using bi-temporal tracking (Valid At vs. Created At) to prevent "zombie facts".
* **Rust-Optimized Serialization:** Integration of Fast LangGraph or similar Rust-backed checkpointers to achieve up to 737x faster state management for nested subgraphs.
* **Economic Safety Middleware:** A dedicated node for real-time cost estimation and provider-level price capping via OpenRouter.
* **Context Engineering Pipeline:** Automated trimming and summarization of message history using the Safe Input Limit Formula ($L_{safe}$) to prevent context overflow.

### 3.2. Out of Scope (For V1)
* **Native Mobile Applications:** V1 will focus exclusively on the local web interface (Gradio).
* **Local LLM Hosting:** Initial inference will be managed via OpenRouter to leverage multi-model agnosticism and dynamic routing.
* **Multi-User Enterprise Authentication:** As a "Personal" agent, the focus is on single-user sovereignty and local deployment.
* **Real-time Voice Synthesis:** V1 will prioritize text-based reasoning and memory accuracy over multimodal output.

## 4. Target Users & Audience
* **Primary User Persona 1: The Technical Power User**
    * *Needs:* High-precision retrieval of software versions, error codes, and architectural constraints; modular "sub-agents" for research and coding.
    * *Pain Points:* Semantic ambiguity in vector search (e.g., confusing "Python 3.8" with "Python 3.9") and high costs from unoptimized agent loops.
* **Primary User Persona 2: The Privacy-Conscious Knowledge Worker**
    * *Needs:* Secure storage of personal preferences, relationship histories, and work contexts; a local interface that prevents data leakage.
    * *Pain Points:* Chatbots that "forget" previous instructions and concerns over cloud providers training on sensitive personal data.

## 5. Core Features & High-Level Requirements (V1)

### 5.1. Feature: Deeply Nested Subgraph Orchestration
* **Description:** Modularization of complex behaviors into isolated sub-states (e.g., Research, Scraper, and Coding subgraphs) to maintain focus and scalability.
* **High-Level Requirements/User Stories:**
    * As an agent, I must use Rust-based serialization to pass context between 10+ nested levels in under 1ms per step.
    * As a developer, I must configure a global recursion_limit of at least 200 to prevent premature crashes in deep topologies.
* **Priority:** Must-Have

### 5.2. Feature: High-Precision Memory with JSONPath Filtering
* **Description:** A memory retrieval system that uses Zep’s Graphiti to surgically separate personal biography from technical documentation.
* **High-Level Requirements/User Stories:**
    * The system must apply a "Bio-Lock" filter (`$.domain == "personal"`) when users ask about individual preferences.
    * The system must use Hybrid Search (Vector + BM25) to resolve rare technical identifiers like error codes or version numbers.
* **Priority:** Must-Have

### 5.3. Feature: Economic "Suicide Pact" Safety Node
* **Description:** A deterministic logic gate that queries real-time pricing and terminates the workflow if costs or steps exceed user-defined limits.
* **High-Level Requirements/User Stories:**
    * The node must fetch real-time model pricing from OpenRouter's `/api/v1/models` endpoint at startup.
    * The node must inject a `max_price` parameter into the API request to strictly cap the cost per million tokens.
* **Priority:** Must-Have

### 5.4. Feature: Asynchronous Gradio Interface
* **Description:** A non-blocking UI that allows the user to interact with the agent while parallel research tasks (Map-Reduce) are running in the background.
* **High-Level Requirements/User Stories:**
    * The Gradio handler must be defined as an `async def` and use `graph.astream_events` for real-time status updates.
    * The interface must support a `concurrency_limit=None` to allow "Stop" commands to interleave with running tasks.
* **Priority:** Should-Have

## 6. Open Questions & Areas for ongoing Research
* **Context Trimming Strategies:** Further research is needed to determine when to use "Summarization" versus "Middle-Out Compression," as the latter can create "Phantom Context" and lead to hallucinations in logical reasoning.
* **Subgraph Isolation vs. Sharing:** Evaluation of "Compiled Graph as Node" versus "Invoked Graph" to balance performance with the need for strict logical separation in sandboxed tasks.
* **Token Counting Precision:** Investigating the discrepancy between OpenRouter's normalized token usage and native provider billing to ensure the "Suicide Pact" node remains accurate.
* **Bi-Temporal Reasoning:** Developing standardized logic for agents to handle conflicting information by updating `invalid_at` timestamps in the memory graph.