import gradio as gr
import asyncio
import uuid
import logging
from src.orchestrator import app as graph
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.logger import gradio_handler, setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Custom User ID / Thread ID for persistence
THREAD_ID = str(uuid.uuid4())

async def chat_interaction(message, history, zep_key, openrouter_key, domain_filter):
    """Async handler for the Gradio chat interface with enhanced inputs.

    Coordinates message processing via the LangGraph orchestrator, 
    injecting settings from the UI sidebar.

    Args:
        message (str): The user's input message.
        history (List[dict]): The conversation history.
        zep_key (str): Zep API Key from Sidebar.
        openrouter_key (str): OpenRouter API Key from Sidebar.
        domain_filter (str): Selected domain for Bio-Lock.

    Yields:
        tuple: Updated history, stats, reasoning, and logs.
    """
    # 1. Update Environment if keys provided (In production, use more secure ways)
    import os
    if zep_key:
        os.environ["ZEP_API_KEY"] = zep_key
    if openrouter_key:
        os.environ["OPENAI_API_KEY"] = openrouter_key

    # 2. Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "context": "",
        "total_cost": 0.0,
        "recursion_count": 0,
        "budget_exceeded": False,
        "thread_id": THREAD_ID
    }
    
    # 3. Stream events from LangGraph
    history.append({"role": "user", "content": [{"type": "text", "text": message}]})
    yield history, "Updating...", "Retrieving...", gradio_handler.get_logs()
    
    ai_response_content = ""
    history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
    
    # Use config for key/domain if orchestrator supports it, or inject into state
    # For now, orchestrator uses env vars directly and checks content for domain
    # We can pass domain as a 'secret' hint in message or update state
    
    # Strategy: Inject domain choice into the first message for the node to pick it up
    if domain_filter == "Personal":
        initial_state["messages"][0].content += " (Domain: personal)"

    async for event in graph.astream(
        initial_state, 
        config={"configurable": {"thread_id": THREAD_ID}}, 
        stream_mode="values"
    ):
        current_context = event.get("context", "No context retrieved yet.")
        total_cost = event.get("total_cost", 0.0)
        recursion = event.get("recursion_count", 0)
        
        stats_str = f"Cost: ${total_cost:.4f} | Dept: {recursion}"
        
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                ai_response_content = last_msg.content
                # Update the last history item
                history[-1]["content"] = [{"type": "text", "text": ai_response_content}]
                
        yield history, stats_str, current_context, gradio_handler.get_logs()

# UI Theme Definition
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

# Custom CSS for SaaS-Quality "Cockpit" Aesthetics
css = """
/* Global Foundation */
.gradio-container {
    background-color: #111827 !important; /* Softer Dark Gray */
    color: #f3f4f6 !important;
}

/* Sidebar: Refined Glassmorphism */
.sidebar {
    background: rgba(17, 24, 39, 0.8) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 4px 0 24px -12px rgba(0, 0, 0, 0.8);
}

/* Chatbot: Bubble Implementation */
.chatbot {
    border: none !important;
    background: transparent !important;
}
.chatbot .message {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    max-width: 85% !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}
.chatbot .message.user {
    background: #3b82f6 !important; /* Vibrant Blue */
    color: white !important;
    align-self: flex-end !important;
    border-bottom-right-radius: 4px !important;
}
.chatbot .message.bot {
    background: #1f2937 !important; /* Muted Gray */
    color: #f3f4f6 !important;
    align-self: flex-start !important;
    border-bottom-left-radius: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Action Hierarchy Styling */
.icon-btn {
    min-width: 50px !important;
    aspect-ratio: 1/1 !important;
    background: #3b82f6 !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.ghost-btn {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #9ca3af !important;
    font-size: 0.85em !important;
    transition: all 0.2s ease !important;
}
.ghost-btn:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
    color: white !important;
}

/* Input Fields Refinement */
input, textarea, select {
    background-color: #1f2937 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

.stats-markdown {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
    color: #60a5fa !important;
    background: rgba(31, 41, 55, 0.5);
    padding: 8px 12px;
    border-radius: 10px;
    border-left: 3px solid #3b82f6;
}

/* Accordion & Grouping */
.accordion {
    border: none !important;
    background: transparent !important;
}
"""

with gr.Blocks(fill_height=True, title="Agent Nexus v1") as demo:
    with gr.Sidebar(label="Control Center", open=True, elem_classes=["sidebar"]):
        gr.Markdown("# 🛸 Nexus")
        gr.Markdown("---")

        with gr.Accordion("⚙️ Configuration", open=False):
            zep_input = gr.Textbox(label="Zep API Key", type="password", placeholder="ZEP_...")
            or_input = gr.Textbox(label="OpenRouter Key", type="password", placeholder="sk-or-...")
            domain_drop = gr.Dropdown(
                choices=["General", "Personal"], 
                value="General", 
                label="Bio-Lock Domain"
            )
        
        gr.Markdown("### 📊 Live Metrics")
        stats_box = gr.Markdown("Cost: $0.0000 | Depth: 0", elem_classes=["stats-markdown"])
        
        gr.Markdown("---")
        gr.Markdown("Thread: `" + THREAD_ID[:8] + "`")

    # The "Stage" (Main Canvas)
    with gr.Column(scale=4, elem_id="canvas-area"):
        gr.Markdown("## 🧠 Personal Context-Aware Agent")
        
        chatbot = gr.Chatbot(
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Agent"),
            height=600,
            elem_id="main-chatbot"
        )
        
        # Action Bar (Cockpit Controls)
        with gr.Group():
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your objective...",
                    container=False,
                    scale=7,
                    autofocus=True
                )
                submit = gr.Button("🚀", variant="primary", elem_classes=["icon-btn"], min_width=50, scale=0)
            
            with gr.Row():
                stop = gr.Button("🛑 Stop", variant="secondary", elem_classes=["ghost-btn"], scale=1)
                clear = gr.Button("🧹 Clear Session", elem_classes=["ghost-btn"], scale=1)

    # Inspection Panels (Drawer)
    with gr.Accordion("🔍 Forensic Inspection", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Context Context")
                context_display = gr.Textbox(
                    placeholder="Memory will reveal here...",
                    show_label=False,
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("#### Logs")
                log_display = gr.Textbox(
                    show_label=False,
                    lines=8,
                    max_lines=15,
                    interactive=False
                )

    # Logic Integration
    def clear_wrapper():
        gradio_handler.clear()
        return [], "Cost: $0.0000 | Depth: 0", "", ""

    submit_event = msg.submit(
        chat_interaction, 
        inputs=[msg, chatbot, zep_input, or_input, domain_drop], 
        outputs=[chatbot, stats_box, context_display, log_display]
    )
    msg.submit(lambda: "", outputs=msg, queue=False)

    click_event = submit.click(
        chat_interaction, 
        inputs=[msg, chatbot, zep_input, or_input, domain_drop], 
        outputs=[chatbot, stats_box, context_display, log_display]
    )
    submit.click(lambda: "", outputs=msg, queue=False)
    
    clear.click(clear_wrapper, outputs=[chatbot, stats_box, context_display, log_display])
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, click_event])

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=css)
