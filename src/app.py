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

# UI Theme Definition: Native Gradio v6 Styling
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    # Global Backgrounds
    body_background_fill="#111827",
    body_text_color="#f3f4f6",
    block_background_fill="#1f2937",
    block_label_text_color="#9ca3af",
    
    # Input Backgrounds
    input_background_fill="#374151",
    input_border_color="rgba(255, 255, 255, 0.1)",

    # Button Colors
    button_primary_background_fill="#3b82f6",
    button_primary_text_color="white",
    
    # Chat specific overrides (if supported by theme variables)
    background_fill_secondary="#111827",
    border_color_primary="rgba(255, 255, 255, 0.05)",
)

# Minimal CSS for branding and glassmorphism (where native variables are insufficient)
css = """
.sidebar-glass {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.stats-card {
    border-left: 4px solid #3b82f6;
    background: rgba(255, 255, 255, 0.03);
    padding: 12px;
    border-radius: 8px;
}

/* Fix for Chatbot background and bubble styling in v6 */
#main-chatbot {
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Force dark theme for Gradio prose/markdown items in chat */
.prose {
    color: #f3f4f6 !important;
}

/* Chat bubble overrides */
.chatbot .message.user {
    background-color: #2563eb !important;
    color: white !important;
}
.chatbot .message.bot {
    background-color: #1f2937 !important;
    color: #f3f4f6 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Layout stabilization for Integrated Submit Button */
.integrated-submit-fix .gr-button {
    height: 42px !important;
    align-self: flex-end !important;
    margin-bottom: 2px !important;
}
"""

with gr.Blocks(fill_height=True, title="Nexus Agent v1") as demo:
    with gr.Sidebar(label="Navigation", open=True, elem_classes=["sidebar-glass"]):
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
        with gr.Group(elem_classes=["stats-card"]):
            stats_box = gr.Markdown("Cost: $0.0000 | Depth: 0")
        
        gr.Markdown("---")
        gr.Markdown("Thread: `" + THREAD_ID[:8] + "`")

    # Main Canvas
    with gr.Column(scale=4):
        gr.Markdown("## 🧠 Personal Context-Aware Agent")
        
        chatbot = gr.Chatbot(
            value=[], # Prevent "Error" state on load
            show_label=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Agent"),
            height=620,
            elem_id="main-chatbot"
        )
        
        # Action Bar: Native Textbox Integration
        with gr.Group(elem_classes=["integrated-submit-fix"]):
            msg = gr.Textbox(
                placeholder="Declare your objective...",
                container=False,
                scale=7,
                autofocus=True,
                show_label=False,
                value="", # Prevent Error state
                buttons=["submit"] # Native v6 Integrated Button
            )
            
            with gr.Row():
                stop = gr.Button("🛑 Stop", variant="secondary", size="sm")
                clear = gr.Button("🧹 Reset", variant="secondary", size="sm")

    # Forensic Inspection
    with gr.Accordion("🔍 Forensic Inspection", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Context Engine")
                context_display = gr.Textbox(
                    value="", # Prevent Error state
                    placeholder="Memory nodes will appear here...",
                    show_label=False,
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("#### Process Logs")
                log_display = gr.Textbox(
                    value="", # Prevent Error state
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
    
    clear.click(clear_wrapper, outputs=[chatbot, stats_box, context_display, log_display])
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event])

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=css)
