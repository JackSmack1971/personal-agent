import gradio as gr
import asyncio
import uuid
import logging
import json
from src.orchestrator import app as graph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.utils.logger import gradio_handler, setup_logging
from src.memory_manager import memory_manager

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Custom User ID / Thread ID for persistence
THREAD_ID = str(uuid.uuid4())
USER_DAILY_BUDGET = 1.00  # USD

async def chat_interaction(message, history, zep_key, openrouter_key, domain_filter):
    """Async handler for the Gradio chat interface with detailed telemetry.

    Uses astream_events to provide real-time updates on internal graph status.
    """
    import os
    if zep_key: os.environ["ZEP_API_KEY"] = zep_key
    if openrouter_key: os.environ["OPENAI_API_KEY"] = openrouter_key

    # Client-side validation
    if not message.strip():
        yield history, "IDLE", 0, 0, "", gradio_handler.get_logs()
        return

    # Initialize State
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "context": "",
        "total_cost": 0.0,
        "recursion_count": 0,
        "budget_exceeded": False,
        "thread_id": THREAD_ID
    }
    
    if domain_filter == "Personal":
        initial_state["messages"][0].content += " (Domain: personal)"

    # Updated history format for Gradio 6 Messages
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    
    current_status = "IDLE"
    cost = 0.0
    recursion = 0
    context_text = ""
    ai_content = ""

    # Stream Events
    async for event in graph.astream_events(
        initial_state, 
        version="v1",
        config={"configurable": {"thread_id": THREAD_ID}}
    ):
        kind = event["event"]
        
        if kind == "on_chain_start":
            current_status = "EXECUTING_GRAPH"
        elif kind == "on_node_start":
            node_name = event["name"]
            current_status = f"NODE: {node_name.upper()}"
        elif kind == "on_node_end":
            output = event["data"].get("output", {})
            cost = output.get("total_cost", cost)
            recursion = output.get("recursion_count", recursion)
            context_text = output.get("context", context_text)
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                ai_content += content
                history[-1]["content"] = ai_content
        elif kind == "on_chain_end":
            current_status = "IDLE"

        # Update Telemetry Sensors
        graph_stats = await memory_manager.get_graph_stats()
        mem_health = f"{graph_stats['active_facts']} ACTIVE | {graph_stats['expired_facts']} EXPIRED"
        
        # Yield updated UI state
        yield (
            history, 
            current_status, 
            cost, 
            recursion, 
            context_text, 
            gradio_handler.get_logs(),
            mem_health
        )

# Bloomberg "Terminal Black" Theme
bloomberg_theme = gr.themes.Default(
    primary_hue="amber",
    neutral_hue="slate",
    font=["JetBrains Mono", "Roboto Mono", "monospace"]
).set(
    # Core Colors
    body_background_fill="#000000",
    block_background_fill="#000000",
    block_border_width="1px",
    block_border_color="#333333",
    
    # Text Primary (Amber)
    body_text_color="#FFB100",
    block_label_text_color="#FFB100",
    
    # Interactive Elements
    input_background_fill="#000000",
    input_border_color="#444444",
    button_primary_background_fill="#FFB100",
    button_primary_text_color="#000000",
)

css = """
.terminal-border { border: 1px solid #333333 !important; }
.status-green { color: #00FF00 !important; }
.amber-text { color: #FFB100 !important; }

/* Bloomberg Chatbot Styling */
#main-chatbot { background-color: #000000 !important; border: 1px solid #333333 !important; }
.chatbot .message.user { 
    background-color: #111111 !important; 
    border: 1px solid #FFB100 !important; 
    color: #FFB100 !important;
    border-radius: 0 !important;
}
.chatbot .message.bot { 
    background-color: #000000 !important; 
    border: 1px solid #00FF00 !important; 
    color: #00FF00 !important;
    border-radius: 0 !important;
}

/* Recursion Gauge Placeholder Styling */
.gauge-container { 
    text-align: center; 
    padding: 10px; 
    border: 1px solid #333333; 
    background: #0a0a0a;
}
"""

with gr.Blocks(title="BLOOMBERG_AGENT_v1.0") as demo:
    with gr.Sidebar(label="CONTEXT_MONITOR", open=True):
        gr.Markdown("### 📡 MEMORY_HEALTH")
        mem_health_display = gr.Markdown("0 ACTIVE | 0 EXPIRED", elem_classes=["status-green"])
        
        gr.Markdown("---")
        gr.Markdown("### 🛠️ OPERATIONS")
        mode_personal = gr.Button("PERSONAL_MODE", variant="secondary", size="sm")
        mode_tech = gr.Button("TECHNICAL_MODE", variant="secondary", size="sm")
        hybrid_search = gr.Checkbox(label="HYBRID_SEARCH", value=True)
        
        gr.Markdown("---")
        with gr.Accordion("CONFIG_CMD", open=False):
            zep_input = gr.Textbox(label="ZEP_KEY", type="password")
            or_input = gr.Textbox(label="OR_KEY", type="password")
            domain_drop = gr.Dropdown(choices=["General", "Personal"], value="General", label="BIO_LOCK")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## ⚡ COMMAND_AND_CONTROL")
            chatbot = gr.Chatbot(
                show_label=False,
                height=700,
                elem_id="main-chatbot"
            )
            
            with gr.Group():
                msg = gr.Textbox(
                    placeholder="ENTER_COMMAND_...",
                    container=False,
                    scale=7,
                    autofocus=True,
                    show_label=False,
                    buttons=["submit"]
                )
        
        # Right Sidebar: Economic Guardrails
        with gr.Column(scale=1, elem_classes=["terminal-border"]):
            gr.Markdown("### ⚖️ ECONOMIC_GUARDRAILS")
            
            status_box = gr.Textbox(label="TELEMETRY_STATUS", value="IDLE", interactive=False, elem_classes=["status-green"])
            
            gr.Markdown("**SESSION_COST_EXPOSURE**")
            cost_bar = gr.Number(label="USD_SPENT", value=0.0, precision=4)
            cost_progress = gr.Slider(label="BUDGET_UTILIZATION (%)", minimum=0, maximum=100, interactive=False)
            
            gr.Markdown("**RECURSION_DEPTH_GAUGE**")
            recursion_count = gr.Number(label="CURRENT_STEPS", value=0)
            
            gr.HTML(f"""
                <div class='gauge-container'>
                    <div class='amber-text' style='font-size: 0.8em;'>LIMIT: 50</div>
                </div>
            """)

    # Forensic Inspection Drawer
    with gr.Accordion("📋 FORENSIC_LOGS", open=False):
        with gr.Row():
            context_display = gr.Textbox(label="RAW_CONTEXT", lines=5, interactive=False)
            log_display = gr.Textbox(label="SYSTEM_TELEMETRY", lines=5, interactive=False)

    # Event Handlers
    def update_budget_slider(cost):
        return (cost / USER_DAILY_BUDGET) * 100

    def clear_session():
        gradio_handler.clear()
        return [], 0, 0, 0, "", "", "0 ACTIVE | 0 EXPIRED"

    # Integration Logic
    submit_event = msg.submit(
        chat_interaction,
        inputs=[msg, chatbot, zep_input, or_input, domain_drop],
        outputs=[chatbot, status_box, cost_bar, recursion_count, context_display, log_display, mem_health_display]
    )
    msg.submit(lambda: "", outputs=msg, queue=False)
    
    cost_bar.change(update_budget_slider, inputs=[cost_bar], outputs=[cost_progress])

if __name__ == "__main__":
    demo.launch(theme=bloomberg_theme, css=css)
