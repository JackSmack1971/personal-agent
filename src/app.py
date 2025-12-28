import gradio as gr
import asyncio
import uuid
from src.orchestrator import app as graph
from langchain_core.messages import HumanMessage, AIMessage

# Custom User ID / Thread ID for persistence
THREAD_ID = str(uuid.uuid4())

async def chat_interaction(message, history):
    """Async handler for the Gradio chat interface.

    Coordinates message processing via the LangGraph orchestrator, 
    streaming intermediate status and final AI responses back to the UI.

    Args:
        message (str): The user's input message.
        history (List[dict]): The conversation history in Gradio message format.

    Yields:
        tuple: A tuple containing an empty string (for input clearing) and the 
            updated history list.
    """
    # 1. Prepare initial state
    # history in Gradio 6.2.0 is a list of dicts: {"role": "user", "content": [...]}
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "context": "",
        "total_cost": 0.0,
        "recursion_count": 0,
        "budget_exceeded": False,
        "thread_id": THREAD_ID
    }
    
    # 2. Stream events from LangGraph
    history.append({"role": "user", "content": [{"type": "text", "text": message}]})
    yield "", history
    
    ai_response_content = ""
    history.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
    
    async for event in graph.astream(initial_state, config={"configurable": {"thread_id": THREAD_ID}}, stream_mode="values"):
        if "messages" in event and event["messages"]:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage):
                ai_response_content = last_msg.content
                # Update the last history item
                history[-1]["content"] = [{"type": "text", "text": ai_response_content}]
                yield "", history

# Gradio 6.2.0 Theme and Layout
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    block_border_width="1px",
    block_shadow="0 4px 6px -1px rgb(0 0 0 / 0.1)",
)

with gr.Blocks(theme=custom_theme, title="Personal Agent v1") as demo:
    gr.Markdown("# 🧠 Personal Context-Aware Agent")
    gr.Markdown("Persistent reasoning with LangGraph & Zep Memory")
    
    chatbot = gr.Chatbot(
        label="Reasoning Chain",
        type="messages", # Use the new messages format
        show_label=False,
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Agent"),
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Message",
            placeholder="What would you like to research today?",
            scale=4,
            show_label=False,
            container=False
        )
        submit = gr.Button("Execute", variant="primary", scale=1)
        stop = gr.Button("Stop", variant="stop", scale=1)

    # Event handlers
    submit_event = msg.submit(chat_interaction, inputs=[msg, chatbot], outputs=[msg, chatbot])
    click_event = submit.click(chat_interaction, inputs=[msg, chatbot], outputs=[msg, chatbot])
    
    # Non-blocking stop command
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, click_event])

if __name__ == "__main__":
    demo.launch()
