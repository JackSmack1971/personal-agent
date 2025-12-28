---

# GRADIO 6.2.0 TECHNICAL RULESET

**Version**: Gradio 6.2.0 (Released: December 19, 2025)  
**Python Compatibility**: Python 3.8+  
**Architecture**: FastAPI backend, Svelte frontend  
**Documentation Sources**: Official Gradio Docs, GitHub Security Advisories, Community Best Practices

---

## 1. VERSION AWARENESS AND MIGRATION COMPLIANCE

### MANDATORY: Version Declaration
```python
# CORRECT: Always pin exact Gradio version in requirements
gradio==6.2.0

# ANTI-PATTERN: Avoid version ranges in production
gradio>=6.0.0  # ❌ May introduce breaking changes
```

### CRITICAL: Gradio 5.x → 6.x Breaking Changes

**Parameter Relocation (Partially Reversed in 6.1.0)**
```python
# ✅ CORRECT (6.1.0+): Both patterns valid
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".custom { color: red; }"
) as demo:
    pass
demo.launch()

# ✅ ALSO CORRECT (6.0+)
with gr.Blocks() as demo:
    pass
demo.launch(
    theme=gr.themes.Soft(),
    css=".custom { color: red; }"
)

# ❌ ANTI-PATTERN: Gradio 5.x only
with gr.Blocks(show_api=True) as demo:  # Parameter removed
    pass
```

**API Visibility Refactoring**
```python
# ✅ CORRECT (Gradio 6.x)
btn.click(
    fn=process,
    inputs=input_text,
    outputs=output,
    api_visibility="public"  # "public" | "private" | "undocumented"
)

# ❌ ANTI-PATTERN: Gradio 5.x syntax
btn.click(
    fn=process,
    inputs=input_text,
    outputs=output,
    show_api=True,  # Removed in 6.x
    api_name=False  # Now use api_visibility="private"
)
```

**Chatbot Message Format Standardization**
```python
# ✅ CORRECT (Gradio 6.x): Dictionary format with role/content
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}]
    }
]

# ✅ CORRECT: Multimodal content
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image": "path/to/image.jpg"}
        ]
    }
]

# ❌ ANTI-PATTERN: Tuple format removed in 6.x
messages = [
    ["user message", "assistant response"]  # No longer supported
]
```

---

## 2. COMPONENT CONFIGURATION PATTERNS

### Button Consolidation (Unified `buttons` Parameter)

```python
# ✅ CORRECT: Unified button control
audio = gr.Audio(
    buttons=["download", "share"]  # String list for built-in buttons
)

textbox = gr.Textbox(
    buttons=["copy", "submit"]
)

# ✅ CORRECT: Custom buttons (6.2.0+)
refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
custom_textbox = gr.Textbox(
    buttons=["copy", refresh_btn]  # Mix built-in and custom
)

refresh_btn.click(lambda: "Refreshed!", outputs=custom_textbox)

# ❌ ANTI-PATTERN: Gradio 5.x individual parameters
audio = gr.Audio(
    show_download_button=True,  # Removed
    show_share_button=True      # Removed
)
```

### Dataframe Structural Reorganization

```python
# ✅ CORRECT (Gradio 6.x)
df = gr.Dataframe(
    row_count=5,           # Initial rows
    row_limits=(1, 10),    # (min, max) constraints
    col_count=3,           # Initial columns
    col_limits=(1, 5),     # (min, max) constraints
    buttons=["copy", "fullscreen"]
)

# ❌ ANTI-PATTERN: Gradio 5.x tuple format
df = gr.Dataframe(
    row_count=(5, "dynamic"),  # Removed
    col_count=(3, "fixed")     # Removed
)
```

### Cache Examples Strategy

```python
# ✅ CORRECT (Gradio 6.x)
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    examples=["Hello", "World"],
    cache_examples=True,      # Boolean only
    cache_mode="lazy"         # "eager" | "lazy"
)

# ❌ ANTI-PATTERN: Gradio 5.x mixed parameter
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    examples=["Hello", "World"],
    cache_examples="lazy"  # String no longer accepted
)
```

---

## 3. API ENDPOINT CONFIGURATION

### API Visibility Control

```python
# ✅ CORRECT: Explicit visibility control
@gr.render
def api_demo():
    # Public: Shown in docs, callable
    gr.Button("Public").click(
        fn=public_fn,
        inputs=None,
        outputs=None,
        api_visibility="public"  # Default
    )
    
    # Private: Hidden, not callable
    gr.Button("Private").click(
        fn=internal_fn,
        inputs=None,
        outputs=None,
        api_visibility="private"
    )
    
    # Undocumented: Hidden but callable
    gr.Button("Hidden").click(
        fn=hidden_fn,
        inputs=None,
        outputs=None,
        api_visibility="undocumented"
    )
```

### Custom API Endpoints

```python
# ✅ CORRECT: Add custom API route
import gradio as gr
from gradio_client import Client

def custom_logic(a: int, b: int, c: list[str]) -> tuple[int, str]:
    """Process inputs with custom logic."""
    return a + b, "".join(c[a:b])

with gr.Blocks() as demo:
    # Add non-UI API endpoint
    gr.api(
        fn=custom_logic,
        api_name="process_data",
        api_description="Custom data processing endpoint"
    )
    
    # Regular UI components
    gr.Textbox()

demo.launch()

# Client usage
client = Client("http://localhost:7860")
result = client.predict(
    a=3,
    b=5,
    c=["x", "y", "z", "a", "b"],
    api_name="/process_data"
)
```

### API Endpoint Naming

```python
# ✅ CORRECT: Explicit API naming
demo = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    api_name="predict"  # Explicitly set for compatibility
)

# ⚠️ BEHAVIOR CHANGE: Function name becomes default in 6.x
demo = gr.Interface(
    fn=generate_text,  # Creates /generate_text endpoint
    inputs="text",
    outputs="text"
)
# If external services expect /predict, explicitly set api_name
```

---

## 4. STATE MANAGEMENT

### Session State Patterns

```python
# ✅ CORRECT: Per-user session state
import gradio as gr

with gr.Blocks() as demo:
    # State persists per session, not shared between users
    conversation_history = gr.State([])
    
    msg = gr.Textbox(label="Message")
    chatbot = gr.Chatbot(label="Chat History")
    
    def respond(message, history):
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": message}]
        })
        # Process and add response
        response = f"Echo: {message}"
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        return "", history
    
    msg.submit(
        fn=respond,
        inputs=[msg, conversation_history],
        outputs=[msg, conversation_history]
    )
```

### Non-Deepcopyable Objects Management

```python
# ✅ CORRECT: Manual session tracking for complex objects
import gradio as gr
from threading import Lock

instances = {}  # Global dict for session tracking

class NonDeepCopyable:
    def __init__(self):
        self.counter = 0
        self.lock = Lock()
    
    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter

def initialize_instance(request: gr.Request):
    instances[request.session_hash] = NonDeepCopyable()
    return "Session initialized"

def increment_counter(request: gr.Request):
    if request.session_hash in instances:
        return instances[request.session_hash].increment()
    return "Error: Session not initialized"

def cleanup_instance(request: gr.Request):
    if request.session_hash in instances:
        del instances[request.session_hash]

with gr.Blocks() as demo:
    output = gr.Textbox(label="Status")
    counter_display = gr.Number(label="Counter")
    increment_btn = gr.Button("Increment")
    
    increment_btn.click(increment_counter, outputs=counter_display)
    demo.load(initialize_instance, outputs=output)
    demo.unload(cleanup_instance)

demo.launch()
```

---

## 5. FILE HANDLING

### File Upload Security

```python
# ✅ CORRECT: Type-safe file handling
import gradio as gr
from pathlib import Path
import mimetypes

ALLOWED_EXTENSIONS = {".txt", ".csv", ".json", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def process_file(file):
    if file is None:
        raise gr.Error("No file uploaded")
    
    # Type: file can be str (filepath) or bytes depending on type parameter
    if isinstance(file, bytes):
        # Handle bytes
        if len(file) > MAX_FILE_SIZE:
            raise gr.Error("File exceeds 10MB limit")
        return f"Processed {len(file)} bytes"
    
    # Handle filepath
    path = Path(file)
    
    # Validate extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise gr.Error(
            f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Validate size
    if path.stat().st_size > MAX_FILE_SIZE:
        raise gr.Error("File exceeds 10MB limit")
    
    # Validate MIME type
    mime_type, _ = mimetypes.guess_type(file)
    if not mime_type:
        raise gr.Error("Unable to determine file type")
    
    return f"Processed: {path.name}"

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(
        label="Upload File",
        file_types=[".txt", ".csv", ".json", ".pdf"],
        type="filepath"  # "filepath" | "bytes" | "auto"
    ),
    outputs="text"
)

demo.launch()
```

### Multiple File Handling

```python
# ✅ CORRECT: Multiple file upload
import gradio as gr

def process_multiple_files(files):
    """
    Args:
        files: list[str] | list[bytes] depending on type parameter
    """
    if not files:
        raise gr.Error("No files uploaded")
    
    results = []
    for idx, file in enumerate(files):
        if isinstance(file, str):
            results.append(f"File {idx + 1}: {Path(file).name}")
        else:
            results.append(f"File {idx + 1}: {len(file)} bytes")
    
    return "\n".join(results)

demo = gr.Interface(
    fn=process_multiple_files,
    inputs=gr.File(
        label="Upload Multiple Files",
        file_count="multiple",  # "single" | "multiple" | "directory"
        file_types=[".txt", ".pdf"],
        type="filepath"
    ),
    outputs="text"
)

demo.launch()
```

---

## 6. SECURITY BEST PRACTICES

### CRITICAL: CORS Vulnerability Mitigation (CVE-2024-47084)

```python
# ✅ CORRECT: Secure deployment configuration
import gradio as gr

demo = gr.Blocks()

# NEVER deploy with default settings in production
demo.launch(
    server_name="0.0.0.0",  # Only if needed
    server_port=7860,
    auth=("admin", "secure_password"),  # Always use authentication
    ssl_keyfile="path/to/key.pem",      # Use HTTPS
    ssl_certfile="path/to/cert.pem",
    allowed_paths=["/safe/data/dir"],    # Restrict file access
    blocked_paths=["/etc", "/home"],     # Block sensitive paths
    state_session_capacity=1000          # Limit memory usage
)

# ❌ ANTI-PATTERN: Insecure production deployment
demo.launch(
    server_name="0.0.0.0",
    share=True,  # Creates public tunnel - dangerous!
    auth=None    # No authentication
)
```

### Authentication Patterns

```python
# ✅ CORRECT: Basic authentication
import gradio as gr
import os

def check_auth(username, password):
    """Custom authentication logic."""
    valid_users = {
        os.getenv("ADMIN_USER"): os.getenv("ADMIN_PASS")
    }
    return valid_users.get(username) == password

demo = gr.Blocks()

demo.launch(
    auth=check_auth,  # Function or tuple
    auth_message="Please enter credentials"
)

# ✅ BETTER: Use SSO/OAuth for production
# Integrate with Descope, Auth0, or similar services
```

### Input Validation

```python
# ✅ CORRECT: Comprehensive input validation
import gradio as gr
import re

def validate_email(email: str):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return gr.validate(
            passed=False,
            message="Invalid email format"
        )
    return gr.validate(passed=True)

def validate_age(age: int):
    """Validate age range."""
    if age < 0 or age > 150:
        return gr.validate(
            passed=False,
            message="Age must be between 0 and 150"
        )
    return gr.validate(passed=True)

def process_form(email: str, age: int):
    return f"Processing: {email}, Age: {age}"

with gr.Blocks() as demo:
    email_input = gr.Textbox(label="Email")
    age_input = gr.Number(label="Age")
    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Result")
    
    submit_btn.click(
        fn=process_form,
        inputs=[email_input, age_input],
        outputs=output,
        validator=lambda email, age: [
            validate_email(email),
            validate_age(age)
        ]
    )

demo.launch()
```

---

## 7. EVENT HANDLING

### Event Listener Patterns

```python
# ✅ CORRECT: Comprehensive event handling
import gradio as gr

def process_input(text):
    return f"Processed: {text}"

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input")
    output_text = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")
    
    # Method 1: Explicit .click()
    submit_btn.click(
        fn=process_input,
        inputs=input_text,
        outputs=output_text,
        api_name="process",
        api_visibility="public",
        queue=True,
        concurrency_limit=1
    )
    
    # Method 2: Decorator pattern
    @submit_btn.click(inputs=input_text, outputs=output_text)
    def decorated_handler(text):
        return f"Decorated: {text}"
    
    # Method 3: Component-specific events
    input_text.change(  # Fires on any change
        fn=lambda x: f"Changed to: {x}",
        inputs=input_text,
        outputs=output_text
    )
    
    input_text.submit(  # Fires on Enter key
        fn=process_input,
        inputs=input_text,
        outputs=output_text
    )

demo.launch()
```

### Concurrency Control

```python
# ✅ CORRECT: Concurrency management
import gradio as gr
import time

def slow_process(data):
    time.sleep(5)
    return f"Processed: {data}"

def fast_process(data):
    return f"Fast: {data}"

with gr.Blocks() as demo:
    input_data = gr.Textbox()
    output_slow = gr.Textbox(label="Slow Output")
    output_fast = gr.Textbox(label="Fast Output")
    
    slow_btn = gr.Button("Slow Process")
    fast_btn = gr.Button("Fast Process")
    
    # Limit slow operations
    slow_btn.click(
        fn=slow_process,
        inputs=input_data,
        outputs=output_slow,
        concurrency_limit=2,  # Max 2 simultaneous slow operations
        concurrency_id="slow_ops"
    )
    
    # No limit on fast operations
    fast_btn.click(
        fn=fast_process,
        inputs=input_data,
        outputs=output_fast,
        concurrency_limit=None  # Unlimited
    )

# Configure global queue settings
demo.queue(
    default_concurrency_limit=1,  # Default for all events
    max_size=100  # Max queue size
)

demo.launch()
```

---

## 8. THEMING AND CUSTOMIZATION

### Custom Theme Creation

```python
# ✅ CORRECT: Production-grade custom theme
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class ProductionTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.cyan,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str = (
            fonts.GoogleFont("Roboto Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        
        # Fine-grained CSS variable overrides
        super().set(
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
            button_primary_border_color="*primary_500",
            button_secondary_background_fill="*secondary_500",
            block_shadow="*shadow_drop",
            block_border_width="1px",
            input_border_width="1px",
            input_shadow_focus="*shadow_inset"
        )

theme = ProductionTheme()

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## Custom Themed Application")
    gr.Textbox(label="Input")
    gr.Button("Submit", variant="primary")

demo.launch()
```

### CSS and JavaScript Injection

```python
# ✅ CORRECT: Custom CSS and JS
import gradio as gr
from pathlib import Path

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

.custom-button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
"""

custom_js = """
function() {
    console.log('Gradio app initialized');
    
    // Add analytics
    if (typeof gtag !== 'undefined') {
        gtag('event', 'app_load', {
            'event_category': 'engagement',
            'event_label': 'gradio_app'
        });
    }
}
"""

custom_head = """
<meta name="description" content="AI-powered application">
<meta name="keywords" content="AI, machine learning, Gradio">
<meta property="og:title" content="My Gradio App">
<meta property="og:description" content="Production ML application">
<link rel="icon" type="image/x-icon" href="/file=favicon.ico">
"""

with gr.Blocks(
    css=custom_css,
    js=custom_js,
    head=custom_head,
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("## Styled Application")
    gr.Button("Custom Styled Button", elem_classes="custom-button")

demo.launch()
```

---

## 9. ERROR HANDLING

### User-Facing Error Messages

```python
# ✅ CORRECT: User-friendly error handling
import gradio as gr

def process_data(value):
    try:
        # Validation
        if not value:
            raise gr.Error(
                "Input cannot be empty!",
                duration=5  # Display for 5 seconds
            )
        
        # Processing
        result = int(value)
        
        if result < 0:
            raise gr.Error(
                "Value must be positive!",
                duration=10,
                visible=True  # Show in UI (default)
            )
        
        return f"Result: {result * 2}"
        
    except ValueError:
        raise gr.Error(
            "Please enter a valid number!",
            duration=None  # Display until user closes
        )
    except Exception as e:
        # Log internal errors, show generic message
        print(f"Internal error: {e}")
        raise gr.Error(
            "An unexpected error occurred. Please try again.",
            duration=10
        )

demo = gr.Interface(
    fn=process_data,
    inputs=gr.Textbox(label="Enter a number"),
    outputs=gr.Textbox(label="Result")
)

demo.launch()
```

### Info and Warning Messages

```python
# ✅ CORRECT: Using Info and Warning
import gradio as gr

def process_with_warnings(data):
    if len(data) > 100:
        gr.Warning(
            "Input exceeds recommended length. Processing may be slow.",
            duration=8
        )
    
    if not data.strip():
        gr.Info(
            "Empty input detected. Using default value.",
            duration=5
        )
        data = "default"
    
    return f"Processed: {data}"

demo = gr.Interface(
    fn=process_with_warnings,
    inputs="text",
    outputs="text"
)

demo.launch()
```

---

## 10. PERFORMANCE OPTIMIZATION

### Queuing and Batching

```python
# ✅ CORRECT: Efficient batch processing
import gradio as gr

def batch_predict(inputs: list[str]) -> tuple[list[str], list[float]]:
    """
    Process batch of inputs efficiently.
    
    Args:
        inputs: List of equal-length input lists
    
    Returns:
        Tuple of output lists (must match input length)
    """
    predictions = [f"Predicted: {inp}" for inp in inputs]
    confidences = [0.95] * len(inputs)
    
    return predictions, confidences

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input")
    output_pred = gr.Textbox(label="Prediction")
    output_conf = gr.Number(label="Confidence")
    
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(
        fn=batch_predict,
        inputs=input_text,
        outputs=[output_pred, output_conf],
        batch=True,              # Enable batching
        max_batch_size=8,        # Process up to 8 at once
        queue=True               # Required for batching
    )

demo.queue(
    default_concurrency_limit=1,
    api_open=True
)

demo.launch()
```

### Caching Strategies

```python
# ✅ CORRECT: Efficient caching
import gradio as gr
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def expensive_computation(input_str: str) -> str:
    """Cache results of expensive operations."""
    # Simulate expensive computation
    import time
    time.sleep(2)
    return f"Computed: {input_str.upper()}"

def cached_predict(input_data):
    # Create cache key
    cache_key = hashlib.md5(input_data.encode()).hexdigest()[:16]
    return expensive_computation(cache_key)

# Example caching configuration
demo = gr.Interface(
    fn=cached_predict,
    inputs="text",
    outputs="text",
    examples=[
        ["example 1"],
        ["example 2"],
        ["example 3"]
    ],
    cache_examples=True,     # Cache example outputs
    cache_mode="lazy"        # "eager" | "lazy"
)

demo.launch()
```

---

## 11. DEPLOYMENT CONFIGURATIONS

### Production Launch Settings

```python
# ✅ CORRECT: Production deployment
import gradio as gr
import os
from pathlib import Path

demo = gr.Blocks()

# Production configuration
demo.launch(
    # Server configuration
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", 7860)),
    
    # Security
    auth=(os.getenv("AUTH_USER"), os.getenv("AUTH_PASS")),
    ssl_keyfile=os.getenv("SSL_KEY_PATH"),
    ssl_certfile=os.getenv("SSL_CERT_PATH"),
    
    # Resource limits
    max_threads=40,
    state_session_capacity=1000,
    
    # API configuration
    footer_links=[
        ("Documentation", "https://docs.example.com"),
        ("Support", "https://support.example.com")
    ],
    
    # Performance
    queue=True,
    max_size=100,
    
    # Development features (disable in production)
    show_error=False,  # Don't show detailed errors to users
    debug=False,       # Disable debug mode
    
    # Monitoring
    analytics_enabled=True,
    
    # File handling
    allowed_paths=[Path("/app/data")],
    blocked_paths=[Path("/etc"), Path("/root")]
)
```

### Docker Deployment

```dockerfile
# ✅ CORRECT: Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Non-root user
RUN useradd -m -u 1000 gradio && \
    chown -R gradio:gradio /app
USER gradio

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]
```

```python
# app.py for Docker
import gradio as gr
import os

demo = gr.Blocks()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=(os.getenv("AUTH_USER"), os.getenv("AUTH_PASS")),
        show_error=False,
        quiet=False  # Show startup logs
    )
```

---

## 12. TESTING PATTERNS

### Pytest Integration

```python
# ✅ CORRECT: Comprehensive testing
import pytest
import gradio as gr
from gradio_client import Client

@pytest.fixture
def demo_app():
    """Fixture providing demo app."""
    with gr.Blocks() as demo:
        input_box = gr.Textbox()
        output_box = gr.Textbox()
        
        @gr.render(inputs=input_box, outputs=output_box)
        def echo(text):
            return f"Echo: {text}"
    
    return demo

@pytest.fixture
def client(demo_app):
    """Fixture providing test client."""
    _, local_url, _ = demo_app.launch(prevent_thread_lock=True)
    client = Client(local_url)
    yield client
    demo_app.close()

def test_api_endpoint(client):
    """Test API endpoint functionality."""
    result = client.predict("test input", api_name="/echo")
    assert result == "Echo: test input"

def test_file_upload(client):
    """Test file upload handling."""
    from gradio_client import file
    
    result = client.predict(
        file("test_file.txt"),
        api_name="/process_file"
    )
    assert "processed" in result.lower()

def test_error_handling(demo_app):
    """Test error handling."""
    def failing_fn(x):
        if x == "error":
            raise gr.Error("Test error")
        return x
    
    with pytest.raises(gr.Error):
        failing_fn("error")

def test_validation():
    """Test input validation."""
    def validate_positive(num):
        if num <= 0:
            return gr.validate(
                passed=False,
                message="Must be positive"
            )
        return gr.validate(passed=True)
    
    result = validate_positive(-1)
    assert not result.passed
    assert "positive" in result.message.lower()
```

---

## 13. KNOWN ISSUES AND MITIGATIONS

### CRITICAL: CORS Vulnerability (CVE-2024-47084)

**Issue**: CORS origin validation not performed when request has cookie  
**Impact**: Allows attacker websites to make unauthorized requests  
**Affected Versions**: Gradio < 6.0.2  
**Mitigation**:

```python
# ✅ CORRECT: Upgrade to 6.0.2+ and enable authentication
import gradio as gr

demo = gr.Blocks()

demo.launch(
    auth=("admin", "secure_password"),  # REQUIRED in production
    ssl_certfile="cert.pem",             # Use HTTPS
    ssl_keyfile="key.pem",
    allowed_paths=["/safe/data"]         # Restrict file access
)

# ❌ VULNERABLE: No authentication with public access
demo.launch(
    server_name="0.0.0.0",
    share=True,  # Creates public tunnel
    auth=None    # No authentication
)
```

### Migration from 5.x Deprecation Warnings

```python
# ✅ CORRECT: Use Gradio 5.50 for migration audit
# Install: pip install gradio==5.50.0
# Run application and check console for deprecation warnings
# Example warnings:
# - "show_api parameter is deprecated, use api_visibility"
# - "api_name=False is deprecated, use api_visibility='private'"
# - "Tuple format for chatbot messages is deprecated"
```

### Component Parameter Consolidation

```python
# ✅ CORRECT: Updated component parameters
# Audio/Video components
audio = gr.Audio(
    buttons=["download"],  # Not show_download_button
    webcam_options=gr.WebcamOptions(mirror=True)  # Not mirror_webcam
)

# HTML component padding
html = gr.HTML(
    "<div>Content</div>",
    padding=True  # Default changed to False in 6.x
)

# ImageEditor canvas
editor = gr.ImageEditor(
    canvas_size=(512, 512)  # Not crop_size
)
```

---

## 14. COMMON ANTI-PATTERNS TO AVOID

```python
# ❌ ANTI-PATTERN: Blocking the main thread
def slow_operation():
    import time
    time.sleep(60)  # Blocks entire server
    return "Done"

# ✅ CORRECT: Use async or threading
import asyncio

async def async_operation():
    await asyncio.sleep(60)
    return "Done"

# ❌ ANTI-PATTERN: Unvalidated file uploads
def process_file(file):
    exec(open(file).read())  # SEVERE SECURITY RISK

# ✅ CORRECT: Validate all inputs
def safe_process_file(file):
    allowed_extensions = {".txt", ".csv"}
    if Path(file).suffix not in allowed_extensions:
        raise gr.Error("Invalid file type")
    
    # Process safely
    with open(file, 'r') as f:
        return f.read()

# ❌ ANTI-PATTERN: Storing secrets in code
demo.launch(auth=("admin", "password123"))

# ✅ CORRECT: Use environment variables
import os
demo.launch(auth=(os.getenv("USER"), os.getenv("PASS")))

# ❌ ANTI-PATTERN: Ignoring concurrency limits
btn.click(fn=expensive_fn, concurrency_limit=None)  # Can overwhelm server

# ✅ CORRECT: Set appropriate limits
btn.click(fn=expensive_fn, concurrency_limit=2)

# ❌ ANTI-PATTERN: Not handling errors
def process(data):
    return int(data) * 2  # ValueError if data is not numeric

# ✅ CORRECT: Explicit error handling
def safe_process(data):
    try:
        return int(data) * 2
    except ValueError:
        raise gr.Error("Please enter a valid number")
```

---

## 15. CHECKLIST: PRE-DEPLOYMENT VALIDATION

**Security**:
- [ ] Authentication enabled (`auth` parameter set)
- [ ] HTTPS configured (SSL certificates provided)
- [ ] File paths restricted (`allowed_paths`, `blocked_paths`)
- [ ] Input validation implemented for all user inputs
- [ ] Gradio version >= 6.0.2 (CORS vulnerability patched)

**Performance**:
- [ ] Queue enabled for long-running operations
- [ ] Concurrency limits configured appropriately
- [ ] Batch processing used where applicable
- [ ] Caching enabled for examples
- [ ] Resource limits set (`state_session_capacity`, `max_threads`)

**API Configuration**:
- [ ] API visibility explicitly set for all endpoints
- [ ] API names explicitly defined for backward compatibility
- [ ] API descriptions provided for documentation
- [ ] Custom API routes tested programmatically

**Error Handling**:
- [ ] User-facing error messages use `gr.Error()`
- [ ] Internal errors logged server-side
- [ ] Validation functions implemented for complex inputs
- [ ] Error duration and visibility configured

**Testing**:
- [ ] Unit tests cover all API endpoints
- [ ] Integration tests verify file uploads
- [ ] Error conditions tested explicitly
- [ ] Client library tested against deployed API

**Monitoring**:
- [ ] Logging configured appropriately
- [ ] Analytics enabled (if applicable)
- [ ] Health check endpoint configured
- [ ] Performance metrics tracked

---

## VERSION HISTORY

- **6.2.0** (2025-12-19): Custom buttons, `ty` library upgrade
- **6.1.0** (2025-11-XX): Restored Blocks constructor args, `playback_position`, `link_target`
- **6.0.2** (2025-10-XX): CORS vulnerability patch (CVE-2024-47084)
- **6.0.0** (2025-10-XX): Major version with breaking changes
- **5.50.0** (2025-XX-XX): Deprecation warnings for 6.x migration

---

## ADDITIONAL RESOURCES

- **Official Documentation**: https://www.gradio.app/docs
- **Migration Guide**: https://www.gradio.app/main/guides/gradio-6-migration-guide
- **Security Advisories**: https://github.com/gradio-app/gradio/security/advisories
- **GitHub Issues**: https://github.com/gradio-app/gradio/issues
- **Testing Guidelines**: https://github.com/gradio-app/gradio/tree/main/testing-guidelines

---

**END OF RULESET**  
**Last Updated**: December 27, 2025  
**Gradio Version**: 6.2.0  
**Compliance Level**: Production-Ready