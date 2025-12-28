import logging
import io
import sys

class GradioLogHandler(logging.Handler):
    """Custom logging handler that stores logs in a buffer for Gradio display."""
    def __init__(self):
        super().__init__()
        self.buffer = io.StringIO()

    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + "\n")

    def get_logs(self):
        return self.buffer.getvalue()

    def clear(self):
        self.buffer.truncate(0)
        self.buffer.seek(0)

# Create a global instance
gradio_handler = GradioLogHandler()
gradio_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

def setup_logging():
    """Configures the root logger to use the GradioLogHandler."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if setup is called multiple times
    if not any(isinstance(h, GradioLogHandler) for h in logger.handlers):
        logger.addHandler(gradio_handler)
    
    # Also redirect stdout for standard print statements
    # WARNING: This is a bit aggressive, but useful for capturing 'print' from orchestrator nodes
    class StdoutRedirector:
        def write(self, data):
            if data.strip():
                gradio_handler.buffer.write(f"STDOUT: {data.strip()}\n")
            sys.__stdout__.write(data)
        def flush(self):
            sys.__stdout__.flush()

    # Uncomment below to capture ALL prints - use with caution
    # sys.stdout = StdoutRedirector()
