import logging
from colorama import init

# Configure Logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)

# Configuration
base_experiment_dir = "results"
base_data_dir = "data"

# Initialize Colorama
init(autoreset=True)
