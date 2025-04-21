"""
Configuration module for the Resume Scorer application.
Contains settings for optimizing memory usage and performance.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = Path(os.path.join(BASE_DIR, "src", "data"))
CACHE_DIR = Path(os.path.join(BASE_DIR, "model_cache"))

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Cache settings
ENABLE_DB_CACHE = True
DB_CACHE_PATH = os.path.join(DATA_DIR, "cache.db")
EMBEDDINGS_CACHE_SIZE = 5000  # Maximum number of embeddings to cache

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers")
SPACY_MODEL = "en_core_web_sm"

# Memory optimization
BATCH_SIZE = 16  # Batch size for encoding texts
MAX_TEXT_LENGTH = 100000  # Maximum text length to process
ENABLE_TORCH_MEMORY_EFFICIENT_LOAD = True
ENABLE_MODEL_OFFLOADING = True  # Offload model when not in use

# File processing
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB max file size

# Performance settings
DEFAULT_WORKERS = 2  # Default number of workers for batch processing
MAX_WORKERS = 4  # Maximum number of workers

# Environment-specific settings
IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development").lower() == "production"

# If running on Render, optimize for their environment
ON_RENDER = "RENDER" in os.environ
if ON_RENDER:
    # More aggressive optimizations for Render free tier
    BATCH_SIZE = 4  # Further reduce batch size
    DEFAULT_WORKERS = 1
    MAX_WORKERS = 1  # Limit to single worker on Render free tier
    MAX_TEXT_LENGTH = 50000  # Reduce max text length to process
    EMBEDDINGS_CACHE_SIZE = 2000  # Reduce cache size on Render
    # Use pre-downloaded models
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
# Memory monitoring
ENABLE_MEMORY_MONITORING = True
MEMORY_MONITORING_INTERVAL = 60 if ON_RENDER else 300  # Check more frequently on Render

# Function to get environment variable with fallback
def get_env(name, default=None):
    """Get environment variable with fallback to default"""
    return os.environ.get(name, default)

# Export environment variables for better integration
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)

# Additional optimizations for Python memory management
if ON_RENDER:
    os.environ["PYTHONMALLOC"] = "malloc"
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
    # Disable JIT compilation to save memory
    os.environ["PYTORCH_JIT"] = "0" 