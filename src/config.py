"""
Configuration module for the Resume Scorer application.
Contains settings for optimizing memory usage and performance.
"""

import os
import psutil
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = Path(os.path.join(BASE_DIR, "src", "data"))
CACHE_DIR = Path(os.path.join(BASE_DIR, "model_cache"))

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Detect memory constraints
def is_memory_constrained():
    """Check if system is memory-constrained"""
    try:
        mem = psutil.virtual_memory()
        # Consider memory constrained if available memory is less than 2GB or 
        # if memory usage is above 75%
        return (mem.available < 2 * 1024 * 1024 * 1024) or (mem.percent > 75)
    except:
        # If can't determine, assume constraints if explicitly set
        return os.environ.get('MEMORY_CONSTRAINED', '0') == '1'

# Cache settings
ENABLE_DB_CACHE = True
DB_CACHE_PATH = os.path.join(DATA_DIR, "cache.db")
EMBEDDINGS_CACHE_SIZE = 5000  # Maximum number of embeddings to cache

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers")
SPACY_MODEL = "en_core_web_sm"

# Enable quantization based on memory constraints
USE_QUANTIZED_MODEL = is_memory_constrained()

# Task-specific model configurations
# Different models for different tasks to improve efficiency
TASK_SPECIFIC_MODELS = {
    # Main embedding model for general matching
    "general": {
        "model_name": "all-MiniLM-L6-v2",
        "quantize": USE_QUANTIZED_MODEL,  # Now dynamic based on memory
        "dimension": 384
    },
    # Specialized model for skills extraction (more focused)
    "skills": {
        "model_name": "paraphrase-MiniLM-L3-v2",  # Smaller model (only 17MB)
        "quantize": True,
        "dimension": 384
    },
    # Education and certification extractor model
    "education": {
        "model_name": None,  # Use SpaCy's NER by default (no embedding model)
        "use_rules": True,
        "ner_labels": ["ORG", "DATE"]
    }
}

# Enable task-specific models
USE_TASK_SPECIFIC_MODELS = False  # Set to True to enable multi-model approach

# Memory optimization
BATCH_SIZE = 16  # Batch size for encoding texts
MAX_TEXT_LENGTH = 100000  # Maximum text length to process
ENABLE_TORCH_MEMORY_EFFICIENT_LOAD = True
ENABLE_MODEL_OFFLOADING = True  # Offload model when not in use

# Resource monitoring
def check_memory_usage():
    """Check if we're approaching memory limits"""
    try:
        mem = psutil.virtual_memory()
        return mem.percent > 90
    except:
        return False

# Dynamic resource management
ENABLE_DYNAMIC_RESOURCE_MANAGEMENT = True
MEM_CHECK_INTERVAL = 100  # Check memory every N requests

# File processing
MAX_FILE_SIZE = 10 * 1024 * 1024

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