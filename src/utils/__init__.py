"""Utilities for resume analysis"""

import os
from pathlib import Path

# Create model cache directory
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True) 