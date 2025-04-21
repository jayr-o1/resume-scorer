#!/usr/bin/env python3
"""
Script to pre-download all required models and cache them for Render deployment.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set cache directories
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_dir, "model_cache")
os.environ["HF_HOME"] = os.path.join(base_dir, "model_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(base_dir, "model_cache")

def download_sentence_transformer():
    """Download and cache the sentence transformer model"""
    logger.info("Downloading sentence transformer model...")
    model_name = "all-MiniLM-L6-v2"
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(base_dir, "model_cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model to cache it
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    logger.info(f"Model {model_name} downloaded and cached")
    
    # Verify the model works
    embeddings = model.encode(["Test sentence to verify the model works"])
    logger.info(f"Model verification: Embedding shape {embeddings.shape}")
    
    return model_name

def download_spacy_model():
    """Download and cache spaCy model"""
    logger.info("Downloading spaCy model...")
    model_name = "en_core_web_sm"
    
    # Download spaCy model if not already downloaded
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    
    # Load model to verify
    nlp = spacy.load(model_name)
    logger.info(f"SpaCy model {model_name} downloaded and ready")
    
    return model_name

def initialize_db_cache():
    """Initialize the SQLite database for caching"""
    from src.utils.analyzer import init_db_cache
    
    logger.info("Initializing database cache...")
    init_db_cache()
    logger.info("Database cache initialized")

def main():
    """Main function to download all models"""
    logger.info("Starting model download process...")
    
    # Create main cache directory
    os.makedirs(os.path.join(base_dir, "model_cache"), exist_ok=True)
    
    # Download and cache models
    st_model = download_sentence_transformer()
    spacy_model = download_spacy_model()
    
    # Initialize database cache
    initialize_db_cache()
    
    logger.info("All models and caches prepared successfully")
    logger.info(f"Cached models: {st_model}, {spacy_model}")
    logger.info(f"Cache directory: {os.path.join(base_dir, 'model_cache')}")

if __name__ == "__main__":
    main() 