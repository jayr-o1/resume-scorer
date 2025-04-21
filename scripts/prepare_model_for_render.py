#!/usr/bin/env python3
"""
Script to prepare sentence transformer models for offline use on Render.
This creates a completely self-contained model directory that can be used without network access.
Run this script before deployment to ensure models work in offline mode.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import tempfile
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Model settings
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(parent_dir, "model_cache")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers")

def ensure_dirs_exist():
    """Ensure all required directories exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    logger.info(f"Created cache directories: {CACHE_DIR}, {MODEL_CACHE_DIR}")

def download_and_save_model():
    """Download the model and save it for offline use"""
    try:
        # Import here to avoid early errors
        from sentence_transformers import SentenceTransformer
        import torch
        
        logger.info(f"Downloading model: {MODEL_NAME}")
        
        # Load the model (this will download it if not already cached)
        model = SentenceTransformer(MODEL_NAME)
        
        # Create model directory name (replacing slashes)
        model_dir_name = MODEL_NAME.replace('/', '_')
        offline_model_path = os.path.join(MODEL_CACHE_DIR, model_dir_name)
        
        # Create a clean directory
        if os.path.exists(offline_model_path):
            logger.info(f"Removing existing model directory: {offline_model_path}")
            shutil.rmtree(offline_model_path)
        
        os.makedirs(offline_model_path, exist_ok=True)
        
        # Save the model to the offline directory
        logger.info(f"Saving model to: {offline_model_path}")
        model.save(offline_model_path)
        
        # Create a simple test file to verify it works
        test_sentences = ["This is a test sentence to encode.", "This is another sentence to test the model."]
        embeddings = model.encode(test_sentences)
        
        # Save a sample embedding for verification
        with open(os.path.join(offline_model_path, "test_embedding.json"), "w") as f:
            json.dump({
                "test_sentences": test_sentences,
                "embedding_shape": embeddings.shape,
                "embedding_sample": embeddings[0][:5].tolist()  # Save just a small sample
            }, f)
        
        logger.info(f"Successfully saved model to {offline_model_path}")
        logger.info(f"Model files: {os.listdir(offline_model_path)}")
        return offline_model_path
        
    except Exception as e:
        logger.error(f"Error downloading and saving model: {e}")
        raise

def test_offline_model(model_path):
    """Test the saved model in offline mode"""
    try:
        # Set offline environment variables
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Try to load the model from the saved path
        from sentence_transformers import SentenceTransformer
        logger.info(f"Testing offline model from: {model_path}")
        
        # Load the model in offline mode
        model = SentenceTransformer(model_path)
        
        # Test encoding
        test_text = "Testing the offline model functionality"
        embeddings = model.encode([test_text])
        
        logger.info(f"Successfully tested offline model. Embedding shape: {embeddings.shape}")
        return True
    except Exception as e:
        logger.error(f"Error testing offline model: {e}")
        return False

def prepare_model_files():
    """Prepare and organize model files for offline use"""
    try:
        # Ensure all config files and module configs are in place
        logger.info("Ensuring all necessary model files are in place")
        
        model_dir_name = MODEL_NAME.replace('/', '_')
        offline_model_path = os.path.join(MODEL_CACHE_DIR, model_dir_name)
        
        # Create additional files needed for offline mode
        config_path = os.path.join(offline_model_path, "config.json")
        if not os.path.exists(config_path):
            logger.info(f"Creating default config file at {config_path}")
            with open(config_path, "w") as f:
                json.dump({
                    "model_type": "sentence-transformer",
                    "model_name": MODEL_NAME,
                    "dimension": 384,
                    "pooling_mode": "mean"
                }, f)
        
        # Create a README file for documentation
        with open(os.path.join(offline_model_path, "README.md"), "w") as f:
            f.write(f"""# Offline Sentence Transformer Model

This is an offline-ready version of the {MODEL_NAME} model for use in environments without internet access.
Created with prepare_model_for_render.py script.

## Usage

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{offline_model_path}")
embeddings = model.encode(["Your text here"])
```
""")
        
        logger.info(f"Model preparation complete: {offline_model_path}")
        return offline_model_path
    
    except Exception as e:
        logger.error(f"Error preparing model files: {e}")
        raise

def main():
    """Main function to prepare the model for offline use"""
    logger.info("Starting model preparation for offline use on Render")
    
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Download and save model
    model_path = download_and_save_model()
    
    # Prepare model files
    prepare_model_files()
    
    # Test the model in offline mode
    if test_offline_model(model_path):
        logger.info("✅ Model preparation successful. Ready for offline use on Render.")
    else:
        logger.error("❌ Model preparation failed. See logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 