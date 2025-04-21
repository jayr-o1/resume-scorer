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

def download_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """
    Download and cache the sentence transformer model
    
    Args:
        model_name: Name of the model to download (default: all-MiniLM-L6-v2)
        
    Returns:
        Name of the downloaded model
    """
    logger.info(f"Downloading sentence transformer model: {model_name}")
    
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

def download_spacy_model(model_name="en_core_web_sm"):
    """
    Download and cache the spaCy model
    
    Args:
        model_name: Name of the spaCy model to download (default: en_core_web_sm)
        
    Returns:
        Name of the downloaded model
    """
    logger.info(f"Downloading spaCy model: {model_name}")
    
    try:
        # Check if model is already installed
        try:
            nlp = spacy.load(model_name)
            logger.info(f"SpaCy model {model_name} already installed")
        except OSError:
            # Download the model
            spacy.cli.download(model_name)
            logger.info(f"SpaCy model {model_name} downloaded")
            
        # Verify the model works
        nlp = spacy.load(model_name)
        doc = nlp("This is a test sentence to verify the model works.")
        logger.info(f"SpaCy model verification: {len(doc)} tokens processed")
        
        return model_name
    except Exception as e:
        logger.error(f"Error downloading spaCy model: {e}")
        return None

def download_task_specific_models():
    """
    Download task-specific models defined in the configuration
    
    Returns:
        List of downloaded model names
    """
    logger.info("Downloading task-specific models")
    
    try:
        # Import configuration
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from config import TASK_SPECIFIC_MODELS
        
        # Download models for each task
        downloaded_models = []
        for task, config in TASK_SPECIFIC_MODELS.items():
            if config.get("model_name"):
                model_name = config["model_name"]
                logger.info(f"Downloading model for task '{task}': {model_name}")
                download_sentence_transformer(model_name)
                downloaded_models.append(model_name)
        
        return downloaded_models
    except ImportError as e:
        logger.error(f"Error importing task-specific model config: {e}")
        return []

def prepare_quantized_models():
    """
    Prepare models for quantization
    
    Returns:
        List of prepared model names
    """
    logger.info("Preparing models for quantization")
    
    try:
        # Import transformers and bitsandbytes
        from transformers import AutoTokenizer, AutoModel
        import bitsandbytes as bnb
        
        # Get list of models to prepare
        model_names = ["all-MiniLM-L6-v2"]
        
        # Try to import task-specific models
        try:
            sys.path.insert(0, os.path.join(base_dir, "src"))
            from config import TASK_SPECIFIC_MODELS
            
            for task, config in TASK_SPECIFIC_MODELS.items():
                if config.get("model_name") and config.get("quantize", False):
                    model_name = config["model_name"]
                    if model_name not in model_names:
                        model_names.append(model_name)
        except ImportError:
            pass
        
        # Prepare each model
        for model_name in model_names:
            logger.info(f"Preparing model for quantization: {model_name}")
            
            # Get full model path if it's a SentenceTransformer model
            model_id = f"sentence-transformers/{model_name}"
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            
            # Save downloaded files
            cache_dir = os.path.join(base_dir, "model_cache", "transformers")
            os.makedirs(cache_dir, exist_ok=True)
            
            # No need to save, just ensure it's downloaded
            logger.info(f"Model {model_name} prepared for quantization")
        
        return model_names
    except ImportError as e:
        logger.error(f"Error preparing models for quantization: {e}")
        return []

def main():
    """Main function to download all required models"""
    logger.info("Starting model download process")
    
    # Create model cache directory
    os.makedirs(os.path.join(base_dir, "model_cache"), exist_ok=True)
    
    # Download main sentence transformer model
    download_sentence_transformer()
    
    # Download spaCy model
    download_spacy_model()
    
    # Download task-specific models
    use_task_specific = os.environ.get("USE_TASK_SPECIFIC_MODELS") == "1"
    if use_task_specific:
        download_task_specific_models()
    
    # Prepare quantized models if quantization is enabled
    use_quantization = os.environ.get("USE_QUANTIZED_MODEL") == "1"
    if use_quantization:
        prepare_quantized_models()
    
    logger.info("All models downloaded and cached successfully")

if __name__ == "__main__":
    main() 