#!/usr/bin/env python3
"""
Script to pre-download all required models and cache them for Render deployment.
"""

import os
import sys
import logging
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set cache directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_dir, "model_cache")
os.environ["HF_HOME"] = os.path.join(base_dir, "model_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(base_dir, "model_cache")

# Add retry mechanism for network operations
def with_retry(func, max_retries=3, retry_delay=5):
    """Retry a function with exponential backoff"""
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                    raise
                wait_time = retry_delay * (2 ** (retries - 1))
                logger.warning(f"Retry {retries}/{max_retries} after error: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
    return wrapper

def download_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    """
    Download and cache the sentence transformer model
    
    Args:
        model_name: Name of the model to download (default: all-MiniLM-L6-v2)
        
    Returns:
        Name of the downloaded model or None if failed
    """
    logger.info(f"Downloading sentence transformer model: {model_name}")
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(base_dir, "model_cache", "sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Try to import package
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("Could not import sentence_transformers. Installing package...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
        
        # Load model to cache it
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Model {model_name} downloaded and cached")
        
        # Verify the model works
        try:
            import torch
            embeddings = model.encode(["Test sentence to verify the model works"])
            logger.info(f"Model verification: Embedding shape {embeddings.shape}")
        except Exception as e:
            logger.warning(f"Model verification failed, but model was downloaded: {e}")
        
        return model_name
    except Exception as e:
        logger.error(f"Error downloading sentence transformer model: {e}")
        logger.info("Trying alternative download method...")
        try:
            # Try alternative method by directly fetching the files
            import subprocess
            from huggingface_hub import snapshot_download
            
            model_id = f"sentence-transformers/{model_name}"
            snapshot_download(repo_id=model_id, local_dir=cache_dir)
            logger.info(f"Alternative download for {model_name} completed")
            return model_name
        except Exception as alt_e:
            logger.error(f"Alternative download also failed: {alt_e}")
            return None

def download_spacy_model(model_name="en_core_web_sm"):
    """
    Download and cache the spaCy model
    
    Args:
        model_name: Name of the spaCy model to download (default: en_core_web_sm)
        
    Returns:
        Name of the downloaded model or None if failed
    """
    logger.info(f"Downloading spaCy model: {model_name}")
    
    try:
        # Try to import spacy
        try:
            import spacy
        except ImportError:
            logger.error("Could not import spacy. Installing package...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            import spacy
        
        # Check if model is already installed
        try:
            nlp = spacy.load(model_name)
            logger.info(f"SpaCy model {model_name} already installed")
        except OSError:
            # Download the model
            try:
                spacy.cli.download(model_name)
                logger.info(f"SpaCy model {model_name} downloaded")
            except Exception as e:
                logger.error(f"Error with spacy.cli.download: {e}")
                # Try alternative method
                import subprocess
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                logger.info(f"SpaCy model {model_name} downloaded with alternative method")
            
        # Verify the model works
        try:
            nlp = spacy.load(model_name)
            doc = nlp("This is a test sentence to verify the model works.")
            logger.info(f"SpaCy model verification: {len(doc)} tokens processed")
        except Exception as e:
            logger.warning(f"Model verification failed, but model was downloaded: {e}")
        
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
        try:
            from config import TASK_SPECIFIC_MODELS
        except ImportError:
            logger.warning("Could not import TASK_SPECIFIC_MODELS from config. Using defaults.")
            TASK_SPECIFIC_MODELS = {
                "main": {"model_name": "all-MiniLM-L6-v2"}
            }
        
        # Download models for each task
        downloaded_models = []
        for task, config in TASK_SPECIFIC_MODELS.items():
            if config.get("model_name"):
                model_name = config["model_name"]
                logger.info(f"Downloading model for task '{task}': {model_name}")
                result = download_sentence_transformer(model_name)
                if result:
                    downloaded_models.append(result)
        
        return downloaded_models
    except Exception as e:
        logger.error(f"Error in download_task_specific_models: {e}")
        return []

def prepare_quantized_models():
    """
    Prepare models for quantization
    
    Returns:
        List of prepared model names or empty list if failed
    """
    logger.info("Preparing models for quantization")
    
    try:
        # Import transformers and bitsandbytes
        try:
            from transformers import AutoTokenizer, AutoModel
            import bitsandbytes as bnb
        except ImportError:
            logger.error("Could not import transformers or bitsandbytes. Installing packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "bitsandbytes"])
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
            logger.warning("Could not import TASK_SPECIFIC_MODELS from config. Using defaults.")
        
        # Prepare each model
        prepared_models = []
        for model_name in model_names:
            try:
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
                prepared_models.append(model_name)
            except Exception as e:
                logger.warning(f"Failed to prepare model {model_name} for quantization: {e}")
        
        return prepared_models
    except Exception as e:
        logger.error(f"Error preparing models for quantization: {e}")
        return []

def main():
    """Main function to download all required models"""
    logger.info("Starting model download process")
    
    # Create model cache directory
    os.makedirs(os.path.join(base_dir, "model_cache"), exist_ok=True)
    
    # First, attempt to download main sentence transformer model
    st_result = with_retry(download_sentence_transformer)()
    if not st_result:
        logger.warning("Failed to download sentence transformer model after retries.")
    
    # Attempt to download spaCy model
    spacy_result = with_retry(download_spacy_model)()
    if not spacy_result:
        logger.warning("Failed to download spaCy model after retries.")
    
    # Download task-specific models if requested
    use_task_specific = os.environ.get("USE_TASK_SPECIFIC_MODELS") == "1"
    if use_task_specific:
        download_task_specific_models()
    
    # Prepare quantized models if requested
    use_quantization = os.environ.get("USE_QUANTIZED_MODEL") == "1"
    if use_quantization:
        prepare_quantized_models()
    
    # Report on results
    if st_result or spacy_result:
        logger.info("Models downloaded and cached successfully")
    else:
        logger.warning("Some models failed to download. The application may still work with reduced functionality.")

if __name__ == "__main__":
    main() 