#!/usr/bin/env python3
"""
Script to test loading models without onnxruntime dependency
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set cache directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_dir, "model_cache")
os.environ["HF_HOME"] = os.path.join(base_dir, "model_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(base_dir, "model_cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(base_dir, "model_cache", "sentence_transformers")

def test_sentence_transformer():
    """Test if sentence-transformers can be loaded without onnxruntime"""
    logger.info("Testing sentence-transformers without onnxruntime")
    
    try:
        # Force torch backend
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        from sentence_transformers import SentenceTransformer
        
        # Create cache directory
        cache_dir = os.path.join(base_dir, "model_cache", "sentence_transformers")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load model with torch backend
        logger.info("Loading model with torch backend")
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        
        # Test the model
        test_sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_sentences)
        
        logger.info(f"Successfully generated embeddings of shape: {embeddings.shape}")
        return True
    except Exception as e:
        logger.error(f"Error testing sentence-transformers: {e}")
        return False

def test_transformers():
    """Test if Hugging Face transformers can be loaded"""
    logger.info("Testing transformers library")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test the model
        test_text = "This is a test sentence."
        encoded_input = tokenizer(test_text, return_tensors="pt")
        model_output = model(**encoded_input)
        
        logger.info(f"Successfully ran transformers model with output shape: {model_output.last_hidden_state.shape}")
        return True
    except Exception as e:
        logger.error(f"Error testing transformers: {e}")
        return False

def test_spacy():
    """Test if spaCy can be loaded"""
    logger.info("Testing spaCy")
    
    try:
        import spacy
        
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        # Test the model
        test_text = "This is a test sentence."
        doc = nlp(test_text)
        
        logger.info(f"Successfully processed text with spaCy: {len(doc)} tokens")
        return True
    except Exception as e:
        logger.error(f"Error testing spaCy: {e}")
        return False

def main():
    """Main function to test models without onnxruntime"""
    logger.info("Starting model testing without onnxruntime")
    
    # Create model cache directory
    os.makedirs(os.path.join(base_dir, "model_cache"), exist_ok=True)
    
    # Test all components
    results = {
        "sentence_transformer": test_sentence_transformer(),
        "transformers": test_transformers(),
        "spacy": test_spacy()
    }
    
    # Report results
    logger.info("Test results:")
    for name, success in results.items():
        logger.info(f"  {name}: {'SUCCESS' if success else 'FAILED'}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("All tests passed!")
    else:
        logger.warning("Some tests failed")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 