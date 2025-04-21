"""
Model Manager Module for Resume Scorer

This module handles task-specific model loading, caching, and efficient memory usage.
It supports multiple model types and quantization strategies.
"""

import os
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache
import threading
from typing import Dict, Optional, List, Union, Any, Callable
import torch
from ..config import (
    TASK_SPECIFIC_MODELS, 
    USE_TASK_SPECIFIC_MODELS,
    MODEL_CACHE_DIR,
    ENABLE_MODEL_OFFLOADING,
    CACHE_DIR,
    check_memory_usage,
    MEM_CHECK_INTERVAL
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model cache with thread safety
_model_cache = {}
_model_cache_lock = threading.RLock()
_request_counter = 0

class ModelManager:
    """Manager class for task-specific models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = TASK_SPECIFIC_MODELS
        self.use_task_specific = USE_TASK_SPECIFIC_MODELS
        self.loaded_tasks = set()
        self.request_count = 0
        self.lazy_loading = True  # Enable lazy loading of models by default
        self.models_warmed_up = False
    
    def get_model(self, task: str = "general"):
        """
        Get the appropriate model for a specific task
        
        Args:
            task: The task identifier (e.g., "general", "skills", "education")
            
        Returns:
            The model for the specified task
        """
        global _model_cache
        
        # If not using task-specific models, always use general model
        if not self.use_task_specific:
            task = "general"
        
        # If task doesn't exist in configs, default to general
        if task not in self.model_configs:
            logger.warning(f"Task '{task}' not found in model configs. Using 'general' model.")
            task = "general"
        
        # Increment request counter for memory checking
        _request_counter += 1
        
        # Check memory usage periodically
        if _request_counter % MEM_CHECK_INTERVAL == 0:
            if check_memory_usage():
                logger.warning("High memory usage detected - clearing unused models")
                self.clear_unused_models()
        
        # Thread-safe model cache access
        with _model_cache_lock:
            # Return cached model if available
            if task in _model_cache and _model_cache[task] is not None:
                logger.debug(f"Using cached model for task '{task}'")
                return _model_cache[task]
            
            # Get config for this task
            config = self.model_configs[task]
            model = None
            
            # Handle rule-based models (no actual ML model)
            if config.get("use_rules", False) and not config.get("model_name"):
                from .rule_based_models import create_rule_based_model
                model = create_rule_based_model(task, config)
            else:
                # Load embedding model
                model = self._load_embedding_model(config)
            
            # Cache the model
            _model_cache[task] = model
            self.loaded_tasks.add(task)
            return model
    
    def _load_embedding_model(self, config: Dict[str, Any]):
        """
        Load the appropriate embedding model based on configuration
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            The loaded model
        """
        model_name = config.get("model_name", "all-MiniLM-L6-v2")
        quantize = config.get("quantize", False)
        
        # Try different approaches to load the model
        if quantize:
            model = self._load_quantized_model(model_name)
        else:
            model = self._load_standard_model(model_name)
        
        # If model loading failed, use fallback
        if model is None:
            model = self._create_fallback_model(config.get("dimension", 384))
        
        return model
    
    def _load_quantized_model(self, model_name: str):
        """Load a quantized model using transformers"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            from transformers.utils import logging as tf_logging
            
            # Reduce verbosity of transformers
            tf_logging.set_verbosity_error()
            
            # Check offline mode
            offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            
            # Determine model_id
            model_id = f"sentence-transformers/{model_name}"
            if offline_mode:
                cache_path = Path(MODEL_CACHE_DIR) / model_name.replace('/', '_')
                if cache_path.exists():
                    model_id = str(cache_path)
            
            # Load model with 8-bit quantization
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, load_in_8bit=True)
            
            # Create a wrapper to match sentence-transformers interface
            class QuantizedModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.dimension = model.config.hidden_size
                
                def encode(self, sentences, **kwargs):
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    
                    # Process in small batches to save memory
                    batch_size = kwargs.get('batch_size', 8)
                    embeddings = []
                    
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i:i+batch_size]
                        inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                               return_tensors="pt", max_length=512)
                        
                        # Move to appropriate device
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                        
                        # Get model output
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        # Add to result
                        embeddings.append(batch_embeddings.cpu().numpy())
                    
                    return np.vstack(embeddings)
            
            return QuantizedModelWrapper(model, tokenizer)
            
        except Exception as e:
            logger.warning(f"Failed to load quantized model: {e}")
            return None
    
    def _load_standard_model(self, model_name: str):
        """Load a standard sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check offline mode 
            offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            
            # Try different approaches to load the model
            if offline_mode:
                # Try loading from specific path
                cache_path = Path(MODEL_CACHE_DIR) / model_name.replace('/', '_')
                if cache_path.exists():
                    return SentenceTransformer(str(cache_path))
            
            # Try loading with cache folder
            return SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR)
            
        except Exception as e:
            logger.warning(f"Failed to load standard model: {e}")
            return None
    
    def _create_fallback_model(self, dimension: int = 384):
        """Create a simple fallback model for situations where model loading fails"""
        
        class SimpleFallbackModel:
            """Simple model that generates embeddings based on word frequencies"""
            def __init__(self, dimension):
                self.dimension = dimension
            
            def encode(self, sentences, **kwargs):
                from collections import Counter
                import re
                
                if isinstance(sentences, str):
                    sentences = [sentences]
                
                embeddings = []
                for sentence in sentences:
                    # Simple tokenization
                    words = re.findall(r'\b\w+\b', sentence.lower())
                    if not words:
                        embeddings.append(np.zeros(self.dimension))
                        continue
                        
                    # Count words
                    counter = Counter(words)
                    
                    # Generate a simple embedding
                    embedding = np.zeros(self.dimension)
                    for i, (word, count) in enumerate(counter.most_common(min(self.dimension, len(counter)))):
                        # Hash the word to a value
                        word_val = sum(ord(c) for c in word) / 255.0
                        embedding[i % self.dimension] = word_val * count / len(words)
                    
                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        
        return SimpleFallbackModel(dimension)
    
    def get_embedding(self, text: str, task: str = "general"):
        """
        Get embedding for text using the appropriate model for the task
        
        Args:
            text: Text to encode
            task: Task identifier
            
        Returns:
            Embedding vector
        """
        model = self.get_model(task)
        
        # Get embedding using the model's encode method
        try:
            embedding = model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zeros as fallback
            dimension = self.model_configs.get(task, {}).get("dimension", 384)
            return np.zeros(dimension)
    
    def unload_model(self, task: str = None):
        """
        Unload a model to free memory
        
        Args:
            task: Task identifier or None to unload all models
        """
        global _model_cache
        
        if not ENABLE_MODEL_OFFLOADING:
            return
            
        with _model_cache_lock:
            if task is None:
                # Unload all models
                _model_cache.clear()
                self.loaded_tasks.clear()
                logger.info("Unloaded all models")
            elif task in _model_cache:
                # Unload specific model
                del _model_cache[task]
                if task in self.loaded_tasks:
                    self.loaded_tasks.remove(task)
                logger.info(f"Unloaded model for task '{task}'")
    
    def clear_unused_models(self, keep_tasks=None):
        """
        Clear all models except those needed for specified tasks
        
        Args:
            keep_tasks: List of tasks to keep models for
        """
        if keep_tasks is None:
            keep_tasks = ["general"]
            
        with _model_cache_lock:
            tasks_to_remove = [task for task in _model_cache.keys() if task not in keep_tasks]
            for task in tasks_to_remove:
                del _model_cache[task]
                if task in self.loaded_tasks:
                    self.loaded_tasks.remove(task)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def warmup_models(self, tasks=None):
        """
        Pre-load models to reduce first-request latency
        
        Args:
            tasks: List of tasks to warm up models for
        """
        if tasks is None:
            tasks = ["general"]
            
        logger.info(f"Warming up models for tasks: {tasks}")
        
        for task in tasks:
            model = self.get_model(task)
            # Run a dummy embedding
            dummy_text = "This is a warmup request."
            if hasattr(model, 'encode'):
                model.encode([dummy_text])
            
        self.models_warmed_up = True
        logger.info("Model warmup complete")

# Create a rule-based models module
def create_rule_based_model(task, config):
    """Create a rule-based model for tasks that don't need embedding models"""
    if task == "education":
        return EducationExtractor(config)
    return None

class EducationExtractor:
    """Rule-based education and certification extractor"""
    
    def __init__(self, config):
        self.ner_labels = config.get("ner_labels", ["ORG", "DATE"])
        self.education_keywords = [
            "degree", "bachelor", "master", "phd", "doctorate", "diploma",
            "certification", "certificate", "university", "college", "school"
        ]
        self.dimension = 1  # Not used for embeddings
        
        # Load spaCy model
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_loaded = True
        except:
            self.spacy_loaded = False
    
    def encode(self, sentences, **kwargs):
        """
        This doesn't return embeddings but structured extraction results
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        results = []
        for sentence in sentences:
            result = self.extract_education_info(sentence)
            results.append(result)
        
        return results
    
    def extract_education_info(self, text):
        """Extract education information from text"""
        if not self.spacy_loaded:
            return {"degrees": [], "institutions": [], "dates": []}
        
        doc = self.nlp(text)
        education_info = {
            "degrees": [],
            "institutions": [],
            "dates": []
        }
        
        # Extract organizations (potential educational institutions)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                education_info["institutions"].append(ent.text)
            elif ent.label_ == "DATE":
                education_info["dates"].append(ent.text)
        
        # Extract degree information using keywords
        for keyword in self.education_keywords:
            pattern = r'\b' + keyword + r'[\s\w]+(?:of|in|on)[\s\w]+\b'
            import re
            matches = re.findall(pattern, text, re.IGNORECASE)
            education_info["degrees"].extend(matches)
        
        return education_info

# Initialize a global model manager
_model_manager = None

def get_model_manager():
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager 

def warmup_models(tasks=None):
    """Pre-load models to reduce first-request latency"""
    manager = get_model_manager()
    manager.warmup_models(tasks)

def expand_skill_variations(skill):
    """Return common variations of a skill"""
    variations = {
        'react': ['reactjs', 'react.js'],
        'node.js': ['nodejs', 'node'],
        'express.js': ['expressjs', 'express'],
        'javascript': ['js', 'ecmascript'],
        'typescript': ['ts'],
        'mongodb': ['mongo'],
        'postgresql': ['postgres', 'psql'],
        'python': ['py', 'python3'],
        'java': ['core java', 'java se'],
        'aws': ['amazon web services', 'amazon cloud'],
        'azure': ['microsoft azure', 'azure cloud'],
        'docker': ['container', 'containerization'],
        'kubernetes': ['k8s', 'kube'],
        'machine learning': ['ml', 'machine-learning'],
        'artificial intelligence': ['ai'],
        'data science': ['data analytics', 'data analysis'],
    }
    return variations.get(skill.lower(), [skill]) 