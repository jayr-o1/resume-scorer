import os
import json
import re
import hashlib
import pickle
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import logging
import time
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
import sqlite3
import threading
import multiprocessing
from joblib import Parallel, delayed

# Import project configuration
from ..config import TASK_SPECIFIC_MODELS, USE_TASK_SPECIFIC_MODELS

# Import model manager for task-specific models
try:
    from .model_manager import get_model_manager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)
DB_CACHE_PATH = CACHE_DIR / "cache.db"

# Model constants
MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller model than all-mpnet-base-v2
USE_QUANTIZED_MODEL = False  # Set to True to enable quantization

# Global model cache
_model_cache = None

# Thread-local storage for database connections
local_storage = threading.local()

# Industry-specific keywords and weightings
INDUSTRY_KEYWORDS = {
    "tech": {
        "keywords": ["software", "development", "programming", "cloud", "devops", "api", 
                    "frontend", "backend", "fullstack", "web", "mobile", "data", "ai", "ml"],
        "skills_weight": 0.35,
        "exp_weight": 0.25,
        "edu_weight": 0.15,
        "cert_weight": 0.15,
        "keyword_weight": 0.10
    },
    "finance": {
        "keywords": ["financial", "accounting", "banking", "investment", "trading", "portfolio", 
                    "analysis", "compliance", "risk", "audit", "tax", "regulatory"],
        "skills_weight": 0.25,
        "exp_weight": 0.30,
        "edu_weight": 0.20,
        "cert_weight": 0.15,
        "keyword_weight": 0.10
    },
    "healthcare": {
        "keywords": ["clinical", "patient", "medical", "health", "care", "nursing", 
                    "physician", "treatment", "pharmacy", "biotech", "research"],
        "skills_weight": 0.30,
        "exp_weight": 0.25,
        "edu_weight": 0.20,
        "cert_weight": 0.15, 
        "keyword_weight": 0.10
    },
    "marketing": {
        "keywords": ["marketing", "brand", "social media", "campaign", "digital", 
                    "seo", "content", "analytics", "strategy", "advertising"],
        "skills_weight": 0.35,
        "exp_weight": 0.20,
        "edu_weight": 0.15,
        "cert_weight": 0.10,
        "keyword_weight": 0.20
    }
}

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_LOADED = True
except (OSError, ImportError):
    logger.warning("SpaCy model not found. NER features will be limited.")
    SPACY_LOADED = False

# Initialize the database cache
def init_db_cache():
    """Initialize SQLite database for caching"""
    conn = sqlite3.connect(str(DB_CACHE_PATH))
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS analysis_cache (
        hash TEXT PRIMARY KEY, 
        result BLOB,
        timestamp REAL
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS embedding_cache (
        text_hash TEXT PRIMARY KEY,
        embedding BLOB,
        model_name TEXT,
        timestamp REAL
    )''')
    conn.commit()
    conn.close()

# Create cache tables on module load
init_db_cache()

def get_db_connection():
    """Get a thread-local database connection"""
    if not hasattr(local_storage, "conn"):
        local_storage.conn = sqlite3.connect(str(DB_CACHE_PATH))
    return local_storage.conn

def extract_experiences(text):
    """Extract years of experience from resume text"""
    # Look for patterns like "X years of experience" or "X+ years"
    patterns = [
        r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
        r'experience\s+(?:of\s+)?(\d+)\+?\s+years?',
        r'(\d+)-year\s+(?:of\s+)?experience',
        r'career\s+(?:of\s+)?(\d+)\+?\s+years?',
        r'(?:over|more than)\s+(\d+)\s+years?\s+(?:of\s+)?experience'
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years.extend([int(y) for y in matches])
    
    return max(years) if years else None

def extract_education(text):
    """Extract education level from resume text"""
    education_levels = {
        'phd': "PhD",
        'doctor': "Doctorate",
        'master': "Master's degree",
        'mba': "MBA",
        'bachelor': "Bachelor's degree",
        'bs ': "Bachelor of Science",
        'ba ': "Bachelor of Arts",
        'associate': "Associate degree",
        'high school': "High School"
    }
    
    found_levels = []
    for keyword, level in education_levels.items():
        if keyword in text.lower():
            found_levels.append(level)
    
    # Return the highest education level found
    if 'PhD' in found_levels or 'Doctorate' in found_levels:
        return "PhD/Doctorate"
    elif "Master's degree" in found_levels or "MBA" in found_levels:
        return "Master's degree"
    elif any(level for level in found_levels if "Bachelor" in level):
        return "Bachelor's degree"
    elif "Associate degree" in found_levels:
        return "Associate degree"
    elif "High School" in found_levels:
        return "High School"
    else:
        return "Not specified"

def extract_certifications(text):
    """Extract certifications from resume text"""
    # First normalize the text to handle encoding issues
    text = normalize_text(text)
    
    # Look for specific certifications by name
    specific_certs = [
        'aws certified', 'azure certified', 'gcp certified', 'google cloud certified',
        'comptia', 'cisco certified', 'ccna', 'ccnp', 'cissp', 'ceh', 'security+',
        'pmp', 'capm', 'scrum master', 'safe', 'itil', 'prince2',
        'cka', 'ckad', 'terraform', 'aws developer', 'aws solutions architect',
        'mongodb certified', 'oracle certified', 'mcsa', 'mcse', 'mcts',
        'rhce', 'rhcsa', 'lpic', 'cfa', 'cpa', 'six sigma'
    ]
    
    found_certs = []
    for cert in specific_certs:
        if cert in text:
            # Clean up and format the certification name
            clean_cert = cert.title()
            if cert.startswith('aws'):
                clean_cert = cert.upper().replace('AWS', 'AWS ')
            elif cert.startswith('azure'):
                clean_cert = cert.replace('azure', 'Azure')
            elif cert.startswith('gcp') or cert.startswith('google'):
                clean_cert = cert.replace('gcp', 'GCP').replace('google cloud', 'Google Cloud')
            
            found_certs.append(clean_cert)
    
    # Also use regex to find certification patterns
    cert_patterns = [
        r'(\w+)\s+certified\s+(developer|architect|professional|associate|engineer|administrator)',
        r'certified\s+(\w+)\s+(developer|architect|professional|associate|engineer|administrator)',
        r'(aws|azure|gcp|google cloud|comptia|cisco|oracle)\s+(certification|certified)',
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                clean_cert = ' '.join([part.title() for part in match if part])
                if 'aws' in clean_cert.lower():
                    clean_cert = clean_cert.replace('Aws', 'AWS')
                found_certs.append(clean_cert)
    
    # Remove duplicates and very short entries
    found_certs = [cert for cert in found_certs if len(cert) > 3]
    found_certs = list(set(found_certs))
    
    return found_certs

def normalize_text(text):
    """Normalize text to handle encoding issues"""
    # Replace common encoding issues
    text = text.replace('â€¢', '•')
    text = text.replace('â€"', '-')
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â', '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower().strip()

def extract_entities_with_ner(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy NER"""
    if not SPACY_LOADED:
        return {"SKILL": [], "ORG": [], "PRODUCT": []}
    
    doc = nlp(text)
    entities = {}
    
    # Extract standard entities
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    # Custom skill extraction using patterns
    skill_patterns = [
        r'\b(?:proficient|skilled|experienced|expertise)\s+in\s+([\w\s,]+)',
        r'\b(?:knowledge|understanding)\s+of\s+([\w\s,]+)',
        r'\b(?:familiar|worked)\s+with\s+([\w\s,]+)'
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            # Split by commas and 'and'
            for skill in re.split(r',|\sand\s', match):
                skill = skill.strip()
                if len(skill) > 2:  # Avoid very short matches
                    skills.add(skill)
    
    if "SKILL" not in entities:
        entities["SKILL"] = []
    entities["SKILL"].extend(list(skills))
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def get_industry_from_text(text: str, job_details: Dict[str, str]) -> str:
    """Determine the industry based on text content"""
    combined_text = text.lower() + " " + " ".join([v.lower() for v in job_details.values()])
    
    industry_scores = {}
    for industry, data in INDUSTRY_KEYWORDS.items():
        score = sum(1 for keyword in data["keywords"] if keyword in combined_text)
        industry_scores[industry] = score
    
    # Get the industry with the highest score
    if not industry_scores:
        return "tech"  # Default to tech
    
    return max(industry_scores.items(), key=lambda x: x[1])[0]

def check_skill_match(resume_text, skill):
    """Check if a skill is present in the resume text using more robust matching"""
    # Normalize both texts
    resume_text = normalize_text(resume_text)
    skill = normalize_text(skill)
    
    # Technical skills that should only match exactly
    tech_skills = {
        'typescript', 'node.js', 'express', 'graphql', 'postgresql', 'mongodb', 
        'redis', 'aws', 'azure', 'docker', 'kubernetes', 'ci/cd', 'microservices',
        'rest api', 'java', 'python', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt'
    }
    
    # For tech skills, only allow exact matches
    if skill.lower() in tech_skills:
        # Direct exact match for tech skills with word boundaries
        regex_match = re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE) is not None
        
        # Check for standalone tech word with word boundaries
        if regex_match:
            return True
            
        # Tech skills should be strictly matched, so check variants only for these skills
        # Get variations using the expanded skill variations function
        try:
            from .model_manager import expand_skill_variations
            variants = expand_skill_variations(skill)
        except ImportError:
            # Fallback to the hardcoded variants if model_manager not available
            tech_skill_variants = {
                'react': ['reactjs', 'react.js'],
                'node.js': ['nodejs', 'node'],
                'express.js': ['expressjs', 'express'],
                'javascript': ['js', 'ecmascript'],
                'typescript': ['ts'],
                'mongodb': ['mongo'],
                'postgresql': ['postgres', 'psql'],
                'rest': ['restful', 'rest api', 'restapi'],
                'git': ['github', 'gitlab', 'version control'],
                'ci/cd': ['continuous integration', 'continuous deployment', 'jenkins', 'github actions'],
                'aws': ['amazon web services', 'ec2', 's3', 'lambda'],
                'azure': ['microsoft azure', 'azure cloud'],
                'gcp': ['google cloud platform', 'google cloud'],
            }
            variants = tech_skill_variants.get(skill.lower(), [])
        
        # Check only specific tech variants with word boundaries
        for variant in variants:
            # Only match variants at word boundaries to avoid partial matches
            variant_match = re.search(r'\b' + re.escape(variant) + r'\b', resume_text, re.IGNORECASE) is not None
            if variant_match:
                return True
        
        # No match found for tech skill
        return False
    
    # For non-tech skills, continue with the existing matching logic
    # Direct match
    if skill in resume_text:
        return True
    
    # Check for skill as a standalone word
    if re.search(r'\b' + re.escape(skill) + r'\b', resume_text):
        return True
    
    # Common variants for specific technologies
    skill_variants = {
        # Marketing skills
        'marketing': ['digital marketing', 'marketers', 'market', 'marketing strategy'],
        'social media marketing': ['social media', 'social marketing', 'facebook marketing', 'instagram marketing'],
        'content marketing': ['content creation', 'content strategy', 'content writing', 'blog'],
        'email marketing': ['email campaigns', 'email newsletters', 'email strategy', 'mailchimp'],
        'seo': ['search engine optimization', 'search optimization', 'google search', 'keywords'],
        'sem': ['search engine marketing', 'ppc', 'pay per click', 'google ads', 'paid search'],
        'analytics': ['google analytics', 'data analysis', 'metrics', 'reporting', 'insights'],
        'campaign management': ['campaign', 'marketing campaign', 'campaign strategy', 'campaign execution'],
        'brand management': ['branding', 'brand strategy', 'brand development', 'brand identity'],
        'market research': ['research', 'competitor analysis', 'market analysis', 'consumer research'],
        
        # Business skills
        'project management': ['project planning', 'project coordination', 'project delivery', 'project lead'],
        'team management': ['team leadership', 'team lead', 'people management', 'managing teams'],
        'leadership': ['leading', 'team lead', 'leadership skills', 'people management'],
        'strategy': ['strategic', 'strategy development', 'strategic planning', 'strategic thinking'],
        'budget management': ['budgeting', 'financial planning', 'expense management', 'budget allocation'],
        'communication': ['written communication', 'verbal communication', 'presentation', 'public speaking'],
        'analysis': ['analytical', 'data analysis', 'market analysis', 'performance analysis']
    }
    
    # Check for variants
    for base_skill, variants in skill_variants.items():
        if skill == base_skill or skill in variants:
            # If the skill we're looking for matches this base_skill or is one of its variants
            if base_skill in resume_text:
                return True
            # Check if any variant is in the resume
            for variant in variants:
                if variant in resume_text:
                    return True
    
    # Check for partial matches for longer skill phrases
    if len(skill) > 10:  # Only for longer skill phrases
        skill_parts = skill.split()
        if len(skill_parts) > 1:
            # If most parts of a multi-word skill are found
            matches = sum(1 for part in skill_parts if part in resume_text and len(part) > 3)
            if matches >= max(2, len(skill_parts) - 1):
                return True
    
    # Special case for common skills that might be described differently
    common_skills_keywords = {
        'communication': ['communicate', 'interpersonal', 'presentation', 'speaking', 'writing'],
        'leadership': ['lead', 'leading', 'leader', 'managed team', 'supervised'],
        'analytical': ['analysis', 'analyze', 'data-driven', 'research', 'problem solving'],
        'creativity': ['creative', 'innovative', 'design thinking', 'new ideas'],
        'customer service': ['client', 'customer support', 'customer satisfaction', 'customer experience'],
        'marketing': ['promoted', 'advertised', 'marketed', 'campaign', 'brand', 'promotion'],
        'sales': ['selling', 'sales strategy', 'customer acquisition', 'business development', 'revenue'],
        'research': ['researched', 'analyzed', 'investigation', 'study', 'data collection']
    }
    
    for base_skill, keywords in common_skills_keywords.items():
        if skill == base_skill:
            for keyword in keywords:
                if keyword in resume_text:
                    return True
    
    return False

def get_cache_path(resume_text: str, job_text: str) -> Path:
    """Generate a cache file path based on the input texts"""
    combined = (resume_text + job_text).encode('utf-8')
    hash_key = hashlib.md5(combined).hexdigest()
    return CACHE_DIR / f"{hash_key}.pkl"

def get_hash_key(text: str) -> str:
    """Generate a hash key for the text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_to_cache(cache_path: Path, result: Dict) -> None:
    """Save analysis result to file cache (legacy method)"""
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

def load_from_cache(cache_path: Path) -> Optional[Dict]:
    """Load analysis result from file cache (legacy method)"""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache file: {e}")
    return None

def save_to_db_cache(hash_key: str, result: Dict) -> None:
    """Save analysis result to database cache"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO analysis_cache (hash, result, timestamp) VALUES (?, ?, strftime('%s','now'))",
            (hash_key, pickle.dumps(result))
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving to database cache: {e}")

def load_from_db_cache(hash_key: str) -> Optional[Dict]:
    """Load analysis result from database cache"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT result FROM analysis_cache WHERE hash = ?", (hash_key,))
        result = c.fetchone()
        if result:
            return pickle.loads(result[0])
    except Exception as e:
        logger.error(f"Error loading from database cache: {e}")
    return None

def save_embedding_to_cache(text_hash: str, embedding: np.ndarray, model_name: str) -> None:
    """Save embedding vector to database cache"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model_name, timestamp) VALUES (?, ?, ?, strftime('%s','now'))",
            (text_hash, pickle.dumps(embedding), model_name)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving embedding to cache: {e}")

def load_embedding_from_cache(text_hash: str, model_name: str) -> Optional[np.ndarray]:
    """Load embedding vector from database cache"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_name = ?", (text_hash, model_name))
        result = c.fetchone()
        if result:
            return pickle.loads(result[0])
    except Exception as e:
        logger.error(f"Error loading embedding from cache: {e}")
    return None

def get_model():
    """
    Get or load the sentence transformer model
    If running in offline mode, uses cached model
    """
    global _model_cache
    
    # Return cached model if available
    if _model_cache is not None:
        return _model_cache
    
    # Define simple fallback model class
    class SimpleFallbackModel:
        """Simple model that generates embeddings based on word frequencies"""
        def __init__(self):
            self.dimension = 384  # Similar to MiniLM-L6
        
        def encode(self, sentences, **kwargs):
            import numpy as np
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
    
    # Even simpler fallback model as a last resort
    class UltraSimpleFallbackModel:
        """Ultra-simple model that just returns zeros"""
        def __init__(self):
            self.dimension = 384
        
        def encode(self, sentences, **kwargs):
            import numpy as np
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.zeros((len(sentences), self.dimension))
    
    # Check if we're in offline mode
    offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             "model_cache", "sentence_transformers")
    
    # Try different approaches to load the model
    approaches = []
    
    # Add quantized model approach if enabled
    if USE_QUANTIZED_MODEL:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Function to load quantized model
            def load_quantized_model():
                # For MiniLM specifically
                model_id = "sentence-transformers/all-MiniLM-L6-v2"
                
                # Check if we should use the cache directory
                if offline_mode:
                    model_path = os.path.join(cache_dir, MODEL_NAME.replace('/', '_'))
                    if os.path.exists(model_path):
                        model_id = model_path
                
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
                        batch_size = 8
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
            
            # Add the quantized model approach
            approaches.append(load_quantized_model)
            
        except ImportError:
            logger.warning("Could not import transformers for quantization. Will use standard model instead.")
    
    # Add standard approaches
    approaches.extend([
        # 1. Try loading from cache directory
        lambda: SentenceTransformer(MODEL_NAME, cache_folder=cache_dir),
        
        # 2. Try loading from specific path for offline mode
        lambda: SentenceTransformer(os.path.join(cache_dir, MODEL_NAME.replace('/', '_'))) 
                if os.path.exists(os.path.join(cache_dir, MODEL_NAME.replace('/', '_'))) else None,
        
        # 3. Try loading directly from HuggingFace (if not in offline mode)
        lambda: SentenceTransformer(MODEL_NAME) if not offline_mode else None,
        
        # 4. Use SimpleFallbackModel
        lambda: SimpleFallbackModel(),
        
        # 5. Final fallback: UltraSimpleFallbackModel
        lambda: UltraSimpleFallbackModel()
    ])
    
    # Try each approach in order
    for i, approach in enumerate(approaches):
        try:
            model = approach()
            if model is not None:
                logger.info(f"Successfully loaded model using approach {i+1}")
                _model_cache = model
                return model
        except Exception as e:
            logger.warning(f"Approach {i+1} failed: {str(e)}")
            continue
    
    # If all approaches fail (should never happen due to fallbacks)
    logger.error("All model loading approaches failed")
    model = UltraSimpleFallbackModel()
    _model_cache = model
    return model

def get_embedding(text: str, model=None, task: str = "general"):
    """
    Get embedding for text with caching
    
    Args:
        text: Text to encode
        model: Pre-loaded model to use (if None, will load)
        task: The task this embedding is for (if using task-specific models)
        
    Returns:
        Embedding vector
    """
    global MODEL_NAME
    
    # Generate hash for the text
    text_hash = get_hash_key(text)
    
    # Try to load from cache first
    cached_embedding = load_embedding_from_cache(text_hash, MODEL_NAME)
    if cached_embedding is not None:
        return cached_embedding
    
    # If task-specific models are available, use the model manager
    if MODEL_MANAGER_AVAILABLE and USE_TASK_SPECIFIC_MODELS:
        try:
            # Get model manager instance
            model_manager = get_model_manager()
            # Get embedding directly
            embedding = model_manager.get_embedding(text, task)
            # Save to cache for future use
            save_embedding_to_cache(text_hash, embedding, MODEL_NAME)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to use model manager for task '{task}': {e}")
            # Fall back to the standard approach
    
    # If not using task-specific models or if it failed, use the standard approach
    if model is None:
        model = get_model()
    
    embedding = model.encode([text])[0]
    
    # Save to cache for future use
    save_embedding_to_cache(text_hash, embedding, MODEL_NAME)
    
    return embedding

def get_improvement_suggestions(analysis_result: Dict) -> Dict[str, List[str]]:
    """Generate improvement suggestions based on analysis results"""
    suggestions = {
        "skills": [],
        "experience": [],
        "education": [],
        "general": []
    }
    
    # Skills suggestions
    missing_skills = analysis_result.get("skills_match", {}).get("missing_skills", [])
    alternative_skills = analysis_result.get("skills_match", {}).get("alternative_skills", {})
    
    if missing_skills:
        suggestions["skills"].append(f"Add the following missing skills: {', '.join(missing_skills[:5])}")
        
        # Suggest alternatives based on what's already in the resume
        for missing_skill, alternatives in alternative_skills.items():
            if alternatives:
                suggestions["skills"].append(
                    f"Highlight your {', '.join(alternatives)} skills as they relate to {missing_skill}"
                )
        
        suggestions["skills"].append("Use specific examples to demonstrate your listed skills")
    
    # Experience suggestions
    exp = analysis_result.get("experience", {})
    required_years = exp.get("required_years", "Not specified")
    applicant_years = exp.get("applicant_years", "Not specified")
    
    if required_years != "Not specified" and applicant_years != "Not specified":
        try:
            req_years = int(required_years)
            app_years = int(applicant_years)
            
            if app_years < req_years:
                suggestions["experience"].append(f"Highlight projects that demonstrate depth of knowledge to compensate for fewer years")
                suggestions["experience"].append(f"Emphasize accomplishments that show expertise beyond your years of experience")
            elif app_years > req_years + 5:
                suggestions["experience"].append("Focus on recent and relevant achievements to avoid appearing overqualified")
        except ValueError:
            pass
    
    # Job title suggestions
    job_titles = exp.get("job_titles", [])
    if job_titles:
        if len(job_titles) > 3:
            suggestions["experience"].append("Consider consolidating similar job titles to show progression")
    else:
        suggestions["experience"].append("Make job titles more prominent in your resume")
    
    # Education suggestions
    edu = analysis_result.get("education", {})
    if edu.get("assessment") == "Below Requirement":
        suggestions["education"].append("Emphasize relevant professional certifications and training")
        suggestions["education"].append("Highlight specific coursework relevant to the job requirements")
        suggestions["education"].append("Consider adding ongoing education or professional development")
    
    # Certifications suggestions
    certs = analysis_result.get("certifications", {}).get("relevant_certs", [])
    if not certs:
        suggestions["general"].append("Add industry-relevant certifications to strengthen your qualifications")
    
    # General suggestions based on match percentage
    match_percentage = int(analysis_result.get("match_percentage", 0))
    if match_percentage < 70:
        suggestions["general"].append("Tailor your resume more specifically to this job description")
        suggestions["general"].append("Use more keywords from the job posting")
        
        # Check if ATS format issues are likely
        if len(analysis_result.get("skills_match", {}).get("matched_skills", [])) < 3:
            suggestions["general"].append("Ensure your resume is in an ATS-friendly format")
            suggestions["general"].append("Place a 'Skills' section near the top of your resume")
    
    return suggestions

def benchmark_against_industry(industry: str, analysis_result: Dict) -> Dict:
    """Compare the resume against industry benchmarks"""
    # Sample industry benchmarks (would be data-driven in production)
    benchmarks = {
        "tech": {"skills": 7, "experience": 3, "education": "Bachelor's degree"},
        "finance": {"skills": 5, "experience": 5, "education": "Bachelor's degree"},
        "healthcare": {"skills": 6, "experience": 4, "education": "Bachelor's degree"},
        "marketing": {"skills": 8, "experience": 3, "education": "Bachelor's degree"}
    }
    
    industry_data = benchmarks.get(industry, benchmarks["tech"])
    
    # Compare skills
    skills_count = len(analysis_result.get("skills_match", {}).get("matched_skills", []))
    skills_benchmark = skills_count / industry_data["skills"] * 100
    
    # Compare experience
    applicant_years = analysis_result.get("experience", {}).get("applicant_years", "0")
    applicant_years = int(applicant_years) if applicant_years.isdigit() else 0
    exp_benchmark = min(100, (applicant_years / industry_data["experience"]) * 100)
    
    # Compare education
    edu_levels = {"High School": 1, "Associate degree": 2, "Bachelor's degree": 3, "Master's degree": 4, "PhD/Doctorate": 5}
    applicant_edu = analysis_result.get("education", {}).get("applicant_education", "Not specified")
    industry_edu = industry_data["education"]
    
    if applicant_edu in edu_levels and industry_edu in edu_levels:
        edu_benchmark = min(100, (edu_levels[applicant_edu] / edu_levels[industry_edu]) * 100)
    else:
        edu_benchmark = 50  # Default value if education is not specified
    
    return {
        "industry": industry,
        "benchmarks": {
            "skills": round(skills_benchmark),
            "experience": round(exp_benchmark),
            "education": round(edu_benchmark),
            "overall": round((skills_benchmark + exp_benchmark + edu_benchmark) / 3)
        }
    }

# Import skill ontology
from .skill_ontology import get_skill_ontology

# Add regex patterns for job title and employment duration extraction
JOB_TITLE_PATTERNS = [
    r"(?i)(Senior|Lead|Principal|Staff|Junior|Associate)\s+(\w+\s+)?(Developer|Engineer|Architect|Designer|Manager|Director|Analyst|Consultant)",
    r"(?i)(Software|Systems|Solutions|Frontend|Backend|Full Stack|UI/UX|QA|Test|Data|Cloud|DevOps|Infrastructure|Security)\s+(Developer|Engineer|Architect|Designer|Manager|Director|Analyst|Consultant)",
    r"(?i)(CTO|CEO|CIO|COO|VP|Director|Head|Manager)\s+(?:of\s+)?(Engineering|Technology|Product|Development|IT)",
    r"(?i)(Product|Project|Program|Technical|Engineering)\s+(Manager|Lead|Director|Owner)"
]

EMPLOYMENT_DURATION_PATTERNS = [
    r"(?i)(\w+\s+\d{4})\s*[-–—]\s*(Present|Current|\w+\s+\d{4})",
    r"(?i)(\d{2}/\d{4})\s*[-–—]\s*(Present|Current|\d{2}/\d{4})",
    r"(?i)(\d{2}/\d{2})\s*[-–—]\s*(Present|Current|\d{2}/\d{2})",
    r"(?i)(\d{2}\.\d{4})\s*[-–—]\s*(Present|Current|\d{2}\.\d{4})"
]

def extract_job_titles(text: str) -> List[str]:
    """Extract job titles from resume text"""
    titles = []
    
    for pattern in JOB_TITLE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            title = match.group(0).strip()
            if title and len(title) > 3:  # Filter very short matches
                titles.append(title)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(titles))

def extract_employment_durations(text: str) -> List[Dict[str, str]]:
    """Extract employment durations from resume text"""
    durations = []
    
    for pattern in EMPLOYMENT_DURATION_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            if len(match.groups()) >= 2:
                start_date = match.group(1)
                end_date = match.group(2)
                
                duration = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "text": match.group(0)
                }
                
                # Calculate duration in months (approximately)
                try:
                    months = calculate_duration_months(start_date, end_date)
                    if months is not None:
                        duration["months"] = months
                except Exception:
                    pass
                
                durations.append(duration)
    
    return durations

def calculate_duration_months(start_date: str, end_date: str) -> Optional[int]:
    """Calculate duration in months from start and end dates"""
    from datetime import datetime
    import calendar
    
    # Helper to extract year and month from various formats
    def extract_year_month(date_str):
        date_str = date_str.lower()
        
        # Handle "Present" or "Current"
        if date_str in ["present", "current"]:
            now = datetime.now()
            return now.year, now.month
            
        # Try various date formats
        formats = [
            # Jan 2020
            (r"(\w+)\s+(\d{4})", 
             lambda m: (int(m.group(2)), list(calendar.month_abbr).index(m.group(1)[:3].title()))),
            # 01/2020
            (r"(\d{1,2})/(\d{4})", 
             lambda m: (int(m.group(2)), int(m.group(1)))),
            # 01/20
            (r"(\d{1,2})/(\d{2})", 
             lambda m: (2000 + int(m.group(2)), int(m.group(1)))),
            # 01.2020
            (r"(\d{1,2})\.(\d{4})", 
             lambda m: (int(m.group(2)), int(m.group(1))))
        ]
        
        for pattern, extract in formats:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return extract(match)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    start = extract_year_month(start_date)
    end = extract_year_month(end_date)
    
    if start and end:
        start_year, start_month = start
        end_year, end_month = end
        
        return (end_year - start_year) * 12 + (end_month - start_month)
    
    return None

def estimate_total_experience(durations: List[Dict[str, str]]) -> int:
    """Estimate total experience in years from duration list"""
    if not durations:
        return 0
        
    total_months = 0
    for duration in durations:
        if "months" in duration:
            total_months += duration["months"]
    
    # Convert to years, rounded
    return round(total_months / 12)

def analyze_resume(extraction_result, job_details):
    """
    Analyze a resume against job requirements
    
    Args:
        extraction_result: Dictionary containing the extracted text and metadata
        job_details: Dictionary containing job requirements
        
    Returns:
        Analysis result dictionary
    """
    try:
        resume_text = extraction_result.get("text", "").strip()
        if not resume_text:
            logger.error("Empty resume text in extraction result")
            return {
                "error": "Empty resume text",
                "match_percentage": 0,
                "recommendation": "Reject"
            }
        
        # Combine job details into a single text for analysis
        job_text = ""
        for key, value in job_details.items():
            if isinstance(value, str) and value.strip():
                job_text += value.strip() + "\n\n"
        
        if not job_text:
            logger.error("Empty job details")
            return {
                "error": "Empty job details",
                "match_percentage": 0,
                "recommendation": "Insufficient data"
            }
        
        # Generate cache path and hash key
        cache_path = get_cache_path(resume_text, job_text)
        hash_key = get_hash_key(resume_text + job_text)
        
        # Try DB cache first
        cached_result = load_from_db_cache(hash_key)
        if cached_result:
            logger.info("Analysis loaded from DB cache")
            return cached_result
            
        # Then try file cache
        cached_result = load_from_cache(cache_path)
        if cached_result:
            logger.info("Analysis loaded from file cache")
            return cached_result
        
        # Prepare model for semantic analysis
        use_task_specific = MODEL_MANAGER_AVAILABLE and USE_TASK_SPECIFIC_MODELS
        model = None
        model_manager = None
        
        try:
            if use_task_specific:
                model_manager = get_model_manager()
            else:
                model = get_model()
                # Ensure model is not None before proceeding
                if model is None:
                    logger.error("Model is None after get_model() call")
                    raise ValueError("Failed to initialize model")
                    
                # Check if model has the expected encode method
                if not hasattr(model, 'encode'):
                    logger.error("Model does not have encode method")
                    raise ValueError("Invalid model object")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Return a basic structure with error information
            return get_fallback_response(resume_text, job_details, str(e))
        
        logger.info("Starting resume analysis")
        
        # Extract job details for analysis
        job_title = job_details.get("job_title", "").strip()
        job_description = job_text.strip()
        
        # Process required years of experience (simple fallback)
        required_experience = 0
        try:
            required_experience = extract_years_from_text(job_description)
        except Exception as e:
            logger.warning(f"Error extracting required experience: {e}")
            
        # Process required education level (simple fallback)
        required_education = "Not specified"
        for level in ["bachelor", "master", "phd", "doctorate", "associate", "high school"]:
            if level in job_description.lower():
                if level in ["phd", "doctorate"]:
                    required_education = "PhD/Doctorate"
                elif level == "master":
                    required_education = "Master's degree"
                elif level == "bachelor":
                    required_education = "Bachelor's degree"
                elif level == "associate":
                    required_education = "Associate degree"
                elif level == "high school":
                    required_education = "High School"
                break
                
        # Get required skills based on job description
        required_skills = extract_skills_from_section(job_description)
        
        # ======== SKILLS ANALYSIS ========
        # Use the skills-specific model or extractor if available
        skills_match_result = {}
        if use_task_specific:
            try:
                # Get skills-specific model
                skills_model = model_manager.get_model("skills")
                
                # Check if it's a rule-based model (returns structured data)
                if hasattr(skills_model, 'process'):
                    # Direct processing
                    skills_data = skills_model.process(resume_text)
                    
                    # Match skills against required skills
                    matched_skills = []
                    missing_skills = []
                    
                    for skill in required_skills:
                        if skill.lower() in [s.lower() for s in skills_data.get("skills", [])]:
                            matched_skills.append(skill)
                        else:
                            missing_skills.append(skill)
                    
                    # Prepare skills match result
                    skills_match_result = {
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills,
                        "proficiency": skills_data.get("proficiency", {}),
                        "confidence": 0.85  # Rule-based typically has high confidence
                    }
                else:
                    # For embedding-based skill matching
                    matched_skills = []
                    missing_skills = []
                    
                    for skill in required_skills:
                        # Get embeddings using skills-specific model
                        skill_emb = get_embedding(skill, skills_model, "skills")
                        resume_emb = get_embedding(resume_text, skills_model, "skills")
                        
                        # Calculate similarity
                        similarity = cosine_similarity([skill_emb], [resume_emb])[0][0]
                        
                        if similarity > 0.8 or check_skill_match(resume_text, skill):
                            matched_skills.append(skill)
                        else:
                            missing_skills.append(skill)
                    
                    # Prepare skills match result
                    skills_match_result = {
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills,
                        "confidence": 0.75
                    }
            except Exception as e:
                logger.warning(f"Error using skills-specific model: {e}")
                # Fall back to the standard approach below
        
        # Use standard approach if task-specific model not used
        if not skills_match_result:
            matched_skills = []
            missing_skills = []
            
            for skill in required_skills:
                if check_skill_match(resume_text, skill):
                    matched_skills.append(skill)
                else:
                    # Try semantic matching
                    # Get general purpose model
                    skill_embedding = get_embedding(skill, model)
                    resume_embedding = get_embedding(resume_text, model)
                    similarity = cosine_similarity([skill_embedding], [resume_embedding])[0][0]
                    
                    if similarity > 0.8:  # Threshold for semantic similarity - increased from 0.6 to 0.8 for more stringent matching
                        matched_skills.append(skill)
                    else:
                        missing_skills.append(skill)
            
            # Use skill ontology to suggest alternatives
            alternative_skills = {}
            if missing_skills:
                try:
                    from .skill_ontology import get_skill_ontology
                    ontology = get_skill_ontology()
                    missing_data = ontology.find_missing_skills(resume_text, missing_skills)
                    alternative_skills = missing_data.get("alternatives", {})
                except Exception as e:
                    logger.warning(f"Error using skill ontology: {e}")
            
            skills_match_result = {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "alternative_skills": alternative_skills,
                "confidence": 0.7
            }
        
        # ======== EDUCATION ANALYSIS ========
        # Extract education information - use education-specific model if available
        education_info = {}
        if use_task_specific:
            try:
                # Get education-specific model
                education_model = model_manager.get_model("education")
                
                # Process education data
                if hasattr(education_model, 'process'):
                    education_data = education_model.process(resume_text)
                    
                    # Extract the highest education level
                    highest_edu = "Not specified"
                    for degree in education_data.get("degrees", []):
                        degree_lower = degree.lower()
                        if "phd" in degree_lower or "doctor" in degree_lower:
                            highest_edu = "PhD/Doctorate"
                            break
                        elif "master" in degree_lower or "mba" in degree_lower:
                            highest_edu = "Master's degree"
                        elif "bachelor" in degree_lower and highest_edu not in ["PhD/Doctorate", "Master's degree"]:
                            highest_edu = "Bachelor's degree"
                        elif "associate" in degree_lower and highest_edu not in ["PhD/Doctorate", "Master's degree", "Bachelor's degree"]:
                            highest_edu = "Associate degree"
                    
                    education_info = {
                        "applicant_education": highest_edu,
                        "required_education": required_education,
                        "institutions": education_data.get("institutions", []),
                        "degrees": education_data.get("degrees", []),
                        "dates": education_data.get("dates", []),
                        "confidence": 0.85
                    }
                    
                    # Assess if education meets requirements
                    edu_levels = {
                        "PhD/Doctorate": 5, 
                        "Master's degree": 4, 
                        "Bachelor's degree": 3, 
                        "Associate degree": 2, 
                        "High School": 1,
                        "Not specified": 0
                    }
                    
                    applicant_level = edu_levels.get(highest_edu, 0)
                    required_level = edu_levels.get(required_education, 0)
                    
                    if required_level == 0:
                        education_info["assessment"] = "No Requirement"
                    elif applicant_level >= required_level:
                        education_info["assessment"] = "Meets Requirement"
                    else:
                        education_info["assessment"] = "Below Requirement"
                else:
                    # Fallback to standard method if not a rule-based model
                    highest_edu = extract_education(resume_text)
                    education_info = standard_education_analysis(highest_edu, required_education)
            except Exception as e:
                logger.warning(f"Error using education-specific model: {e}")
                # Fall back to standard approach
        
        # Use standard approach if task-specific model not used 
        if not education_info:
            highest_edu = extract_education(resume_text)
            education_info = standard_education_analysis(highest_edu, required_education)
        
        # ======== EXPERIENCE ANALYSIS ========
        # Use experience-specific model if available
        experience_info = {}
        if use_task_specific:
            try:
                # Get experience-specific model
                experience_model = model_manager.get_model("experience")
                
                # Process experience data
                if hasattr(experience_model, 'process'):
                    experience_data = experience_model.process(resume_text)
                    
                    # Get years of experience
                    applicant_years = experience_data.get("years_of_experience")
                    if applicant_years is None:
                        # Calculate from durations
                        durations = experience_data.get("durations", [])
                        total_months = estimate_total_experience(durations)
                        applicant_years = total_months // 12
                    
                    experience_info = {
                        "required_years": str(required_experience) if required_experience else "Not specified",
                        "applicant_years": str(applicant_years) if applicant_years else "Not specified",
                        "job_titles": experience_data.get("job_titles", []),
                        "company_names": experience_data.get("company_names", []),
                        "durations": experience_data.get("durations", []),
                        "confidence": 0.8
                    }
                else:
                    # Fallback to standard method
                    experience_info = standard_experience_analysis(resume_text, required_experience)
            except Exception as e:
                logger.warning(f"Error using experience-specific model: {e}")
                # Fall back to standard approach
        
        # Use standard approach if task-specific model not used
        if not experience_info:
            experience_info = standard_experience_analysis(resume_text, required_experience)
        
        # The rest of the analysis follows the standard approach
        # ...
        
        # ======== CERTIFICATION ANALYSIS ========
        certifications = extract_certifications(resume_text)
        
        # Check for certifications mentioned in job description
        cert_keywords = ["certification", "certified", "certificate"]
        job_requires_certs = any(keyword in job_description.lower() for keyword in cert_keywords)
        
        certification_info = {
            "has_certifications": len(certifications) > 0,
            "relevant_certs": certifications,
            "job_requires_certs": job_requires_certs,
            "confidence": 0.75
        }
        
        # ======== OVERALL ANALYSIS ========
        # Calculate match percentage based on skills, experience, education
        weights = {
            "skills": 0.5,
            "experience": 0.3,
            "education": 0.15,
            "certifications": 0.05
        }
        
        # Adjust weights based on job industry if available
        industry = get_industry_from_text(job_text, job_details)
        if industry in INDUSTRY_KEYWORDS:
            industry_weights = INDUSTRY_KEYWORDS[industry]
            weights = {
                "skills": industry_weights["skills_weight"],
                "experience": industry_weights["exp_weight"],
                "education": industry_weights["edu_weight"],
                "certifications": industry_weights["cert_weight"]
            }
        
        # Calculate individual scores
        skill_score = len(skills_match_result["matched_skills"]) / max(1, len(required_skills)) * 100
        
        exp_score = 0
        if required_experience and isinstance(required_experience, (int, float)):
            try:
                applicant_exp = experience_info.get("applicant_years", "0")
                if isinstance(applicant_exp, str) and applicant_exp.isdigit():
                    applicant_exp = int(applicant_exp)
                elif not isinstance(applicant_exp, (int, float)):
                    applicant_exp = 0
                
                if applicant_exp >= required_experience:
                    exp_score = 100
                else:
                    exp_score = (applicant_exp / max(1, required_experience)) * 100
            except (ValueError, ZeroDivisionError):
                exp_score = 50  # Default if we can't calculate
        else:
            exp_score = 80  # No specific requirement
        
        # Education score
        edu_score = 0
        if education_info["assessment"] == "Meets Requirement":
            edu_score = 100
        elif education_info["assessment"] == "No Requirement":
            edu_score = 80
        elif education_info["assessment"] == "Below Requirement":
            # Give partial credit
            edu_levels = {
                "PhD/Doctorate": 5, 
                "Master's degree": 4, 
                "Bachelor's degree": 3, 
                "Associate degree": 2, 
                "High School": 1,
                "Not specified": 0
            }
            applicant_level = edu_levels.get(education_info["applicant_education"], 0)
            required_level = edu_levels.get(education_info["required_education"], 0)
            if required_level > 0:
                edu_score = (applicant_level / required_level) * 100
            else:
                edu_score = 50
        
        # Certification score
        cert_score = 0
        if certification_info["job_requires_certs"]:
            if certification_info["has_certifications"]:
                cert_score = 100
            else:
                cert_score = 0
        else:
            cert_score = 70  # Not required but good to have
        
        # Calculate weighted score
        match_percentage = (
            weights["skills"] * skill_score +
            weights["experience"] * exp_score +
            weights["education"] * edu_score +
            weights["certifications"] * cert_score
        )
        
        # Round to nearest integer
        match_percentage = round(match_percentage)
        
        # Generate recommendation
        recommendation = "Reject"
        if match_percentage >= 80:
            recommendation = "Hire"
        elif match_percentage >= 60:
            recommendation = "Interview"
        
        # Add improvement suggestions
        improvement_suggestions = get_improvement_suggestions({
            "match_percentage": match_percentage,
            "skills_match": skills_match_result,
            "experience": experience_info,
            "education": education_info,
            "certifications": certification_info
        })
        
        # Industry benchmarking
        benchmark = benchmark_against_industry(industry, {
            "skills_match": skills_match_result,
            "experience": experience_info,
            "education": education_info,
        })
        
        # Salary estimation
        salary_estimate = estimate_salary(
            industry, 
            int(experience_info.get("applicant_years", "0")) if experience_info.get("applicant_years", "0").isdigit() else 0,
            skills_match_result.get("matched_skills", []),
            education_info.get("applicant_education", "Not specified")
        )
        
        # Assemble final result
        analysis_result = {
            "match_percentage": match_percentage,
            "recommendation": recommendation,
            "skills_match": skills_match_result,
            "experience": experience_info,
            "education": education_info,
            "certifications": certification_info,
            "improvement_suggestions": improvement_suggestions,
            "industry_benchmark": benchmark,
            "salary_estimate": salary_estimate,
            "analysis_date": time.time()
        }
        
        # Save to cache
        save_to_cache(cache_path, analysis_result)
        save_to_db_cache(hash_key, analysis_result)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        logger.error(traceback.format_exc())
        return get_fallback_response(resume_text, job_details, str(e))

def standard_education_analysis(highest_edu, required_education):
    """Standard education analysis without task-specific models"""
    # Assess if education meets requirements
    edu_levels = {
        "PhD/Doctorate": 5, 
        "Master's degree": 4, 
        "Bachelor's degree": 3, 
        "Associate degree": 2, 
        "High School": 1,
        "Not specified": 0
    }
    
    applicant_level = edu_levels.get(highest_edu, 0)
    required_level = edu_levels.get(required_education, 0)
    
    if required_level == 0:
        assessment = "No Requirement"
    elif applicant_level >= required_level:
        assessment = "Meets Requirement"
    else:
        assessment = "Below Requirement"
    
    return {
        "applicant_education": highest_edu,
        "required_education": required_education,
        "assessment": assessment,
        "confidence": 0.7
    }

def standard_experience_analysis(resume_text, required_experience):
    """Standard experience analysis without task-specific models"""
    # Extract job titles
    job_titles = extract_job_titles(resume_text)
    
    # Extract employment durations
    durations = extract_employment_durations(resume_text)
    
    # Estimate total experience
    total_months = estimate_total_experience(durations)
    applicant_years = total_months // 12
    
    # Fallback to regex extraction if no durations found
    if applicant_years == 0:
        extracted_years = extract_experiences(resume_text)
        if extracted_years:
            applicant_years = extracted_years
    
    return {
        "required_years": str(required_experience) if required_experience else "Not specified",
        "applicant_years": str(applicant_years) if applicant_years else "Not specified",
        "job_titles": job_titles,
        "durations": durations,
        "confidence": 0.7
    }

def get_fallback_response(resume_text, job_details, error_message):
    """Return a fallback response structure when analysis fails"""
    logger.warning(f"Using fallback response due to: {error_message}")
    
    # Try basic matching as fallback
    matched_skills = []
    required_skills = job_details.get("required_skills", [])
    
    for skill in required_skills:
        if skill.lower() in resume_text.lower():
            matched_skills.append(skill)
    
    match_count = len(matched_skills)
    total_required = len(required_skills) if required_skills else 1
    match_ratio = f"{match_count}/{total_required}"
    match_percentage = int((match_count / total_required) * 100) if total_required > 0 else 0
    
    # Create a basic response with essential fields
    return {
        "error": f"Analysis error: {error_message}",
        "match_percentage": f"{max(20, match_percentage)}%",  # Ensure it's a string with %
        "recommendation": "Error during analysis - please review manually",
        "skills_match": {
            "matched_skills": matched_skills,
            "matched_count": match_count,
            "required_count": total_required,
            "match_ratio": match_ratio,
            "match_percentage": match_percentage,
            "additional_skills": []
        },
        "experience": {
            "required_years": job_details.get("required_experience", 0),
            "applicant_years": 0,
            "meets_requirement": False,
            "percentage_impact": "+0%"
        },
        "education": {
            "required_education": job_details.get("required_education", "Not specified"),
            "applicant_education": "Not detected",
            "assessment": "No impact"
        },
        "certifications": {
            "relevant_certs": [],
            "percentage_impact": "+0%"
        },
        "keywords": {
            "match_ratio": "0/0",
            "match_percentage": 0
        },
        "industry": {
            "detected": "unknown",
            "confidence": 0,
            "percentage_impact": "+0%"
        }
    }

def estimate_salary(industry: str, experience_years: int, skills: List[str], education: str) -> Dict:
    """
    Estimate salary range based on industry, experience, skills and education
    This is a simplified placeholder - a real implementation would use ML or market data
    """
    # Base salary by industry (simplified for demonstration)
    base_ranges = {
        "tech": {"min": 60000, "max": 150000},
        "finance": {"min": 65000, "max": 160000},
        "healthcare": {"min": 55000, "max": 140000},
        "marketing": {"min": 50000, "max": 130000}
    }
    
    base = base_ranges.get(industry, {"min": 50000, "max": 120000})
    
    # Experience multiplier
    exp_multiplier = min(2.0, 1.0 + (experience_years * 0.07))
    
    # Education multiplier
    edu_multipliers = {
        "High School": 0.85,
        "Associate degree": 0.9,
        "Bachelor's degree": 1.0,
        "Master's degree": 1.15,
        "PhD/Doctorate": 1.3,
        "Not specified": 0.9
    }
    
    edu_multiplier = edu_multipliers.get(education, 1.0)
    
    # Skill premium for in-demand skills
    premium_skills = {
        "machine learning": 0.15, "artificial intelligence": 0.15, "data science": 0.12,
        "cloud": 0.1, "aws": 0.1, "azure": 0.1, "gcp": 0.1,
        "react": 0.08, "angular": 0.08, "vue": 0.08,
        "python": 0.07, "golang": 0.1, "rust": 0.12, 
        "kubernetes": 0.1, "docker": 0.08, "devops": 0.09,
        "blockchain": 0.12, "cybersecurity": 0.11, "security": 0.09
    }
    
    # Calculate skill premium (max 30%)
    skill_premium = 1.0
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in premium_skills:
            skill_premium += premium_skills[skill_lower]
    
    skill_premium = min(1.3, skill_premium)
    
    # Calculate final range
    min_salary = int(base["min"] * exp_multiplier * edu_multiplier * skill_premium)
    max_salary = int(base["max"] * exp_multiplier * edu_multiplier * skill_premium)
    
    # Round to nearest 5000
    min_salary = round(min_salary / 5000) * 5000
    max_salary = round(max_salary / 5000) * 5000
    
    return {
        "min": min_salary,
        "max": max_salary,
        "currency": "USD",
        "note": "Estimation based on industry averages and skill demand"
    }

def format_analysis_result(analysis):
    """
    Format the analysis results into a readable string
    """
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    result = "AI Insights\n"
    result += f"Match Percentage:\n{analysis['match_percentage']}\n\n"
    
    # Industry information
    if "industry" in analysis:
        result += f"Industry:\n{analysis['industry']['detected'].capitalize()} (Confidence: {analysis['industry']['confidence']})\n\n"
    
    # Skills match
    skills = analysis['skills_match']
    result += f"Skills Match:\n{', '.join(skills['matched_skills'])} ({skills['match_ratio']})\n\n"
    
    # Additional skills
    if "additional_skills" in skills and skills["additional_skills"]:
        result += f"Additional Skills Detected:\n{', '.join(skills['additional_skills'])}\n\n"
    
    # Experience
    exp = analysis['experience']
    result += f"Experience:\nRequired: {exp['required_years']} | Applicant: {exp['applicant_years']} ({exp['percentage_impact']})\n\n"
    
    # Education
    edu = analysis['education']
    result += f"Education:\n{edu['applicant_education']} ({edu['assessment']})\n\n"
    
    # Certifications
    certs = analysis['certifications']
    if certs['relevant_certs']:
        result += f"Certifications:\n{', '.join(certs['relevant_certs'])} ({certs['percentage_impact']})\n\n"
    else:
        result += "Certifications:\nNone\n\n"
    
    # Keywords
    kw = analysis['keywords']
    result += f"Resume Keywords:\nMatched {kw['match_ratio']}\n\n"
    
    # Industry benchmark
    if "benchmark" in analysis:
        bench = analysis["benchmark"]["benchmarks"]
        result += f"Industry Benchmarks:\n"
        result += f"Skills: {bench['skills']}% | Experience: {bench['experience']}% | "
        result += f"Education: {bench['education']}% | Overall: {bench['overall']}%\n\n"
    
    # Improvement suggestions
    if "improvement_suggestions" in analysis:
        sugg = analysis["improvement_suggestions"]
        if any(sugg.values()):
            result += "Improvement Suggestions:\n"
            for category, items in sugg.items():
                if items:
                    result += f"- {items[0]}\n"
            result += "\n"
    
    # Recommendation
    rec = analysis['recommendation']
    emoji = "✅" if "hire" in rec.lower() else "❌"
    result += f"Recommendation:\n{rec} {emoji}"
    
    return result 

def batch_process_resumes(resume_files, job_details, n_jobs=-1):
    """
    Process multiple resumes in parallel
    
    Parameters:
    - resume_files: List of paths to resume PDFs
    - job_details: Dictionary of job details
    - n_jobs: Number of parallel jobs (default: use all cores)
    
    Returns:
    - Dictionary mapping filenames to analysis results
    """
    from joblib import Parallel, delayed
    import os
    from utils.pdf_extractor import extract_text_from_pdf
    
    def process_single_resume(resume_file):
        filename = os.path.basename(resume_file)
        extraction_result = extract_text_from_pdf(resume_file)
        if extraction_result and extraction_result.get("text"):
            return filename, analyze_resume(extraction_result, job_details)
        return filename, {"error": "Failed to extract text from PDF"}
    
    # Use joblib for parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_resume)(resume_file) for resume_file in resume_files
    )
    
    return dict(results) 

def extract_skills_from_section(section_text):
    """
    Extract skills from a dedicated skills section in a resume
    
    Parameters:
    - section_text: String with the skills section content
    
    Returns:
    - List of extracted skills
    """
    if not section_text:
        return []
        
    # Normalize text
    section_text = normalize_text(section_text)
    
    skills = []
    
    # Check for common formats
    
    # 1. Comma-separated list
    if "," in section_text:
        skills.extend([s.strip() for s in section_text.split(",") if s.strip()])
        
    # 2. Bullet points or dashes
    elif "-" in section_text or "•" in section_text or ":" in section_text:
        # Split by line
        lines = section_text.split("\n")
        for line in lines:
            # Remove bullet points and dashes
            cleaned = re.sub(r'^[\s•\-–—:]+', '', line).strip()
            if cleaned:
                # If multiple skills on one line with slashes or ampersands
                if "/" in cleaned or "&" in cleaned:
                    parts = re.split(r'\s*/\s*|\s*&\s*', cleaned)
                    skills.extend([p.strip() for p in parts if p.strip()])
                else:
                    skills.append(cleaned)
    
    # 3. Try to split by common phrases
    else:
        # Try to split by common separators
        tokens = re.split(r'\s+[\-–—•:]\s+|\s{2,}|proficient in|familiar with|skilled in|knowledge of', section_text)
        skills.extend([t.strip() for t in tokens if t.strip()])
    
    # Clean up skills:
    # 1. Remove very short items (likely not skills)
    # 2. Remove purely numeric items
    # 3. Remove very long items (likely sentences)
    skills = [s for s in skills if len(s) > 2 and not s.isdigit() and len(s) < 50]
    
    # Additional cleaning for marketing skills
    clean_skills = []
    for skill in skills:
        # Remove any trailing/leading punctuation
        skill = skill.strip('.,;:()[]{}')
        
        # Check if this is a valid skill (not just generic words)
        if len(skill.split()) > 1 or len(skill) > 4:
            clean_skills.append(skill)
    
    # Remove duplicates
    clean_skills = list(dict.fromkeys(clean_skills))
    
    return clean_skills 

def get_embedding_batch(texts: List[str], model=None, task: str = "general", batch_size: int = 8):
    """
    Get embeddings for multiple texts with batching and caching
    
    Args:
        texts: List of texts to encode
        model: Pre-loaded model to use (if None, will load)
        task: The task this embedding is for (if using task-specific models)
        batch_size: Size of batches for processing
        
    Returns:
        Array of embedding vectors
    """
    global MODEL_NAME
    
    # Handle empty input
    if not texts:
        return np.array([])
    
    # If task-specific models are available, use the model manager
    if MODEL_MANAGER_AVAILABLE and USE_TASK_SPECIFIC_MODELS:
        try:
            # Get model manager instance
            model_manager = get_model_manager()
            # Use batch processing
            return model_manager.get_embedding_batch(texts, task, batch_size)
        except Exception as e:
            logger.warning(f"Failed to use model manager for batch embedding: {e}")
            # Fall back to the standard approach
    
    # If not using task-specific models or if it failed, use the standard approach
    if model is None:
        model = get_model()
    
    # Use model's native batch processing if available
    try:
        if hasattr(model, 'encode'):
            return model.encode(texts, batch_size=batch_size)
    except Exception as e:
        logger.warning(f"Native batch processing failed: {e}")
    
    # Fallback to processing one by one
    embeddings = []
    for text in texts:
        embedding = get_embedding(text, model, task)
        embeddings.append(embedding)
    
    return np.array(embeddings) 