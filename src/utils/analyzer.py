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
from typing import Dict, List, Tuple, Optional, Union, Any
import sqlite3
import threading
import multiprocessing
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)
DB_CACHE_PATH = CACHE_DIR / "cache.db"

# Model constants
MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller model than all-mpnet-base-v2
USE_QUANTIZED_MODEL = False  # Disable quantization to reduce memory usage

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
    
    # Direct match
    if skill in resume_text:
        return True
    
    # Common variants for specific technologies
    skill_variants = {
        'react': ['reactjs', 'react.js'],
        'node.js': ['nodejs', 'node'],
        'express.js': ['expressjs', 'express'],
        'javascript': ['js', 'ecmascript'],
        'typescript': ['ts'],
        'mongodb': ['mongo'],
        'postgresql': ['postgres', 'psql'],
        'rest': ['restful', 'rest api', 'restapi'],
        'git': ['github', 'gitlab', 'version control'],
        'agile': ['scrum', 'kanban', 'sprint'],
        'ci/cd': ['continuous integration', 'continuous deployment', 'jenkins', 'github actions'],
        'unit testing': ['jest', 'mocha', 'testing', 'test driven'],
        'aws': ['amazon web services', 'ec2', 's3', 'lambda'],
        'azure': ['microsoft azure', 'azure cloud'],
        'gcp': ['google cloud platform', 'google cloud']
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
    If running on Render or in offline mode, uses cached model
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    # Check if we're in offline mode
    offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    on_render = "RENDER" in os.environ
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_cache", "sentence_transformers")
    
    try:
        # First attempt - try to load from the cache directory
        logger.info(f"Attempting to load model from cache: {cache_dir}")
        model = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        _model_cache = model
        logger.info("Successfully loaded model from cache")
        return model
    except Exception as e:
        if offline_mode or on_render:
            logger.error(f"Error loading model in offline mode: {e}")
            
            # Fallback for Render: Use a direct path to the cached model if it exists
            model_dir = os.path.join(cache_dir, MODEL_NAME.replace('/', '_'))
            if os.path.exists(model_dir):
                try:
                    logger.info(f"Attempting to load model from specific path: {model_dir}")
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(model_dir)
                    _model_cache = model
                    logger.info("Successfully loaded model from direct path")
                    return model
                except Exception as inner_e:
                    logger.error(f"Failed to load model from direct path: {inner_e}")
            
            # Return a simple fallback for handling text
            logger.warning("Using simple fallback for text processing")
            from sentence_transformers import SentenceTransformer
            
            # Create a very simple model that returns basic embeddings
            class SimpleFallbackModel:
                def __init__(self):
                    self.dimension = 384  # Similar to MiniLM-L6
                
                def encode(self, sentences, **kwargs):
                    """Generate simple embeddings based on word frequencies"""
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
            
            _model_cache = SimpleFallbackModel()
            return _model_cache
        else:
            # If not in offline mode, try again with regular loading
            try:
                logger.info("Attempting to load model from HuggingFace Hub")
                model = SentenceTransformer(MODEL_NAME)
                _model_cache = model
                return model
            except Exception as load_e:
                logger.error(f"Error loading model from HuggingFace Hub: {load_e}")
                # Always ensure we set a fallback model if everything else fails
                logger.warning("Using emergency fallback for text processing")
                _model_cache = SimpleFallbackModel()  # Re-use the simple fallback model
                return _model_cache
    except Exception as e:
        logger.error(f"Error setting up optimized model: {e}")
        logger.info(f"Falling back to standard model: {MODEL_NAME}")
        try:
            model = SentenceTransformer(MODEL_NAME)
            _model_cache = model
            return model
        except Exception as final_e:
            logger.error(f"Final error loading model: {final_e}")
            # Ensure we always have a fallback
            logger.warning("Using emergency fallback after all attempts failed")
            
            # Create a SimpleFallbackModel instance if it hasn't been defined yet
            try:
                if 'SimpleFallbackModel' not in locals():
                    class SimpleFallbackModel:
                        def __init__(self):
                            self.dimension = 384  # Similar to MiniLM-L6
                        
                        def encode(self, sentences, **kwargs):
                            """Generate simple embeddings based on word frequencies"""
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
                
                _model_cache = SimpleFallbackModel()
                return _model_cache
            except Exception as final_final_e:
                logger.error(f"Error creating fallback model: {final_final_e}")
                # Last resort - create an extremely simple model that just returns zeros
                class UltraSimpleFallbackModel:
                    def __init__(self):
                        self.dimension = 384
                    
                    def encode(self, sentences, **kwargs):
                        if isinstance(sentences, str):
                            sentences = [sentences]
                        return np.zeros((len(sentences), self.dimension))
                
                _model_cache = UltraSimpleFallbackModel()
                return _model_cache

def get_embedding(text: str, model=None):
    """
    Get embedding for text with caching
    """
    global MODEL_NAME
    
    # Generate hash for the text
    text_hash = get_hash_key(text)
    
    # Try to load from cache first
    cached_embedding = load_embedding_from_cache(text_hash, MODEL_NAME)
    if cached_embedding is not None:
        return cached_embedding
    
    # If not in cache, generate embedding
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
    Analyze a resume against job details
    
    Parameters:
    - extraction_result: Dictionary containing text and metadata from PDF
    - job_details: Dictionary with job_description, job_title, etc.
    
    Returns:
    - Dictionary with analysis results
    """
    try:
        # First check for cached result
        resume_text = extraction_result.get("text", "")
        job_text = job_details.get("job_description", "")
        
        if not resume_text:
            return {
                "error": "Resume text is empty",
                "match_percentage": 0,
                "recommendation": "Cannot process empty resume",
                "skills_match": {"matched_skills": [], "matched_count": 0, "required_count": 0, "match_ratio": "0/0", "match_percentage": 0, "additional_skills": []},
                "experience": {"required_years": 0, "applicant_years": 0, "meets_requirement": False, "percentage_impact": "+0%"},
                "education": {"required_education": "Not specified", "applicant_education": "Not specified", "assessment": "No impact"},
                "certifications": {"relevant_certs": [], "percentage_impact": "+0%"},
                "keywords": {"match_ratio": "0/0", "match_percentage": 0}
            }
        
        # Try to load from cache
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
            
        # Prepare for semantic analysis
        try:
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
        
        # Extract job requirements
        job_title = job_details.get("job_title", "").strip()
        job_description = job_details.get("job_description", "").strip()
        required_skills = job_details.get("required_skills", [])
        required_experience = job_details.get("required_experience", 0)
        required_education = job_details.get("required_education", "Not specified")
        
        # Process the resume
        try:
            job_tokens = len(job_description.split())
            resume_tokens = len(resume_text.split())
            logger.info(f"Resume: {resume_tokens} tokens | Job description: {job_tokens} tokens")
            
            # To prevent memory issues with very large documents
            resume_text_trimmed = " ".join(resume_text.split()[:25000])
            job_description_trimmed = " ".join(job_description.split()[:5000])
            
            # Generate embeddings safely with error handling
            try:
                resume_embedding = get_embedding(resume_text_trimmed, model)
                job_embedding = get_embedding(job_description_trimmed, model)
                
                # Verify embeddings are valid
                if resume_embedding is None or job_embedding is None:
                    logger.error("One or both embeddings are None")
                    raise ValueError("Failed to generate embeddings")
                
                # Calculate similarity
                similarity = np.dot(resume_embedding, job_embedding)
                base_match_percentage = int(similarity * 100)
            except Exception as embed_error:
                logger.error(f"Error generating embeddings: {embed_error}")
                # Fall back to a basic keyword matching approach
                common_words = set(normalize_text(resume_text_trimmed).split()) & set(normalize_text(job_description_trimmed).split())
                all_words = set(normalize_text(job_description_trimmed).split())
                if all_words:
                    base_match_percentage = int((len(common_words) / len(all_words)) * 60)  # Max 60% match with fallback
                else:
                    base_match_percentage = 30  # Default fallback
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            return get_fallback_response(resume_text, job_details, str(e))
        
        # Analyze required skills
        if not required_skills and job_description:
            # If no skills provided, try extracting them
            entities = extract_entities_with_ner(job_description)
            if entities and "SKILLS" in entities:
                required_skills = entities["SKILLS"]
        
        # Match skills in resume
        matched_skills = []
        additional_skills = []
        
        # First detect skills in the resume
        try:
            resume_entities = extract_entities_with_ner(resume_text)
            if resume_entities and "SKILLS" in resume_entities:
                detected_skills = resume_entities["SKILLS"]
                logger.info(f"Detected {len(detected_skills)} skills in resume")
            else:
                detected_skills = []
                
            # For each required skill, check if it's in the resume
            for skill in required_skills:
                if check_skill_match(resume_text, skill):
                    matched_skills.append(skill)
                
            # Find additional skills not in required list
            for skill in detected_skills:
                if skill not in required_skills and skill not in matched_skills:
                    additional_skills.append(skill)
        except Exception as skill_error:
            logger.error(f"Error matching skills: {skill_error}")
            matched_skills = []
            # Basic fallback for skill detection
            for skill in required_skills:
                if skill.lower() in resume_text.lower():
                    matched_skills.append(skill)
        
        # Calculate skill match metrics
        total_required = len(required_skills) if required_skills else 1
        match_count = len(matched_skills)
        match_ratio = f"{match_count}/{total_required}"
        match_percentage = int((match_count / total_required) * 100) if total_required > 0 else 0
        
        # Analyze experience
        try:
            experience_info = extract_employment_durations(resume_text)
            years_experience = estimate_total_experience(experience_info) / 12  # Convert months to years
            
            meets_requirement = years_experience >= required_experience
            experience_impact = calculate_experience_impact(years_experience, required_experience)
        except Exception as exp_error:
            logger.error(f"Error analyzing experience: {exp_error}")
            # Fallback for experience
            years_experience = 0
            meets_requirement = False
            experience_impact = "+0%"
        
        # Extract and analyze education
        try:
            education_info = extract_education(resume_text)
            education_level = determine_education_level(education_info)
            education_assessment = assess_education(education_level, required_education)
        except Exception as edu_error:
            logger.error(f"Error analyzing education: {edu_error}")
            # Fallback for education
            education_level = "Not specified"
            education_assessment = "No impact"
        
        # Extract and analyze certifications
        try:
            certifications = extract_certifications(resume_text)
            relevant_certs = []
            
            # Filter for relevant certifications
            for cert in certifications:
                if is_relevant_cert(cert, job_description, job_title):
                    relevant_certs.append(cert)
                    
            cert_impact = calculate_certification_impact(relevant_certs)
        except Exception as cert_error:
            logger.error(f"Error analyzing certifications: {cert_error}")
            # Fallback for certifications
            relevant_certs = []
            cert_impact = "+0%"
        
        # Extract job titles and check for relevant experience
        try:
            job_titles = extract_job_titles(resume_text)
            title_relevance = check_title_relevance(job_titles, job_title)
            past_relevance_score = title_relevance * 5  # 0-5% boost
        except Exception as title_error:
            logger.error(f"Error analyzing job titles: {title_error}")
            past_relevance_score = 0
        
        # Check keyword matches
        try:
            job_keywords = extract_keywords(job_description)
            resume_keywords = extract_keywords(resume_text)
            common_keywords = set(job_keywords) & set(resume_keywords)
            
            keyword_ratio = f"{len(common_keywords)}/{len(job_keywords)}" if job_keywords else "0/0"
            keyword_percentage = int((len(common_keywords) / len(job_keywords)) * 100) if job_keywords else 0
        except Exception as kw_error:
            logger.error(f"Error analyzing keywords: {kw_error}")
            keyword_ratio = "0/0"
            keyword_percentage = 0
        
        # Calculate adjusted match percentage
        skill_weight = 0.40
        exp_weight = 0.25
        edu_weight = 0.15
        cert_weight = 0.10
        keyword_weight = 0.10
        
        # Calculate base score components
        skill_score = match_percentage * skill_weight
        exp_score = (100 if meets_requirement else min(int(years_experience / required_experience * 100), 100) if required_experience > 0 else 100) * exp_weight
        
        # Calculate education score
        edu_scores = {
            "Exceeds requirements": 100,
            "Meets requirements": 90,
            "Partially meets requirements": 75,
            "Does not meet requirements": 50,
            "No impact": 80
        }
        edu_score = edu_scores.get(education_assessment, 80) * edu_weight
        
        # Certification and keyword scores
        cert_score = min(len(relevant_certs) * 20, 100) * cert_weight
        keyword_score = keyword_percentage * keyword_weight
        
        # Apply relevance boost
        adjusted_match = skill_score + exp_score + edu_score + cert_score + keyword_score
        final_match = min(100, int(adjusted_match + past_relevance_score))
        
        # Try to get industry match if possible
        industry_info = {"detected": "unknown", "confidence": 0.0}
        try:
            industry = get_industry_from_text(resume_text, job_details)
            if industry:
                industry_info = {"detected": industry, "confidence": 0.85}
        except Exception as ind_error:
            logger.error(f"Error detecting industry: {ind_error}")
        
        # Prepare the analysis result
        analysis_result = {
            "match_percentage": final_match,
            "skills_match": {
                "matched_skills": matched_skills,
                "matched_count": match_count,
                "required_count": total_required,
                "match_ratio": match_ratio,
                "match_percentage": match_percentage,
                "additional_skills": additional_skills[:10]  # Limit to top 10
            },
            "experience": {
                "required_years": required_experience,
                "applicant_years": round(years_experience, 1),
                "meets_requirement": meets_requirement,
                "percentage_impact": experience_impact
            },
            "education": {
                "required_education": required_education,
                "applicant_education": education_level,
                "assessment": education_assessment
            },
            "certifications": {
                "relevant_certs": relevant_certs,
                "percentage_impact": cert_impact
            },
            "keywords": {
                "match_ratio": keyword_ratio,
                "match_percentage": keyword_percentage
            },
            "industry": industry_info
        }
        
        # Generate improvement suggestions
        improvement_suggestions = get_improvement_suggestions(analysis_result)
        analysis_result["improvement_suggestions"] = improvement_suggestions
        
        # Industry benchmarking
        try:
            if industry_info["detected"] != "unknown":
                benchmark_data = benchmark_against_industry(industry_info["detected"], analysis_result)
                analysis_result["benchmark"] = benchmark_data
        except Exception as bench_error:
            logger.error(f"Error generating benchmarks: {bench_error}")
        
        # Generate recommendation
        if final_match >= 80:
            recommendation = "Strong match - Recommend to interview"
        elif final_match >= 70:
            recommendation = "Good match - Consider interviewing"
        elif final_match >= 60:
            recommendation = "Moderate match - May be worth reviewing"
        else:
            recommendation = "Low match - Not recommended for this position"
        
        analysis_result["recommendation"] = recommendation
        
        # Try to estimate a salary range
        try:
            if industry_info["detected"] != "unknown":
                salary_estimate = estimate_salary(
                    industry_info["detected"],
                    years_experience,
                    matched_skills + additional_skills,
                    education_level
                )
                analysis_result["salary_estimate"] = salary_estimate
        except Exception as salary_error:
            logger.error(f"Error estimating salary: {salary_error}")
        
        # Save to cache for future requests
        try:
            save_to_cache(cache_path, analysis_result)
            save_to_db_cache(hash_key, analysis_result)
        except Exception as cache_error:
            logger.error(f"Error saving to cache: {cache_error}")
        
        return analysis_result
    except Exception as e:
        logger.error(f"Unhandled exception in analyze_resume: {e}", exc_info=True)
        return get_fallback_response(extraction_result.get("text", ""), job_details, str(e))

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
        "match_percentage": max(20, match_percentage),  # Minimum 20% to avoid zero
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
    result += f"Match Percentage:\n{analysis['match_percentage']}%\n\n"
    
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