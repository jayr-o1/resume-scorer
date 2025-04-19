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
MODEL_NAME = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L12-v2
USE_QUANTIZED_MODEL = True  # Enable quantized model for faster inference

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
    Get the sentence embedding model, optimized with quantization if enabled
    """
    global MODEL_NAME, USE_QUANTIZED_MODEL
    
    if not USE_QUANTIZED_MODEL:
        return SentenceTransformer(MODEL_NAME)
    
    try:
        # First try to use optimized ONNX model
        import onnxruntime
        from sentence_transformers.models import Pooling
        from sentence_transformers import __version__ as st_version
        from packaging import version
        
        # Only use ONNX if sentence-transformers version supports it
        if version.parse(st_version) >= version.parse("2.2.0"):
            # Check if model is cached
            model_path = CACHE_DIR / f"{MODEL_NAME}_quantized"
            
            if not model_path.exists():
                logger.info(f"Quantizing model {MODEL_NAME} for faster inference...")
                # First load the regular model
                model = SentenceTransformer(MODEL_NAME)
                # Then convert to ONNX format with quantization
                model.onnx_export(
                    output_path=str(model_path),
                    quantize=True,
                    opset_version=14
                )
                logger.info(f"Model quantized and saved to {model_path}")
            
            # Load with ONNX optimizations
            return SentenceTransformer(str(model_path))
        else:
            logger.warning(f"sentence-transformers {st_version} does not fully support ONNX. Using regular model.")
            return SentenceTransformer(MODEL_NAME)
    except ImportError:
        logger.warning("ONNX Runtime not installed. Using regular model.")
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Error setting up optimized model: {e}")
        logger.info(f"Falling back to standard model: {MODEL_NAME}")
        return SentenceTransformer(MODEL_NAME)

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
from utils.skill_ontology import get_skill_ontology

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

def analyze_resume(resume_text, job_details):
    """
    Analyze resume against job details using a local model with enhanced capabilities
    
    Parameters:
    - resume_text: Extracted text from the resume
    - job_details: Dictionary containing job summary, duties, skills, etc.
    
    Returns:
    - Dictionary with analysis results
    """
    try:
        # Check if resume_text is a dict from the enhanced extractor
        if isinstance(resume_text, dict) and "text" in resume_text:
            # Get language info
            language = resume_text.get("language", "en")
            extraction_method = resume_text.get("extraction_method", "")
            
            # Use the actual text for analysis
            resume_text = resume_text["text"]
            
            if not resume_text:
                return {
                    "error": "No text could be extracted from the resume",
                    "match_percentage": "0",
                    "recommendation": "Error during analysis"
                }
        
        # Extract job requirements
        job_text = " ".join([
            job_details.get('summary', ''),
            job_details.get('duties', ''),
            job_details.get('skills', ''),
            job_details.get('qualifications', '')
        ])
        
        # Generate hash for caching
        combined = (resume_text + job_text).encode('utf-8')
        hash_key = hashlib.md5(combined).hexdigest()
        
        # Check database cache first
        cached_result = load_from_db_cache(hash_key)
        if cached_result:
            return cached_result
        
        # Also check legacy file cache as fallback
        cache_path = get_cache_path(resume_text, job_text)
        cached_result = load_from_cache(cache_path)
        if cached_result:
            # Migrate to DB cache
            save_to_db_cache(hash_key, cached_result)
            return cached_result
        
        # Determine industry
        industry = get_industry_from_text(resume_text, job_details)
        industry_weights = INDUSTRY_KEYWORDS.get(industry, INDUSTRY_KEYWORDS["tech"])
        
        # Load model
        model = get_model()
        
        # Extract skills from job description
        skills_text = job_details.get('skills', '')
        skills_list = [s.strip().lower() for s in re.findall(r'[-•]?\s*([\w\s\+\#\.\-]+)(?:,|\n|$)', skills_text)]
        skills_list = [s for s in skills_list if len(s) > 2]  # Filter out very short items
        
        # Use NER to enhance skills extraction
        entities = extract_entities_with_ner(resume_text)
        ner_skills = entities.get("SKILL", [])
        
        # Use skill ontology for better skill matching
        skill_ontology = get_skill_ontology()
        skill_results = skill_ontology.detect_skills(resume_text)
        
        # Normalize required skills
        normalized_required_skills = [skill_ontology.normalize_skill(s) for s in skills_list]
        
        # Extract job titles and employment durations
        job_titles = extract_job_titles(resume_text)
        employment_durations = extract_employment_durations(resume_text)
        estimated_years = estimate_total_experience(employment_durations)
        
        # Create embeddings with caching
        resume_embedding = get_embedding(resume_text, model)
        job_embedding = get_embedding(job_text, model)
        
        # Calculate overall similarity
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        match_percentage = int(similarity * 100)
        
        # Extract skills matches using the improved matcher
        matched_skills = []
        missing_skills = []
        skill_details = []
        
        # First use the skill ontology results
        resume_skill_names = {skill["normalized_name"] for skill in skill_results}
        
        for skill in normalized_required_skills:
            if skill in resume_skill_names:
                matched_skills.append(skill)
                # Find the corresponding skill detail
                for skill_detail in skill_results:
                    if skill_detail["normalized_name"] == skill:
                        skill_details.append(skill_detail)
                        break
            else:
                # Fallback to traditional matching if not found in ontology
                if check_skill_match(resume_text, skill):
                    matched_skills.append(skill)
                else:
                    # Last resort - check embedding similarity
                    skill_embedding = get_embedding(skill, model)
                    skill_similarity = cosine_similarity([resume_embedding], [skill_embedding])[0][0]
                    
                    if skill_similarity > 0.5:  # Threshold for skill match
                        matched_skills.append(skill)
                    else:
                        missing_skills.append(skill)
        
        # Get alternative skills for missing required skills
        missing_skills_results = skill_ontology.find_missing_skills(resume_text, missing_skills)
        alternative_skills = missing_skills_results.get("alternative_skills", {})
        
        # Add NER-discovered skills that weren't in the job posting
        additional_skills = [skill for skill in ner_skills if skill not in matched_skills and skill not in missing_skills]
        
        # Add ontology skills not already included
        ontology_skill_names = {skill["name"] for skill in skill_results}
        additional_ontology_skills = [
            skill for skill in ontology_skill_names 
            if skill not in matched_skills and skill not in missing_skills and skill not in additional_skills
        ]
        additional_skills.extend(additional_ontology_skills)
        
        # Calculate match ratio
        total_skills = len(normalized_required_skills)
        matched_count = len(matched_skills)
        match_ratio = f"{matched_count}/{total_skills}"
        
        # Extract years of experience - now uses both explicit statements and employment durations
        applicant_years_explicit = extract_experiences(resume_text)
        
        # Use the greater of the two methods
        applicant_years = max(estimated_years, applicant_years_explicit or 0)
        if applicant_years == 0:
            applicant_years = "Not specified"
        
        required_years_match = re.search(r'(\d+)\+?\s+years?', job_details.get('qualifications', ''))
        required_years = required_years_match.group(1) if required_years_match else "Not specified"
        
        # Calculate experience impact on score
        if isinstance(applicant_years, int) and isinstance(required_years, str) and required_years.isdigit():
            req_years = int(required_years)
            if applicant_years < req_years:
                percentage_impact = f"-{min(20, (req_years - applicant_years) * 5)}%"
            elif applicant_years > req_years:
                percentage_impact = f"+{min(10, (applicant_years - req_years) * 2)}%"
            else:
                percentage_impact = "0%"
        else:
            percentage_impact = "0%"
        
        # Extract education
        applicant_education = extract_education(resume_text)
        required_education = extract_education(job_details.get('qualifications', ''))
        
        education_levels = ["Not specified", "High School", "Associate degree", "Bachelor's degree", "Master's degree", "PhD/Doctorate"]
        applicant_level = education_levels.index(applicant_education)
        required_level = education_levels.index(required_education)
        
        if applicant_level >= required_level:
            education_assessment = "Meets Requirement"
        else:
            education_assessment = "Below Requirement"
        
        # Extract certifications
        certifications = extract_certifications(resume_text)
        cert_impact = f"+{min(10, len(certifications) * 5)}%" if certifications else "0%"
        
        # Calculate keyword matching
        important_keywords = []
        for section in [job_details.get('summary', ''), job_details.get('duties', ''), job_details.get('qualifications', '')]:
            words = re.findall(r'\b\w{3,}\b', section.lower())
            important_keywords.extend([w for w in words if len(w) > 3])
        
        important_keywords = list(set(important_keywords))
        
        # Normalize resume text for keyword matching
        normalized_resume = normalize_text(resume_text)
        resume_words = set(re.findall(r'\b\w{3,}\b', normalized_resume))
        
        matched_keywords = [k for k in important_keywords if k in resume_words]
        keyword_ratio = f"{len(matched_keywords)}/{len(important_keywords)}"
        
        # Calculate weighted score based on industry
        skills_score = (matched_count / max(1, total_skills)) * 100
        
        exp_score = 0
        if isinstance(applicant_years, int) and isinstance(required_years, str) and required_years.isdigit():
            req_years = int(required_years)
            exp_score = min(100, (applicant_years / max(1, req_years)) * 100)
        
        edu_score = (applicant_level / max(1, required_level)) * 100 if required_level > 0 else 100
        
        cert_score = min(100, len(certifications) * 25)
        
        keyword_score = (len(matched_keywords) / max(1, len(important_keywords))) * 100
        
        weighted_score = (
            skills_score * industry_weights["skills_weight"] +
            exp_score * industry_weights["exp_weight"] +
            edu_score * industry_weights["edu_weight"] +
            cert_score * industry_weights["cert_weight"] +
            keyword_score * industry_weights["keyword_weight"]
        )
        
        match_percentage = int(weighted_score)
        
        # Determine recommendation
        if match_percentage >= 85:
            recommendation = "Strong Hire"
        elif match_percentage >= 70:
            recommendation = "Hire"
        elif match_percentage >= 50:
            recommendation = "Consider"
        else:
            recommendation = "Reject"
        
        # Create the resume analysis dictionary
        resume_analysis = {
            "match_percentage": str(match_percentage),
            "skills_match": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "additional_skills": additional_skills,
                "alternative_skills": alternative_skills,
                "match_ratio": match_ratio,
                "skill_details": skill_details
            },
            "experience": {
                "required_years": str(required_years),
                "applicant_years": str(applicant_years),
                "percentage_impact": percentage_impact,
                "job_titles": job_titles,
                "employment_durations": employment_durations,
                "total_estimated_years": str(estimated_years) if estimated_years > 0 else "Not detected"
            },
            "education": {
                "requirement": required_education,
                "applicant_education": applicant_education,
                "assessment": education_assessment
            },
            "certifications": {
                "relevant_certs": certifications,
                "percentage_impact": cert_impact
            },
            "keywords": {
                "matched": str(len(matched_keywords)),
                "total": str(len(important_keywords)),
                "match_ratio": keyword_ratio
            },
            "industry": {
                "detected": industry,
                "confidence": "high" if match_percentage > 70 else "medium"
            }
        }
        
        # Add benchmark and improvement suggestions
        resume_analysis["benchmark"] = benchmark_against_industry(industry, resume_analysis)
        resume_analysis["improvement_suggestions"] = get_improvement_suggestions(resume_analysis)
        resume_analysis["recommendation"] = recommendation
        
        # Add salary estimation if enough data
        if isinstance(applicant_years, int) and applicant_years > 0 and len(matched_skills) > 0:
            resume_analysis["salary_estimate"] = estimate_salary(
                industry, applicant_years, matched_skills, applicant_education
            )
        
        # Add confidence scores
        resume_analysis["confidence_scores"] = {
            "skills_match": min(100, 50 + (matched_count / max(1, total_skills)) * 50),
            "experience_match": 90 if isinstance(applicant_years, int) else 50,
            "education_match": 90 if applicant_level >= required_level else 70,
            "overall": similarity * 100
        }
        
        # Save to cache
        save_to_db_cache(hash_key, resume_analysis)
        save_to_cache(cache_path, resume_analysis)  # Legacy cache for backward compatibility
        
        return resume_analysis
        
    except Exception as e:
        logger.error(f"Error during local analysis: {e}", exc_info=True)
        return {
            "error": str(e),
            "match_percentage": "0",
            "recommendation": "Error during analysis"
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
    Process multiple resume files in parallel
    
    Parameters:
    - resume_files: List of paths to resume PDF files
    - job_details: Dictionary with job details
    - n_jobs: Number of parallel jobs (-1 means using all processors)
    
    Returns:
    - Dictionary mapping filenames to analysis results
    """
    from utils.pdf_extractor import extract_text_from_pdf
    
    def process_single_resume(resume_file):
        filename = os.path.basename(resume_file)
        resume_text = extract_text_from_pdf(resume_file)
        if resume_text:
            return filename, analyze_resume(resume_text, job_details)
        return filename, {"error": "Failed to extract text from PDF"}
    
    # Use joblib for parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_resume)(resume_file) for resume_file in resume_files
    )
    
    return dict(results) 