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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("model_cache")
CACHE_DIR.mkdir(exist_ok=True)

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
    """Generate a unique cache path based on input content"""
    # Create a unique hash based on both texts
    combined = resume_text + job_text
    hash_key = hashlib.md5(combined.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"analysis_cache_{hash_key}.pkl"

def save_to_cache(cache_path: Path, result: Dict) -> None:
    """Save analysis result to cache"""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved analysis to cache: {cache_path}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def load_from_cache(cache_path: Path) -> Optional[Dict]:
    """Load analysis result from cache if it exists"""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Loaded analysis from cache: {cache_path}")
            return result
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    return None

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
    if missing_skills:
        suggestions["skills"].append(f"Add the following missing skills: {', '.join(missing_skills[:5])}")
        suggestions["skills"].append("Use specific examples to demonstrate your listed skills")
    
    # Experience suggestions
    exp = analysis_result.get("experience", {})
    required_years = exp.get("required_years", "Not specified")
    applicant_years = exp.get("applicant_years", "Not specified")
    
    if required_years != "Not specified" and applicant_years != "Not specified":
        if int(required_years) > int(applicant_years):
            suggestions["experience"].append(f"Highlight projects that demonstrate depth of knowledge to compensate for fewer years")
        
    # Education suggestions
    edu = analysis_result.get("education", {})
    if edu.get("assessment") == "Below Requirement":
        suggestions["education"].append("Emphasize relevant professional certifications and training")
        suggestions["education"].append("Highlight specific coursework relevant to the job requirements")
    
    # General suggestions based on match percentage
    match_percentage = int(analysis_result.get("match_percentage", 0))
    if match_percentage < 70:
        suggestions["general"].append("Tailor your resume more specifically to this job description")
        suggestions["general"].append("Use more keywords from the job posting")
    
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
        # Extract job requirements
        job_text = " ".join([
            job_details.get('summary', ''),
            job_details.get('duties', ''),
            job_details.get('skills', ''),
            job_details.get('qualifications', '')
        ])
        
        # Check cache first
        cache_path = get_cache_path(resume_text, job_text)
        cached_result = load_from_cache(cache_path)
        if cached_result:
            return cached_result
        
        # Determine industry
        industry = get_industry_from_text(resume_text, job_details)
        industry_weights = INDUSTRY_KEYWORDS.get(industry, INDUSTRY_KEYWORDS["tech"])
        
        # Load more powerful sentence-transformers model
        model = SentenceTransformer('all-MiniLM-L12-v2')
        
        # Extract skills from job description
        skills_text = job_details.get('skills', '')
        skills_list = [s.strip().lower() for s in re.findall(r'[-•]?\s*([\w\s\+\#\.\-]+)(?:,|\n|$)', skills_text)]
        skills_list = [s for s in skills_list if len(s) > 2]  # Filter out very short items
        
        # Use NER to enhance skills extraction
        entities = extract_entities_with_ner(resume_text)
        ner_skills = entities.get("SKILL", [])
        
        # Create embeddings
        resume_embedding = model.encode([resume_text])[0]
        job_embedding = model.encode([job_text])[0]
        
        # Calculate overall similarity
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        match_percentage = int(similarity * 100)
        
        # Extract skills matches using the improved matcher
        matched_skills = []
        missing_skills = []
        
        for skill in skills_list:
            if check_skill_match(resume_text, skill):
                matched_skills.append(skill)
            else:
                # Fall back to embedding similarity only if direct match fails
                skill_embedding = model.encode([skill])[0]
                skill_similarity = cosine_similarity([resume_embedding], [skill_embedding])[0][0]
                
                if skill_similarity > 0.5:  # Threshold for skill match
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        # Add NER-discovered skills that weren't in the job posting
        additional_skills = [skill for skill in ner_skills if skill not in matched_skills and skill not in missing_skills]
        
        # Calculate match ratio
        total_skills = len(skills_list)
        matched_count = len(matched_skills)
        match_ratio = f"{matched_count}/{total_skills}"
        
        # Extract years of experience
        applicant_years = extract_experiences(resume_text) or "Not specified"
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
                "match_ratio": match_ratio
            },
            "experience": {
                "required_years": str(required_years),
                "applicant_years": str(applicant_years),
                "percentage_impact": percentage_impact
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
        
        # Save to cache
        save_to_cache(cache_path, resume_analysis)
        
        return resume_analysis
        
    except Exception as e:
        logger.error(f"Error during local analysis: {e}", exc_info=True)
        return {
            "error": str(e),
            "match_percentage": "0",
            "recommendation": "Error during analysis"
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