"""
Rule-based Models Module

This module contains rule-based extraction models for specific tasks
that don't require the full embedding model approach.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
import spacy
from spacy.language import Language

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_LOADED = True
except (OSError, ImportError):
    logger.warning("Could not load spaCy model. Using fallback pattern matching.")
    SPACY_LOADED = False

def create_rule_based_model(task: str, config: Dict[str, Any]) -> Any:
    """
    Factory function to create rule-based models for specific tasks
    
    Args:
        task: Task identifier
        config: Configuration for the model
        
    Returns:
        Appropriate rule-based model instance
    """
    models = {
        "education": EducationExtractor,
        "skills": SkillsExtractor,
        "experience": ExperienceExtractor
    }
    
    if task in models:
        return models[task](config)
    
    logger.warning(f"No rule-based model available for task '{task}'. Using fallback.")
    return FallbackExtractor(config)

class BaseRuleModel:
    """Base class for all rule-based models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimension = 1  # Not used for embeddings but needed for interface compatibility
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Process texts and return structured information instead of embeddings
        
        Args:
            texts: Single text or list of texts to process
            
        Returns:
            List of extracted information dictionaries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            result = self.process(text)
            results.append(result)
        
        return results
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process a single text
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary of extracted information
        """
        # Override in subclasses
        return {}

class EducationExtractor(BaseRuleModel):
    """Rule-based education and certification extractor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Education-related keywords
        self.degree_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|associate)(?:\'s)?\s+(?:degree|of|in)\s+([^\.,]+)',
            r'\b(?:b\.?s\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|m\.?b\.?a\.?)\s+(?:in|of)?\s+([^\.,]+)',
            r'\bcertificate\s+in\s+([^\.,]+)',
            r'\bdiploma\s+in\s+([^\.,]+)'
        ]
        
        # Education institution patterns
        self.institution_patterns = [
            r'\b(?:university|college|institute|school)\s+of\s+([^\.,]+)',
            r'\b([^\.,]+)\s+(?:university|college|institute|school)\b'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]+\d{4}',
            r'\d{4}[\s]*-[\s]*(?:\d{4}|present|current|now)',
            r'\d{1,2}/\d{4}[\s]*-[\s]*(?:\d{1,2}/\d{4}|present|current|now)'
        ]
    
    def process(self, text: str) -> Dict[str, Any]:
        """Extract education-related information"""
        result = {
            "degrees": [],
            "institutions": [],
            "dates": [],
            "certifications": []
        }
        
        # Use spaCy for named entity recognition if available
        if SPACY_LOADED:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Check if it looks like an educational institution
                    if any(edu_term in ent.text.lower() for edu_term in ["university", "college", "institute", "school"]):
                        result["institutions"].append(ent.text)
                elif ent.label_ == "DATE":
                    result["dates"].append(ent.text)
        
        # Extract degrees using patterns
        for pattern in self.degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    degree = match.group(0).strip()
                    if "certif" in degree.lower():
                        result["certifications"].append(degree)
                    else:
                        result["degrees"].append(degree)
        
        # Extract institutions using patterns if spaCy didn't find them
        if not result["institutions"]:
            for pattern in self.institution_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    institution = match.group(0).strip()
                    result["institutions"].append(institution)
        
        # Extract dates using patterns if spaCy didn't find them
        if not result["dates"]:
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    date = match.group(0).strip()
                    result["dates"].append(date)
        
        return result

class SkillsExtractor(BaseRuleModel):
    """Rule-based skills extractor with contextual awareness"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Skill context patterns
        self.context_patterns = {
            "expert": [
                r"expert\s+in\s+([^\.,:;]+)",
                r"proficient\s+(?:in|with)\s+([^\.,:;]+)",
                r"advanced\s+knowledge\s+of\s+([^\.,:;]+)",
                r"extensive\s+experience\s+with\s+([^\.,:;]+)"
            ],
            "intermediate": [
                r"intermediate\s+([^\.,:;]+)",
                r"working\s+knowledge\s+of\s+([^\.,:;]+)",
                r"experience\s+with\s+([^\.,:;]+)",
                r"familiar\s+with\s+([^\.,:;]+)"
            ],
            "beginner": [
                r"basic\s+knowledge\s+of\s+([^\.,:;]+)",
                r"entry[- ]level\s+([^\.,:;]+)",
                r"learning\s+([^\.,:;]+)",
                r"beginner\s+([^\.,:;]+)"
            ]
        }
        
        # Load skill ontology if available
        try:
            from .skill_ontology import get_skill_ontology
            self.ontology = get_skill_ontology()
            self.ontology_available = True
        except ImportError:
            logger.warning("Skill ontology not available. Using pattern matching only.")
            self.ontology_available = False
            
            # Simple skill list if ontology not available
            self.skill_terms = [
                "python", "javascript", "java", "c++", "c#", "php", "ruby", 
                "sql", "nosql", "mongodb", "mysql", "postgresql", 
                "react", "angular", "vue", "node", "django", "flask", "spring",
                "docker", "kubernetes", "aws", "azure", "gcp", "devops",
                "machine learning", "ai", "deep learning", "data science",
                "git", "agile", "scrum", "leadership", "communication"
            ]
    
    def process(self, text: str) -> Dict[str, Any]:
        """Extract skills with context information"""
        result = {
            "skills": [],
            "context": {},
            "proficiency": {}
        }
        
        # Extract skills using ontology if available
        if self.ontology_available:
            skills_data = self.ontology.detect_skills(text)
            for skill_info in skills_data:
                skill_name = skill_info.get("name", "")
                if skill_name:
                    result["skills"].append(skill_name)
                    if "proficiency" in skill_info:
                        result["proficiency"][skill_name] = skill_info["proficiency"]
                    if "context" in skill_info:
                        result["context"][skill_name] = skill_info["context"]
        else:
            # Simple skill detection using keyword matching
            for skill in self.skill_terms:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    result["skills"].append(skill)
        
        # Extract proficiency context even if ontology was used
        for level, patterns in self.context_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) > 0:
                        context = match.group(1).strip().lower()
                        # Check if this context contains any of our skills
                        for skill in result["skills"]:
                            if skill.lower() in context:
                                result["proficiency"][skill] = level
                                break
        
        return result

class ExperienceExtractor(BaseRuleModel):
    """Rule-based experience and job title extractor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Job title patterns
        self.job_title_patterns = [
            r"(?i)(Senior|Lead|Principal|Staff|Junior|Associate)\s+(\w+\s+)?(Developer|Engineer|Architect|Designer|Manager|Director|Analyst|Consultant)",
            r"(?i)(Software|Systems|Solutions|Frontend|Backend|Full Stack|UI/UX|QA|Test|Data|Cloud|DevOps|Infrastructure|Security)\s+(Developer|Engineer|Architect|Designer|Manager|Director|Analyst|Consultant)",
            r"(?i)(CTO|CEO|CIO|COO|VP|Director|Head|Manager)\s+(?:of\s+)?(Engineering|Technology|Product|Development|IT)",
            r"(?i)(Product|Project|Program|Technical|Engineering)\s+(Manager|Lead|Director|Owner)"
        ]
        
        # Duration patterns
        self.duration_patterns = [
            r"(?i)(\w+\s+\d{4})\s*[-–—]\s*(Present|Current|\w+\s+\d{4})",
            r"(?i)(\d{2}/\d{4})\s*[-–—]\s*(Present|Current|\d{2}/\d{4})",
            r"(?i)(\d{2}/\d{2})\s*[-–—]\s*(Present|Current|\d{2}/\d{2})",
            r"(?i)(\d{2}\.\d{4})\s*[-–—]\s*(Present|Current|\d{2}\.\d{4})"
        ]
        
        # Year of experience patterns
        self.year_patterns = [
            r"(\d+)\+?\s+years?\s+(?:of\s+)?experience",
            r"experience\s+(?:of|for)?\s+(\d+)\+?\s+years?",
            r"(\d+)\+?\s+years?\s+in\s+(?:the\s+)?(?:field|industry)",
        ]
    
    def process(self, text: str) -> Dict[str, Any]:
        """Extract experience information"""
        result = {
            "job_titles": [],
            "durations": [],
            "years_of_experience": None,
            "company_names": []
        }
        
        # Extract job titles
        for pattern in self.job_title_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                title = match.group(0).strip()
                if title and len(title) > 3:  # Filter very short matches
                    result["job_titles"].append(title)
        
        # Extract durations
        for pattern in self.duration_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    duration = {
                        "start": match.group(1),
                        "end": match.group(2)
                    }
                    result["durations"].append(duration)
        
        # Extract years of experience
        for pattern in self.year_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    try:
                        years = int(match.group(1))
                        if result["years_of_experience"] is None or years > result["years_of_experience"]:
                            result["years_of_experience"] = years
                    except ValueError:
                        pass
        
        # Extract company names using spaCy if available
        if SPACY_LOADED:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Avoid universities and schools (already handled by education extractor)
                    if not any(edu_term in ent.text.lower() for edu_term in ["university", "college", "institute", "school"]):
                        result["company_names"].append(ent.text)
        
        return result

class FallbackExtractor(BaseRuleModel):
    """Generic fallback extractor for tasks without specific implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def process(self, text: str) -> Dict[str, Any]:
        """Simple keyword frequency analysis"""
        result = {
            "keywords": {},
            "entities": {}
        }
        
        # Simple word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip very short words
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        
        # Sort by frequency
        result["keywords"] = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Extract entities if spaCy is available
        if SPACY_LOADED:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ not in result["entities"]:
                    result["entities"][ent.label_] = []
                result["entities"][ent.label_].append(ent.text)
        
        return result 