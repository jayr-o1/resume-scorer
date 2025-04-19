"""
Skill Ontology Module

This module provides functionality for normalizing skill names and detecting skills with context.
"""

import json
import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
DATA_DIR.mkdir(exist_ok=True)
SKILLS_FILE = DATA_DIR / "skills_ontology.json"

# Initialize default skill ontology if not exists
DEFAULT_SKILL_ONTOLOGY = {
    # Programming Languages
    "python": {
        "aliases": ["py", "python3", "python2", "pypy"],
        "category": "programming_language",
        "related": ["django", "flask", "pandas", "numpy", "tensorflow", "pytorch"]
    },
    "javascript": {
        "aliases": ["js", "ecmascript", "es6", "node.js", "nodejs"],
        "category": "programming_language",
        "related": ["typescript", "react", "vue", "angular", "node"]
    },
    "java": {
        "aliases": ["core java", "jdk", "java se", "java ee"],
        "category": "programming_language",
        "related": ["spring", "hibernate", "maven", "junit", "gradle"]
    },
    # Frameworks
    "react": {
        "aliases": ["reactjs", "react.js", "react js"],
        "category": "framework",
        "related": ["javascript", "jsx", "hooks", "redux"]
    },
    "angular": {
        "aliases": ["angularjs", "angular2+", "angular 2", "angular js"],
        "category": "framework",
        "related": ["typescript", "javascript", "rxjs"]
    },
    # DevOps
    "docker": {
        "aliases": ["container", "containerization"],
        "category": "devops",
        "related": ["kubernetes", "containers", "podman"]
    },
    "kubernetes": {
        "aliases": ["k8s", "kube"],
        "category": "devops",
        "related": ["docker", "containers", "helm", "istio"]
    },
    # Database
    "sql": {
        "aliases": ["structured query language"],
        "category": "database",
        "related": ["mysql", "postgresql", "oracle", "sql server"]
    },
    "mongodb": {
        "aliases": ["mongo", "document db"],
        "category": "database",
        "related": ["nosql", "mongoose", "atlas"]
    },
    # Cloud
    "aws": {
        "aliases": ["amazon web services", "amazon aws", "amazon cloud"],
        "category": "cloud",
        "related": ["ec2", "s3", "lambda", "dynamodb", "cloudformation"]
    },
    "azure": {
        "aliases": ["microsoft azure", "azure cloud", "microsoft cloud"],
        "category": "cloud",
        "related": ["azure functions", "cosmos db", "azure devops"]
    },
    # Data Science
    "machine learning": {
        "aliases": ["ml", "machine-learning"],
        "category": "data_science",
        "related": ["scikit-learn", "tensorflow", "pytorch", "data science"]
    },
    "tensorflow": {
        "aliases": ["tf", "tensor flow"],
        "category": "data_science",
        "related": ["machine learning", "deep learning", "keras", "neural networks"]
    },
    # Soft Skills
    "communication": {
        "aliases": ["communication skills", "verbal communication", "written communication"],
        "category": "soft_skill",
        "related": ["teamwork", "interpersonal", "presentation"]
    },
    "leadership": {
        "aliases": ["team lead", "team leader", "tech lead"],
        "category": "soft_skill",
        "related": ["management", "mentoring", "coaching"]
    }
}

# Skill proficiency indicators
PROFICIENCY_PATTERNS = {
    "expert": [
        r"expert\s+in\s+(\w+)",
        r"proficient\s+in\s+(\w+)",
        r"advanced\s+knowledge\s+of\s+(\w+)",
        r"extensive\s+experience\s+with\s+(\w+)"
    ],
    "intermediate": [
        r"intermediate\s+(\w+)",
        r"working\s+knowledge\s+of\s+(\w+)",
        r"experience\s+with\s+(\w+)",
        r"familiar\s+with\s+(\w+)"
    ],
    "beginner": [
        r"basic\s+knowledge\s+of\s+(\w+)",
        r"entry[- ]level\s+(\w+)",
        r"learning\s+(\w+)",
        r"beginner\s+(\w+)"
    ]
}

# Experience indicators
EXPERIENCE_PATTERNS = {
    "professional": [
        r"(\d+)\+?\s+years?\s+(?:of\s+)?experience\s+(?:with|in|using)\s+(\w+)",
        r"worked\s+(?:with|on)\s+(\w+)\s+for\s+(\d+)\+?\s+years?",
        r"professional\s+experience\s+with\s+(\w+)"
    ],
    "academic": [
        r"studied\s+(\w+)",
        r"course(?:work)?\s+in\s+(\w+)",
        r"academic\s+projects?\s+(?:with|in|using)\s+(\w+)"
    ],
    "personal": [
        r"personal\s+projects?\s+(?:with|in|using)\s+(\w+)",
        r"hobby\s+projects?\s+(?:with|in|using)\s+(\w+)",
        r"self[\s-]taught\s+(\w+)"
    ]
}

class SkillOntology:
    """Class for managing skill normalization and context-aware skill detection"""
    
    def __init__(self):
        self.skills_map = {}
        self.skill_patterns = {}
        self.load_ontology()
        self._compile_patterns()
    
    def load_ontology(self):
        """Load the skill ontology from file or initialize with defaults"""
        if SKILLS_FILE.exists():
            try:
                with open(SKILLS_FILE, 'r') as f:
                    self.skills_map = json.load(f)
                logger.info(f"Loaded skills ontology from {SKILLS_FILE}")
            except Exception as e:
                logger.error(f"Error loading skills ontology: {e}")
                self.skills_map = DEFAULT_SKILL_ONTOLOGY
                self._save_ontology()
        else:
            logger.info(f"Initializing default skills ontology")
            self.skills_map = DEFAULT_SKILL_ONTOLOGY
            self._save_ontology()
    
    def _save_ontology(self):
        """Save the skill ontology to file"""
        try:
            with open(SKILLS_FILE, 'w') as f:
                json.dump(self.skills_map, f, indent=2)
            logger.info(f"Saved skills ontology to {SKILLS_FILE}")
        except Exception as e:
            logger.error(f"Error saving skills ontology: {e}")
    
    def _compile_patterns(self):
        """Compile regex patterns for skill detection"""
        # Add patterns based on skill names and aliases
        for skill, info in self.skills_map.items():
            all_names = [skill] + info.get("aliases", [])
            
            # Create pattern for skill name and aliases with word boundaries
            pattern = r'\b(' + '|'.join(re.escape(name) for name in all_names) + r')\b'
            self.skill_patterns[skill] = re.compile(pattern, re.IGNORECASE)
    
    def normalize_skill(self, skill_name: str) -> str:
        """
        Normalize a skill name to its canonical form
        
        Parameters:
        - skill_name: The skill name to normalize
        
        Returns:
        - Normalized skill name or original if no match found
        """
        skill_lower = skill_name.lower()
        
        # Direct match
        if skill_lower in self.skills_map:
            return skill_lower
        
        # Check aliases
        for canonical, info in self.skills_map.items():
            if skill_lower in [alias.lower() for alias in info.get("aliases", [])]:
                return canonical
        
        # No match found, return original
        return skill_name
    
    def detect_skills(self, text: str) -> List[Dict[str, str]]:
        """
        Detect skills in text with proficiency level and experience context
        
        Parameters:
        - text: The text to analyze
        
        Returns:
        - List of dictionaries with skill name, proficiency and context
        """
        results = []
        found_skills = set()
        
        # First find basic skill occurrences
        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text):
                found_skills.add(skill)
        
        # Then find proficiency indicators
        skill_proficiency = {}
        for proficiency, patterns in PROFICIENCY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if len(match.groups()) > 0:
                        skill_name = match.group(1).lower()
                        normalized = self.normalize_skill(skill_name)
                        if normalized in found_skills or skill_name in found_skills:
                            skill_proficiency[normalized] = proficiency
        
        # Find experience context
        skill_experience = {}
        for context, patterns in EXPERIENCE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if len(match.groups()) > 0:
                        # Some patterns capture years and skill, others just skill
                        if len(match.groups()) == 2 and match.group(1).isdigit():
                            years = match.group(1)
                            skill_name = match.group(2).lower()
                        else:
                            skill_name = match.group(1).lower()
                            years = None
                        
                        normalized = self.normalize_skill(skill_name)
                        if normalized in found_skills or skill_name in found_skills:
                            entry = {"context": context}
                            if years:
                                entry["years"] = years
                            skill_experience[normalized] = entry
        
        # Compile results
        for skill in found_skills:
            result = {
                "name": skill,
                "normalized_name": skill,
                "proficiency": skill_proficiency.get(skill, "unknown"),
                "experience": skill_experience.get(skill, {"context": "mentioned"})
            }
            
            # Add related skills info
            if skill in self.skills_map:
                result["category"] = self.skills_map[skill]["category"]
                result["related"] = self.skills_map[skill]["related"]
            
            results.append(result)
        
        return results
    
    def find_missing_skills(self, resume_text: str, required_skills: List[str]) -> Dict[str, List[str]]:
        """
        Find missing skills and suggest alternatives based on related skills
        
        Parameters:
        - resume_text: Text of the resume
        - required_skills: List of required skills
        
        Returns:
        - Dictionary with missing skills and suggestions
        """
        normalized_required = [self.normalize_skill(skill) for skill in required_skills]
        resume_skills = self.detect_skills(resume_text)
        resume_skill_names = {skill["normalized_name"] for skill in resume_skills}
        
        missing = []
        suggestions = {}
        
        for req_skill in normalized_required:
            if req_skill not in resume_skill_names:
                missing.append(req_skill)
                
                # Check if we have related skills to suggest
                if req_skill in self.skills_map:
                    related = self.skills_map[req_skill]["related"]
                    # Check which related skills are in the resume
                    alternatives = [rel for rel in related if rel in resume_skill_names]
                    if alternatives:
                        suggestions[req_skill] = alternatives
        
        return {
            "missing_skills": missing,
            "alternative_skills": suggestions
        }
    
    def add_skill(self, name: str, aliases: List[str] = None, category: str = None, related: List[str] = None) -> bool:
        """
        Add a new skill to the ontology
        
        Parameters:
        - name: Name of the skill
        - aliases: List of alternative names
        - category: Category of the skill
        - related: List of related skills
        
        Returns:
        - True if successful, False otherwise
        """
        try:
            self.skills_map[name.lower()] = {
                "aliases": aliases or [],
                "category": category or "other",
                "related": related or []
            }
            
            # Recompile patterns with the new skill
            self._compile_patterns()
            self._save_ontology()
            return True
        except Exception as e:
            logger.error(f"Error adding skill: {e}")
            return False


# Singleton instance
_skill_ontology_instance = None

def get_skill_ontology() -> SkillOntology:
    """Get the singleton instance of the skill ontology"""
    global _skill_ontology_instance
    if _skill_ontology_instance is None:
        _skill_ontology_instance = SkillOntology()
    return _skill_ontology_instance 