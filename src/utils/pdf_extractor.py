"""
Enhanced PDF extraction module with multi-language support and fallback mechanisms
"""

import pdfplumber
import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# For enhanced extraction
from langdetect import detect, LangDetectException
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace

# PyPDF2 as a fallback
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_with_pdfplumber(pdf_file: str) -> str:
    """Extract text from a PDF file using pdfplumber"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return clean_extra_whitespace(text.strip())
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return ""

def extract_text_with_pypdf2(pdf_file: str) -> str:
    """Extract text from a PDF file using PyPDF2 as fallback"""
    if not PYPDF2_AVAILABLE:
        return ""
    
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return clean_extra_whitespace(text.strip())
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {e}")
        return ""

def extract_text_with_unstructured(pdf_file: str) -> str:
    """Extract text using unstructured library for better extraction quality"""
    try:
        elements = partition_pdf(pdf_file, strategy="hi_res")
        text = "\n\n".join([str(element) for element in elements])
        return clean_extra_whitespace(text.strip())
    except Exception as e:
        logger.error(f"unstructured extraction failed: {e}")
        return ""

def detect_language(text: str) -> str:
    """Detect the language of the text"""
    try:
        if not text or len(text.strip()) < 20:
            return "en"  # Default to English for very short or empty text
        return detect(text)
    except LangDetectException:
        logger.warning("Language detection failed, defaulting to English")
        return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate non-English text to English if necessary
    Currently a placeholder - in a production system, this would use a translation API
    """
    # This is where you would implement a translation service
    # For example using Google Translate, DeepL, or other translation APIs
    
    # For now, we'll just return the original text with a notice
    if source_lang != "en":
        logger.info(f"Text is in {source_lang}, translation would be needed")
        # Add a note for demonstration purposes
        text = f"[DETECTED LANGUAGE: {source_lang}]\n\n{text}"
    
    return text

def extract_text_from_pdf(pdf_file: str, translate: bool = False) -> Dict:
    """
    Enhanced function to extract text from PDF with multiple fallback methods
    and language detection/translation support
    
    Parameters:
    - pdf_file: Path to the PDF file
    - translate: Whether to translate non-English text to English
    
    Returns:
    - Dictionary with extracted text, detected language, and metadata
    """
    result = {
        "text": "",
        "language": "en",
        "extraction_method": "",
        "translated": False,
        "metadata": {}
    }
    
    # Try different extraction methods in order of preference
    extraction_methods = [
        ("unstructured", extract_text_with_unstructured),
        ("pdfplumber", extract_text_with_pdfplumber),
        ("pypdf2", extract_text_with_pypdf2)
    ]
    
    for method_name, method_func in extraction_methods:
        try:
            text = method_func(pdf_file)
            if text and len(text.strip()) > 100:  # Consider successful if we get reasonable text
                result["text"] = text
                result["extraction_method"] = method_name
                logger.info(f"Successfully extracted text using {method_name}")
                break
        except Exception as e:
            logger.error(f"Error using {method_name}: {e}")
    
    # If we have text, detect language
    if result["text"]:
        try:
            result["language"] = detect_language(result["text"])
            
            # Translate if requested and not English
            if translate and result["language"] != "en":
                result["text"] = translate_to_english(result["text"], result["language"])
                result["translated"] = True
        except Exception as e:
            logger.error(f"Error in language processing: {e}")
    else:
        logger.error("Failed to extract text with any method")
        result["text"] = None
    
    return result

def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Extract common sections from a resume text
    
    Parameters:
    - text: The extracted text from the resume
    
    Returns:
    - Dictionary with sections (education, experience, skills, etc.)
    """
    if not text:
        return {}
    
    # Common section headers in resumes
    sections = {
        "summary": [],
        "experience": [],
        "education": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "languages": [],
        "interests": []
    }
    
    # Patterns for section headers (case insensitive)
    section_patterns = {
        "summary": r"(?i)(professional\s+summary|summary|profile|about\s+me|objective)",
        "experience": r"(?i)(experience|employment|work\s+history|career|professional\s+experience)",
        "education": r"(?i)(education|academic|qualifications|degrees|schools)",
        "skills": r"(?i)(skills|technical\s+skills|expertise|competencies|abilities)",
        "projects": r"(?i)(projects|portfolio|works)",
        "certifications": r"(?i)(certifications|certificates|credentials)",
        "languages": r"(?i)(languages|language\s+proficiency)",
        "interests": r"(?i)(interests|hobbies|activities)"
    }
    
    # Find potential section headers
    section_matches = []
    for section, pattern in section_patterns.items():
        for match in re.finditer(pattern, text):
            section_matches.append((match.start(), section))
    
    # Sort by position in text
    section_matches.sort()
    
    # Extract content between section headers
    results = {}
    for i, (pos, section) in enumerate(section_matches):
        # Find the end of the current section
        end_pos = len(text)
        if i < len(section_matches) - 1:
            end_pos = section_matches[i + 1][0]
        
        # Extract section content (starts after the line with the header)
        header_end = text.find('\n', pos)
        if header_end == -1:  # No newline found
            header_end = pos + 20  # Arbitrary offset
        
        section_content = text[header_end:end_pos].strip()
        results[section] = section_content
    
    return results 