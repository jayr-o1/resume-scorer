#!/usr/bin/env python3
"""
Simple test script to verify that the resume analyzer is working correctly.
This script tests the core functionality without requiring API endpoints.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

def test_pdf_extraction():
    """Test PDF text extraction functionality"""
    try:
        from src.utils.pdf_extractor import extract_text_from_pdf
        
        # Check if a sample PDF exists
        sample_paths = [
            "src/data/sample_resume.pdf",
            "tests/data/sample_resume.pdf",
            "tests/sample_resume.pdf"
        ]
        
        pdf_path = None
        for path in sample_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if not pdf_path:
            logger.error("No sample resume found. Please provide a PDF path.")
            return False
        
        logger.info(f"Testing PDF extraction with {pdf_path}")
        result = extract_text_from_pdf(pdf_path)
        
        if result and "text" in result and result["text"]:
            text_length = len(result["text"])
            logger.info(f"Successfully extracted {text_length} characters from PDF")
            logger.info(f"Text sample: {result['text'][:100]}...")
            return True
        else:
            logger.error("Failed to extract text from PDF")
            return False
            
    except Exception as e:
        logger.error(f"Error in PDF extraction test: {e}")
        return False

def test_resume_analysis():
    """Test resume analysis functionality"""
    try:
        from src.utils.analyzer import analyze_resume
        from src.utils.pdf_extractor import extract_text_from_pdf
        
        # Check if a sample PDF exists
        sample_paths = [
            "src/data/sample_resume.pdf",
            "tests/data/sample_resume.pdf",
            "tests/sample_resume.pdf"
        ]
        
        pdf_path = None
        for path in sample_paths:
            if os.path.exists(path):
                pdf_path = path
                break
        
        if not pdf_path:
            logger.error("No sample resume found. Please provide a PDF path.")
            return False
        
        # Extract text from PDF
        extraction_result = extract_text_from_pdf(pdf_path)
        if not extraction_result or not extraction_result.get("text"):
            logger.error("Failed to extract text from PDF")
            return False
        
        # Sample job details
        job_details = {
            "summary": "Software Engineer with experience in Python and web development",
            "duties": "Develop web applications, work with APIs, implement unit tests",
            "skills": "Python, JavaScript, React, API development",
            "qualifications": "Bachelor's degree, 3+ years experience"
        }
        
        # Run analysis
        logger.info("Running resume analysis...")
        analysis = analyze_resume(extraction_result, job_details)
        
        # Check results
        if analysis and "match_percentage" in analysis:
            logger.info(f"Analysis successful. Match percentage: {analysis['match_percentage']}")
            logger.info(f"Recommendation: {analysis.get('recommendation', 'Not available')}")
            if "skills_match" in analysis:
                matched = analysis["skills_match"].get("matched_skills", [])
                missing = analysis["skills_match"].get("missing_skills", [])
                logger.info(f"Matched skills: {matched}")
                logger.info(f"Missing skills: {missing}")
            return True
        else:
            logger.error("Analysis failed or returned incomplete results")
            return False
            
    except Exception as e:
        logger.error(f"Error in resume analysis test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n=== Testing Resume Analyzer Functionality ===\n")
    
    # Test PDF extraction
    print("\n--- Testing PDF Extraction ---")
    extraction_success = test_pdf_extraction()
    
    # Test resume analysis
    print("\n--- Testing Resume Analysis ---")
    analysis_success = test_resume_analysis()
    
    # Summary
    print("\n=== Test Results ===")
    print(f"PDF Extraction: {'✅ PASSED' if extraction_success else '❌ FAILED'}")
    print(f"Resume Analysis: {'✅ PASSED' if analysis_success else '❌ FAILED'}")
    
    if extraction_success and analysis_success:
        print("\n✅ All tests passed! The analyzer is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
        sys.exit(1) 