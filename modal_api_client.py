#!/usr/bin/env python3
"""
Example client for the Resume Scorer API deployed on Modal

This script demonstrates how to call the Resume Scorer API 
that has been deployed on Modal for analyzing resumes.
"""

import requests
import sys
import json
import os
import time

# The URL of the deployed Modal API
MODAL_API_URL = "https://jayr-o1--resume-scorer-api-fastapi-app.modal.run"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{MODAL_API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error checking API health: {e}")
        return None

def analyze_resume(resume_path, job_summary, essential_skills=None, qualifications=None):
    """
    Send a resume and job details to the API for analysis
    
    Args:
        resume_path: Path to resume file (PDF)
        job_summary: Job summary text
        essential_skills: Essential skills for the job
        qualifications: Required qualifications
    
    Returns:
        Analysis result JSON
    """
    start_time = time.time()
    print(f"Analyzing resume: {resume_path}")
    
    # Verify file exists
    if not os.path.exists(resume_path):
        print(f"Error: Resume file {resume_path} not found")
        sys.exit(1)
    
    # Verify file is a PDF
    if not resume_path.lower().endswith('.pdf'):
        print(f"Error: File {resume_path} is not a PDF. Only PDF files are supported.")
        sys.exit(1)
    
    # Prepare form data with the resume file and job details
    form_data = {
        'job_summary': job_summary,
    }
    
    if essential_skills:
        form_data['essential_skills'] = essential_skills
    
    if qualifications:
        form_data['qualifications'] = qualifications
    
    files = {
        'resume': (os.path.basename(resume_path), open(resume_path, 'rb'), 'application/pdf')
    }
    
    # Send request to the API
    try:
        print(f"Sending request to: {MODAL_API_URL}/analyze")
        response = requests.post(
            f"{MODAL_API_URL}/analyze",
            data=form_data,
            files=files,
            timeout=120  # 2 minute timeout for analysis
        )
        response.raise_for_status()
        
        # Close file handle
        files['resume'][1].close()
        
        result = response.json()
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return result
    except Exception as e:
        print(f"Error during API request: {e}")
        # Ensure file handle is closed
        if 'resume' in files:
            files['resume'][1].close()
        return None

def print_analysis_result(result):
    """Print the analysis result in a readable format"""
    if not result:
        print("No result to display")
        return
    
    print("\n=== RESUME ANALYSIS RESULT ===")
    print(f"Match Percentage: {result.get('match_percentage', 'N/A')}")
    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
    
    # Error handling
    if 'error' in result and result['error']:
        print(f"\nError: {result['error']}")
        return
    
    # Skills match
    skills_match = result.get('skills_match', {})
    if skills_match:
        print("\n--- SKILLS MATCH ---")
        print(f"Matched Skills: {', '.join(skills_match.get('matched_skills', []))}")
        print(f"Missing Skills: {', '.join(skills_match.get('missing_skills', []))}")
    
    # Experience
    experience = result.get('experience', {})
    if experience:
        print("\n--- EXPERIENCE ---")
        print(f"Applicant Experience: {experience.get('applicant_years', 'N/A')} years")
        print(f"Required Experience: {experience.get('required_years', 'N/A')} years")
        print(f"Assessment: {experience.get('assessment', 'N/A')}")
    
    # Education
    education = result.get('education', {})
    if education:
        print("\n--- EDUCATION ---")
        print(f"Applicant Education: {education.get('applicant_education', 'N/A')}")
        print(f"Required Education: {education.get('required_education', 'N/A')}")
        print(f"Assessment: {education.get('assessment', 'N/A')}")
    
    # Improvement suggestions
    suggestions = result.get('improvement_suggestions', {})
    if suggestions:
        print("\n--- IMPROVEMENT SUGGESTIONS ---")
        for area, tips in suggestions.items():
            if tips:
                print(f"\n{area.upper()}:")
                for tip in tips:
                    print(f"- {tip}")

if __name__ == "__main__":
    # Check if API is running
    health = check_api_health()
    if health:
        print(f"API is running (version: {health.get('version', 'unknown')})")
    else:
        print("API health check failed. Make sure the API is running.")
        sys.exit(1)
    
    # Get resume path from command line argument or use a default
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
    else:
        # Look for PDF resume files in the tests/sample_resumes directory
        sample_dir = os.path.join('tests', 'sample_resumes')
        if os.path.exists(sample_dir):
            pdf_files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.pdf')]
            if pdf_files:
                resume_path = os.path.join(sample_dir, pdf_files[0])
                print(f"Using sample resume: {resume_path}")
            else:
                print("No PDF files found in the sample directory.")
                print("Please provide a path to a resume PDF file")
                print("Usage: python modal_api_client.py path/to/resume.pdf")
                sys.exit(1)
        else:
            print("Sample directory not found.")
            print("Please provide a path to a resume PDF file")
            print("Usage: python modal_api_client.py path/to/resume.pdf")
            sys.exit(1)
    
    # Verify the file is a PDF
    if not resume_path.lower().endswith('.pdf'):
        print(f"Error: File {resume_path} is not a PDF. Only PDF files are supported.")
        sys.exit(1)
    
    # Example job details
    job_summary = "Software Engineer with experience in Python and web development"
    essential_skills = "Python, FastAPI, Docker, API Development, Git"
    qualifications = "Bachelor's degree in Computer Science or related field, 2+ years experience"
    
    # Analyze the resume
    result = analyze_resume(
        resume_path=resume_path,
        job_summary=job_summary,
        essential_skills=essential_skills,
        qualifications=qualifications
    )
    
    # Print the analysis result
    if result:
        print_analysis_result(result)
    else:
        print("Failed to get analysis result from the API") 