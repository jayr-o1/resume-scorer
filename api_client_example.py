#!/usr/bin/env python3
"""
Example client code for accessing the Resume Scorer API without authentication.
This code demonstrates how to make requests to your Render deployment.
"""

import requests
import json
import sys
import os
from pathlib import Path

# Replace with your Render deployment URL
API_URL = "https://your-app-name.onrender.com"

def analyze_resume(resume_file_path, job_details):
    """
    Upload a resume and analyze it against job details
    
    Parameters:
    - resume_file_path: Path to the resume PDF file
    - job_details: Dictionary with job details
    
    Returns:
    - JSON response from the API
    """
    # Check if file exists
    if not os.path.exists(resume_file_path):
        print(f"Error: File not found: {resume_file_path}")
        return None
    
    # Open the PDF file
    with open(resume_file_path, "rb") as f:
        files = {"resume": (os.path.basename(resume_file_path), f, "application/pdf")}
        
        # Prepare form data
        form_data = {
            "job_summary": job_details.get("summary", ""),
            "key_duties": job_details.get("duties", ""),
            "essential_skills": job_details.get("skills", ""),
            "qualifications": job_details.get("qualifications", ""),
            "industry_override": job_details.get("industry_override", "Auto-detect"),
            "translate": "false"
        }
        
        # Make the request
        print(f"Uploading resume to {API_URL}/analyze...")
        response = requests.post(
            f"{API_URL}/analyze",
            files=files,
            data=form_data
        )
        
        # Handle response
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

def list_skills():
    """Get available skills from the API"""
    response = requests.get(f"{API_URL}/skills")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def check_health():
    """Check if the API is healthy"""
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    # Check if API is healthy
    health = check_health()
    if health:
        print(f"API is healthy: {health}")
    else:
        print("API health check failed")
        return
    
    # Example job details
    job_details = {
        "summary": "Software Developer position focused on building web applications",
        "duties": "Develop and maintain web applications using modern technologies",
        "skills": "Python, JavaScript, React, FastAPI",
        "qualifications": "Bachelor's degree in Computer Science or related field, 2+ years experience"
    }
    
    # Check if resume file path is provided
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
        
        # Analyze the resume
        result = analyze_resume(resume_path, job_details)
        if result:
            print("\nAnalysis Results:")
            print(f"Match Percentage: {result['match_percentage']}")
            print(f"Recommendation: {result['recommendation']}")
            print("\nSkills Match:")
            for skill, details in result.get('skills_match', {}).items():
                print(f"- {skill}: {details}")
            
            # Save full results to file
            output_file = f"{Path(resume_path).stem}_analysis.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nFull results saved to {output_file}")
    else:
        print("Usage: python api_client_example.py <path_to_resume.pdf>")
        
if __name__ == "__main__":
    main() 