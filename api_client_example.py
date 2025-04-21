#!/usr/bin/env python3
"""
Example client for the Resume Scorer API

This script demonstrates how to call the Resume Scorer API 
for analyzing resumes against job descriptions.
"""

import argparse
import requests
import json
import os
import time
from pathlib import Path
import sys

def analyze_resume(url, resume_path, job_path):
    """
    Send a resume and job description to the API for analysis
    
    Args:
        url: API endpoint URL
        resume_path: Path to resume file (PDF, DOCX, or TXT)
        job_path: Path to job description JSON file
    
    Returns:
        Analysis result JSON
    """
    start_time = time.time()
    print(f"Analyzing resume: {resume_path}")
    print(f"Against job description: {job_path}")
    
    # Verify files exist
    if not os.path.exists(resume_path):
        print(f"Error: Resume file {resume_path} not found")
        sys.exit(1)
    
    if not os.path.exists(job_path):
        print(f"Error: Job description file {job_path} not found")
        sys.exit(1)
    
    # Load job details from JSON
    try:
        with open(job_path, 'r') as file:
            job_details = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {job_path}")
        sys.exit(1)
    
    # Prepare API request
    api_url = f"{url.rstrip('/')}/analyze"
    print(f"Sending request to: {api_url}")
    
    try:
        # Prepare the form data with the resume file and job details
        with open(resume_path, 'rb') as resume_file:
            files = {'resume': (os.path.basename(resume_path), resume_file)}
            data = {'job': json.dumps(job_details)}
            
            # Send the request to the API
            response = requests.post(api_url, files=files, data=data, timeout=60)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                # Print formatted output for readability
                print(f"\n✅ Analysis completed in {processing_time:.2f} seconds")
                print(f"API processing time: {result.get('processing_time', 'N/A')} seconds")
                print(f"Match percentage: {result.get('result', {}).get('match_percentage', 'N/A')}")
                print(f"Recommendation: {result.get('result', {}).get('recommendation', 'N/A')}")
                
                # Return the full result
                return result
            else:
                print(f"❌ Error: API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                sys.exit(1)
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making API request: {e}")
        sys.exit(1)

def check_health(url):
    """Check the health of the API"""
    print(f"Checking API health at {url}/health")
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ API is healthy: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking API health: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Client for Resume Scorer API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API endpoint URL (default: http://localhost:8000)")
    parser.add_argument("--resume", default="src/data/sample_resume.pdf", 
                       help="Path to resume file")
    parser.add_argument("--job", default="src/data/sample_job.json", 
                       help="Path to job description JSON file")
    parser.add_argument("--health-check", action="store_true", 
                       help="Check API health only")
    parser.add_argument("--save-output", 
                       help="Path to save analysis results as JSON")
    
    args = parser.parse_args()
    
    # Check if API is running
    if not check_health(args.url):
        print("Exiting due to health check failure")
        sys.exit(1)
    
    # Exit if only health check was requested
    if args.health_check:
        sys.exit(0)
    
    # Analyze resume
    result = analyze_resume(args.url, args.resume, args.job)
    
    # Save output if requested
    if args.save_output and result:
        try:
            output_path = args.save_output
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Analysis results saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving output: {e}")
    
    # Return success/failure exit code
    if result and result.get('status') == 'success':
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 