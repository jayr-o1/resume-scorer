#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.pdf_extractor import extract_text_from_pdf
from utils.analyzer_local import analyze_resume, format_analysis_result

def main():
    parser = argparse.ArgumentParser(description="AI Resume Scorer - Command Line Interface (Local Version)")
    parser.add_argument("pdf_path", help="Path to the resume PDF file")
    parser.add_argument("--job_file", help="Path to a JSON file containing job details")
    parser.add_argument("--summary", help="Job summary text")
    parser.add_argument("--duties", help="Key duties text")
    parser.add_argument("--skills", help="Essential skills text")
    parser.add_argument("--qualifications", help="Qualifications text")
    parser.add_argument("--output", help="Output file to save the analysis results")
    
    args = parser.parse_args()
    
    # Check if the PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: The PDF file '{args.pdf_path}' does not exist.")
        return
    
    # Extract text from the resume
    print("Extracting text from resume...")
    resume_text = extract_text_from_pdf(args.pdf_path)
    
    if not resume_text:
        print("Error: Failed to extract text from the resume.")
        return
    
    # Get job details
    job_details = {}
    
    if args.job_file:
        try:
            with open(args.job_file, 'r') as f:
                job_details = json.load(f)
        except Exception as e:
            print(f"Error loading job details from file: {e}")
            return
    else:
        if args.summary:
            job_details["summary"] = args.summary
        if args.duties:
            job_details["duties"] = args.duties
        if args.skills:
            job_details["skills"] = args.skills
        if args.qualifications:
            job_details["qualifications"] = args.qualifications
    
    if not job_details:
        print("Error: No job details provided. Use --job_file or provide details with --summary, --duties, --skills, --qualifications")
        return
    
    # Analyze the resume
    print("Analyzing resume against job details (using local model)...")
    analysis = analyze_resume(resume_text, job_details)
    
    # Format and display the results
    formatted_results = format_analysis_result(analysis)
    print("\n" + "="*60 + "\n")
    print(formatted_results)
    print("\n" + "="*60 + "\n")
    
    # Save to output file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(formatted_results)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    main() 