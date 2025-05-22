import os
from fpdf import FPDF
import requests
import json
import time

def create_test_resume():
    pdf = FPDF()
    pdf.add_page()
    
    # Add education section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Education", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Bachelor's degree in Computer Science", ln=True)
    pdf.cell(0, 10, "University of Technology", ln=True)
    pdf.cell(0, 10, "2018-2022", ln=True)
    
    # Add experience section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Experience", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Software Developer", ln=True)
    pdf.cell(0, 10, "Tech Company", ln=True)
    pdf.cell(0, 10, "2022-Present", ln=True)
    pdf.multi_cell(0, 10, "Developed web applications using Python and JavaScript. Worked on database optimization and API development.")
    
    # Add skills section
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Skills", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Python, JavaScript, SQL, Git, Docker", ln=True)
    
    # Save the PDF
    pdf_path = "test_resume.pdf"
    pdf.output(pdf_path)
    return pdf_path

def analyze_with_weights(resume_path, weights):
    # Job requirements
    job_details = {
        "required_skills": ["Python", "JavaScript", "SQL", "Docker", "AWS"],
        "required_experience": 2,
        "required_education": "Bachelor's degree",
        "job_title": "Software Developer"
    }
    
    # Prepare the request
    url = "http://localhost:8000/analyze"
    files = {"resume": open(resume_path, "rb")}
    data = {
        "job_details": json.dumps(job_details),
        "weights": json.dumps(weights)
    }
    
    # Send request
    response = requests.post(url, files=files, data=data)
    return response.json()

def main():
    # Create test resume
    resume_path = create_test_resume()
    
    # Test Case 1: Education weight highest (0.5)
    weights1 = {
        "education": 0.5,
        "skills": 0.3,
        "experience": 0.2
    }
    result1 = analyze_with_weights(resume_path, weights1)
    print("\nTest Case 1 (Education Weight Highest 0.5):")
    print(f"Match Percentage: {result1.get('match_percentage', 'N/A')}")
    print(f"Recommendation: {result1.get('recommendation', 'N/A')}")
    print("Skills Match:", result1.get('skills_match', {}).get('match_ratio', 'N/A'))
    print("Education Match:", result1.get('education', {}).get('assessment', 'N/A'))
    print("Experience Match:", f"{result1.get('experience', {}).get('applicant_years', 'N/A')} years vs {result1.get('experience', {}).get('required_years', 'N/A')} required")
    
    # Test Case 2: Skills weight highest (0.5)
    weights2 = {
        "skills": 0.5,
        "education": 0.3,
        "experience": 0.2
    }
    result2 = analyze_with_weights(resume_path, weights2)
    print("\nTest Case 2 (Skills Weight Highest 0.5):")
    print(f"Match Percentage: {result2.get('match_percentage', 'N/A')}")
    print(f"Recommendation: {result2.get('recommendation', 'N/A')}")
    print("Skills Match:", result2.get('skills_match', {}).get('match_ratio', 'N/A'))
    print("Education Match:", result2.get('education', {}).get('assessment', 'N/A'))
    print("Experience Match:", f"{result2.get('experience', {}).get('applicant_years', 'N/A')} years vs {result2.get('experience', {}).get('required_years', 'N/A')} required")
    
    # Test Case 3: Experience weight highest (0.5)
    weights3 = {
        "experience": 0.5,
        "skills": 0.3,
        "education": 0.2
    }
    result3 = analyze_with_weights(resume_path, weights3)
    print("\nTest Case 3 (Experience Weight Highest 0.5):")
    print(f"Match Percentage: {result3.get('match_percentage', 'N/A')}")
    print(f"Recommendation: {result3.get('recommendation', 'N/A')}")
    print("Skills Match:", result3.get('skills_match', {}).get('match_ratio', 'N/A'))
    print("Education Match:", result3.get('education', {}).get('assessment', 'N/A'))
    print("Experience Match:", f"{result3.get('experience', {}).get('applicant_years', 'N/A')} years vs {result3.get('experience', {}).get('required_years', 'N/A')} required")
    
    # Clean up
    os.remove(resume_path)

if __name__ == "__main__":
    main() 