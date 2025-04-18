import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_resume(resume_text, job_details):
    """
    Analyze a resume against job details using OpenAI's API
    
    Parameters:
    - resume_text: Extracted text from the resume
    - job_details: Dictionary containing job summary, duties, skills, and qualifications
    
    Returns:
    - Dictionary with analysis results
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Prepare the prompt
    prompt = f"""
    Analyze this resume against the job requirements and provide a detailed assessment.
    
    RESUME:
    {resume_text}
    
    JOB DETAILS:
    Summary: {job_details.get('summary', 'Not provided')}
    Key Duties: {job_details.get('duties', 'Not provided')}
    Essential Skills: {job_details.get('skills', 'Not provided')}
    Qualifications: {job_details.get('qualifications', 'Not provided')}
    
    Provide a JSON response with the following structure:
    {{
        "match_percentage": "Overall match percentage as an integer (0-100)",
        "skills_match": {{
            "matched_skills": ["List of matched skills"],
            "missing_skills": ["List of missing skills"],
            "match_ratio": "X/Y format (matched/total)"
        }},
        "experience": {{
            "required_years": "Required years of experience",
            "applicant_years": "Applicant's years of experience",
            "percentage_impact": "Percentage impact on overall score (e.g., -10%)"
        }},
        "education": {{
            "requirement": "Required education level",
            "applicant_education": "Applicant's education level",
            "assessment": "Whether it meets requirements"
        }},
        "certifications": {{
            "relevant_certs": ["List of relevant certifications"],
            "percentage_impact": "Percentage impact on overall score (e.g., +5%)"
        }},
        "keywords": {{
            "matched": "Number of matched keywords",
            "total": "Total number of important keywords",
            "match_ratio": "X/Y format"
        }},
        "recommendation": "Recommendation status (Strong Hire, Hire, Consider, Reject)"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert HR consultant specialized in resume analysis."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        analysis_result = json.loads(response.choices[0].message.content)
        return analysis_result
        
    except Exception as e:
        print(f"Error during API call: {e}")
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
    
    # Skills match
    skills = analysis['skills_match']
    result += f"Skills Match:\n{', '.join(skills['matched_skills'])} ({skills['match_ratio']})\n\n"
    
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
    
    # Recommendation
    rec = analysis['recommendation']
    emoji = "✅" if "hire" in rec.lower() else "❌"
    result += f"Recommendation:\n{rec} {emoji}"
    
    return result 